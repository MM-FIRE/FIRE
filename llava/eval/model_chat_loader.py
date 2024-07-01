import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import time
from loguru import logger

def get_inputs(item):
    source = item["image"].split("/")[0]
    image_path_projection = {
        "coco": "data/coco/train2017",
        "gqa": "data/gqa/images",
        "ocr_vqa": "data/ocrvqa/images",
        "textvqa": "data/textvqa/train_images",
        "vg": "data/vg",
        "allava_vflan": "data/allava_vflan/images/images_191task_1k",
        "wikiart": "data/wikiart/images",
        "web-landmark": "data/web-landmark/images/",
        "geoqa+": "data/geoqa+/images",
        "docvqa": "data/docvqa/train/documents/",
        "ai2d": "data/ai2d/images",
        "synthdog-en": "data/synthdog-en/images",
        "share_textvqa": "data/share_textvqa/images",
        "llava": "data/llava/llava_pretrain/images",
        "sam": "data/sam/images",
        "dvqa": "data/dvqa/images",
        "web-celebrity": "data/web-celebrity/images",
        "chartqa": "data/chartqa/train/png"
    }
    image_file = item["image"].split("/")[-1]
    if source == "vg":
        # vg is in two dataset
        image_file = item["image"].split("/")[1:]
        image_file = "/".join(image_file)
    elif source == "llava":
        image_file = item["image"].split("/")[-2:]
        image_file = "/".join(image_file)
    question = item["conversations"][0]["value"]
    groundtruth = item["conversations"][1]["value"]
    image_file = os.path.join(image_path_projection[source], image_file)
    return question, image_file, groundtruth

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, model_config,
                 mode="student"):
        logger.info("dataset path = {}", data_path)
        with open(data_path, "r") as f:
            self.list_data = json.load(f)
        
        self.list_data = [i for i in self.list_data if 'image' in i]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.mode = mode
    
    def get_student_inputs(self, item):
        question, image_file, groundtruth = get_inputs(item)
        conv = conv_templates[args.conv_mode].copy()
        history = item["student_history"] if "student_history" in item else []
        if len(history) == 0:
            history.append(
                {"from": "human", "value": question}
            )
        else:
            assert "teacher_history" in item
            teacher_history = item["teacher_history"]
            feedback = teacher_history[-1]["value"]
            history.append({"from": "human", "value": feedback})
        
        for turn in history:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        
        conv.append_message(conv.roles[1], None)
        item["student_history"] = history
        return question, image_file, groundtruth, conv.get_prompt(), history
    
    def get_teacher_inputs(self, item):
        question, image_file, groundtruth = get_inputs(item)
        conv = conv_templates[args.conv_mode].copy()
        history = item["teacher_history"] if "teacher" in item else []
        teacher_system_prompt = "Please compares my answer with the groundtruth answer and provides helpful, detailed, and polite feedback to help me improve my answer.\nFormulate the feedback as:\n'''\nScore: <compare my answer with the groundtruth answer in terms of accuracy, relevance, helpfulness, and level of detail, and provide an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.>\nFeedback: <provide feedback on my answer. Do NOT directly tell the groundtruth answer. The feedback should identify which parts of my answer are incorrect, what is missing in my answer, and how to improve my answer.>\n'''"
        question_template = "Question: {question}\nGroundtruth: {groundtruth}"
        if len(history) == 0:
            history.append(
                {"from": "human", "value": teacher_system_prompt}
            )
            
            history.append(
                {"from": "human", "value": question_template.format(question=question, groundtruth=groundtruth)}
            )
          
        history.append(
            {"from": "human", "value": item["student_history"][-1]}
        )
        
        
        for turn in history:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        
        conv.append_message(conv.roles[1], None)
        item["teacher_history"] = history
        return question, image_file, groundtruth, conv.get_prompt(), history
    
    def __getitem__(self, index):
        item = self.list_data[index]
        if self.mode == "student":
            question, image_file, groundtruth, prompt, history = self.get_student_inputs(item)
            
        else:
            raise NotImplementedError
        
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return prompt, image_tensor, image.size, item

    def __len__(self):
        return len(self.list_data)

def batch_generate(
    tokenizer, 
    model, 
    image_sizes,
    batch_input_ids,
    batch_input_images,
    max_new_tokens=256):
    current = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            batch_input_ids,
            images=batch_input_images.to(
                dtype=torch.float16, device=model.device, non_blocking=True),
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #print(outputs)
    logger.info("generate consume {} s, input_token_len {}, output_token_len {}", 
                round(time.time() - current, 2),
                batch_input_ids.shape,
                output_ids.shape)
    return outputs

def collate_fn(batch):
    prompts, image_tensors, image_sizes, items = zip(*batch)
    image_tensors = torch.stack(image_tensors, dim=0)
    return prompts, image_tensors, image_sizes, items

# DataLoader
def create_data_loader(data_path, tokenizer, image_processor, model_config, batch_size=32, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(data_path, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

# Left padding to max length
def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype, device=sequence.device), sequence])

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)   
    data_loader = create_data_loader(
        args.dataset_path, 
        tokenizer, 
        image_processor, 
        model.config,
        batch_size=args.batch_size)

    output_chunk = []
    for prompts, image_tensor, image_sizes, source_data in tqdm(data_loader, total=len(data_loader)):
        torch.cuda.empty_cache()
        batch_input_ids = []
        for prompt in prompts:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.to(device=model.device)
            batch_input_ids.append(input_ids)
        max_len = max([len(i) for i in batch_input_ids])
        batch_input_ids = [pad_sequence_to_max_length(seq, max_len, tokenizer.pad_token_id) for seq in batch_input_ids]
        batch_input_ids = torch.stack(batch_input_ids)
        image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)
        
        # print(batch_input_ids.shape, image_tensor.shape)
        
        with torch.inference_mode():
            outputs = batch_generate(
                tokenizer,
                model,
                image_sizes,
                batch_input_ids,
                image_tensor,
                max_new_tokens=args.max_new_tokens
            )
        
        if args.mode == "student":
            for source, output in zip(source_data, outputs):
                source["student_history"].append(
                    {
                        "from": "gpt",
                        "value": output
                    }
                )
                output_chunk.append(source)
        else:
            raise NotImplementedError
    
        # if len(output_chunk) > 256:
        #     break 
    output_path = "/".join(args.dataset_path.split("/")[:-2])
    output_name = args.dataset_path.split("/")[-1]
    output_path = os.path.join(output_path, "outputs")
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, output_name)
    print("write to", output_path)
    with open(output_path, "w") as f:
        json.dump(output_chunk, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1_student_feedback")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=["student", "teacher"], default="student")
    
    args = parser.parse_args()

    eval_model(args)
