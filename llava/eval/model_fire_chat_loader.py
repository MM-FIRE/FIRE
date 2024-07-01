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
from llava.eval.feedback_chat import get_score, get_feedback

from PIL import Image
import math
import time
from loguru import logger
from collections import OrderedDict
import copy
def get_inputs(item):
    question = item["raw"]["question"]["value"]
    if DEFAULT_IMAGE_TOKEN not in question:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
    groundtruth = item["raw"]["groundtruth"]["value"]
    image_file = item["image"]
    if "mathverse" in image_file:
        image_file = image_file.replace("mathverse", "mathverse/images")
    image_file = os.path.join("data/FIRE-test", image_file)
    return question, image_file, groundtruth

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, model_config, mode):
        logger.info("dataset path = {}", data_path)
        # identify is student turn or teacher turn
        
        with open(data_path, "r") as f:
            self.list_data = json.load(f)
        logger.info("original dataset len {}", len(self.list_data))
        self.list_data = [i for i in self.list_data if 'image' in i]
        self.list_data_map = OrderedDict()
        for item in self.list_data:
            self.list_data_map[item["id"]] = copy.deepcopy(item)
        
        # do filter
        keep_item = []
        for item in self.list_data:
            # scores >= 8 can stop; score == 0 means the model did not follow the instruction
            if "scores" in item and len(item["scores"]) > 0 and (item["scores"][-1] >= 8 or item["scores"][-1] == 0):
                continue
            
            # filter non-increasing sequence
            if "scores" in item and len(item["scores"]) >= 2 and item["scores"][-1] <= item["scores"][-2]:
                continue
            keep_item.append(item)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        # using padding as evaluation
        self.model_config.image_aspect_ratio = "pad"
        self.mode = mode
        
        self.list_data = keep_item
        for item in self.list_data:
            if self.mode == "student":
                question, image_file, groundtruth, prompt, history = self.get_student_inputs(item)
                
            else:
                question, image_file, groundtruth, prompt, history = self.get_teacher_inputs(item)
            token_len = len(tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"))
            # token_len = len(tokenizer(prompt).input_ids)
            item["token_len"] = token_len
            
        self.list_data = sorted(self.list_data, key=lambda x: x["token_len"])
        
        logger.info("after filter dataset len {}", len(self.list_data))
        logger.info("current mode {}", mode)
            
    @staticmethod
    def check_mode(dataset_path):
        with open(dataset_path, "r") as f:
            list_data = json.load(f)
            
        list_data = [i for i in list_data if 'image' in i]
        keep_item = []
        for item in list_data:
            # scores >= 8 can stop; score == 0 means the model did not follow the instruction
            if "scores" in item and len(item["scores"]) > 0 and (item["scores"][-1] >= 8 or item["scores"][-1] == 0):
                continue
             # filter non-increasing sequence
            if "scores" in item and len(item["scores"]) >= 2 and item["scores"][-1] <= item["scores"][-2]:
                continue
            keep_item.append(item)
            
        list_data = keep_item
        # find score max length
        # peek = list_data[0]
        score_len = 0
        peek = None
        peek_idx = 0
        for idx, item in enumerate(list_data):
            if "scores" not in item:
                continue
            
            if len(item["scores"]) > score_len:
                peek_idx = idx
            score_len = max(score_len, len(item["scores"]))
        
        peek = list_data[peek_idx]
        logger.info("score_len={},{};student_history={}", score_len, len(peek["scores"]), len(peek["student_history"]))
        if "student_history" not in peek:
            mode = "student"
        elif "scores" not in peek:
            mode = "teacher"
        elif score_len * 2 == len(peek["student_history"]):
            mode = "student"
        elif score_len * 2 < len(peek["student_history"]):
            mode = "teacher"
        else:
            raise ValueError("unhandled condtion for mode identification")
        logger.info("mode={}", mode)
        # exit()
        return mode
    
    def get_mode(self):
        return self.mode
    
    def get_student_inputs(self, item):
        question, image_file, groundtruth = get_inputs(item)
        conv = conv_templates[args.conv_mode].copy()
        history = copy.deepcopy(item["student_history"]) if "student_history" in item else []
        if len(history) == 0:
            history.append(
                {"from": "human", "value": question}
            )
        else:
            assert "teacher_history" in item
            teacher_history = item["teacher_history"]
            try:
                feedback = get_feedback(teacher_history[-1]["value"])
            except Exception as e:
                feedback = teacher_history[-1]["value"]
                logger.error("get feedback failed, set as original {}", feedback)
            history.append({"from": "human", "value": feedback})
        
        for turn in history:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        
        conv.append_message(conv.roles[1], None)
        # item["student_history"] = history
        return question, image_file, groundtruth, conv.get_prompt(), history
    
    def get_teacher_inputs(self, item):
        question, image_file, groundtruth = get_inputs(item)
        conv = conv_templates[args.conv_mode].copy()
        history = []
        teacher_system_prompt = "Please compare my answer with the groundtruth answer and provide helpful, detailed, and polite feedback to help me improve my answer.\nFormulate the feedback as:\n'''\nScore: <compare my answer with the groundtruth answer in terms of accuracy, relevance, helpfulness, and level of detail, and provide an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.>\nFeedback: <provide feedback on my answer. Do NOT directly tell the groundtruth answer. The feedback should identify which parts of my answer are incorrect, what is missing in my answer, and how to improve my answer.>\n'''"
        question_template = "Question: {question}\nGroundtruth: {groundtruth}"
        if len(history) == 0:
            history.append(
                {"from": "human", "value": question_template.format(question=question, groundtruth=groundtruth)}
            )
            history.append(
                {"from": "human", "value": teacher_system_prompt}
            )
          
        history.append(
            {"from": "human", "value": item["student_history"][-1]["value"]}
        )
        
        
        for turn in history:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        
        conv.append_message(conv.roles[1], None)
        # item["teacher_history"] = history
        return question, image_file, groundtruth, conv.get_prompt(), history
    
    def __getitem__(self, index):
        item = self.list_data[index]
        if self.mode == "student":
            question, image_file, groundtruth, prompt, history = self.get_student_inputs(item)
            item["student_history"] = history
            
        else:
            question, image_file, groundtruth, prompt, history = self.get_teacher_inputs(item)
            item["teacher_history"] = history
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        # logger.info("image: {}; {}", image_tensor.shape, image_file)
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
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output_ids = model.generate(
            batch_input_ids,
            images=batch_input_images.to(
                dtype=torch.float16, device=model.device),
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
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
    # logger.info("image_tensors = {}, {}", len(image_tensors), type(image_tensors))
    image_tensors = torch.stack(image_tensors, dim=0)
    return prompts, image_tensors, image_sizes, items

# DataLoader
def create_data_loader(data_path, tokenizer, image_processor, model_config, mode, batch_size=32, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(data_path, tokenizer, image_processor, model_config, mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader, dataset

# Left padding to max length
def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype, device=sequence.device), sequence])

def eval_model(args):
    # Model
    disable_torch_init()
    
    mode = CustomDataset.check_mode(args.dataset_path)
    args.mode = mode
    logger.info("process mode = {}", mode)
    if mode == "student":
        model_path = args.model_path_student
        model_base = args.model_base_student
        conv_mode = args.conv_mode_student
    elif mode == "teacher":
        model_path = args.model_path_teacher
        model_base = args.model_base_teacher
        conv_mode = args.conv_mode_teacher
    else:
        raise ValueError("Unhandled case for model paht base and conv mode")
    
    args.conv_mode = conv_mode
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)   
    
    tokenizer.add_special_tokens(dict(pad_token="<pad>"))
    data_loader, dataset = create_data_loader(
        args.dataset_path, 
        tokenizer, 
        image_processor, 
        model.config,
        mode=args.mode,
        batch_size=args.batch_size,
    )

    output_chunk = []
    loop  = 0
    for datapoint in tqdm(data_loader, total=len(data_loader)):
        torch.cuda.empty_cache()
        # logger.info("datapoint {}", datapoint)
        prompts, image_tensor, image_sizes, source_data = datapoint
        batch_dataset = {}
        
        batch_input_ids = []
        batch_image_tensor = []
        batch_prompts = []
        batch_source_data = []
        for idx, prompt in enumerate(prompts):
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            token_len = len(input_ids)
            if token_len not in batch_dataset:
                batch_dataset[token_len] = {
                    "batch_input_ids": [],
                    "batch_image_tensor": [],
                    "batch_prompts": [],
                    "batch_source_data": [],
                    "batch_image_sizes": []
                }
            # logger.info("token_len = {}", len(input_ids))
            input_ids = input_ids.to(device=model.device)
            batch_dataset[token_len]["batch_input_ids"].append(input_ids)
            batch_dataset[token_len]["batch_image_tensor"].append(image_tensor[idx])
            batch_dataset[token_len]["batch_prompts"].append(prompt)
            batch_dataset[token_len]["batch_source_data"].append(source_data[idx])
            batch_dataset[token_len]["batch_image_sizes"].append(image_sizes[idx])
        
        for token_len, batch in batch_dataset.items():
            batch_input_ids = batch["batch_input_ids"]
            batch_image_tensor = batch["batch_image_tensor"]
            #batch_input_ids = [pad_sequence_to_max_length(seq, max_len, tokenizer.pad_token_id) for seq in batch_input_ids]
            batch_input_ids = torch.stack(batch_input_ids)
            
            batch_image_tensor = torch.stack(batch_image_tensor).to(dtype=torch.float16, device=model.device)
            batch_prompts = batch["batch_prompts"]
            batch_source_data = batch["batch_source_data"]
            batch_image_sizes = batch["batch_image_sizes"]
            # print(batch_input_ids.shape, image_tensor.shape)
            
            with torch.inference_mode():
                outputs = batch_generate(
                    tokenizer,
                    model,
                    batch_image_sizes,
                    batch_input_ids,
                    batch_image_tensor,
                    max_new_tokens=args.max_new_tokens
                )
            
            if args.mode == "student":
                for source, output in zip(batch_source_data, outputs):
                    source["student_history"].append(
                        {
                            "from": "gpt",
                            "value": output
                        }
                    )
                    output_chunk.append(source)
            elif args.mode == "teacher":
                
                for bathc_item_id, (prompt, source, output) in enumerate(zip(batch_prompts, batch_source_data, outputs)):
                    #logger.info("item {} get teacher input={}",  bathc_item_id, prompt)
                    #logger.info("item {} get teacher output={}", bathc_item_id, output)
                    try:
                        score = get_score(output)
                    except Exception as e:
                        score = 0
                        logger.error("item {} get score failed, set score as 0", bathc_item_id)
                    source["teacher_history"].append(
                        {
                            "from": "gpt",
                            "value": output
                        }
                    )
                    
                    if "scores" not in source:
                        source["scores"] = []
                    
                    source["scores"].append(score)
                    output_chunk.append(source)
    
    # do merge
    list_data_map = dataset.list_data_map
    for item in output_chunk:
        if item["id"] in list_data_map:
            list_data_map[item["id"]] = item
    
    outputs_to_file = []
    for k, v in list_data_map.items():
        outputs_to_file.append(v)
    
    output_path = args.dataset_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("write to {} len= {}", output_path, len(outputs_to_file))
    with open(output_path, "w") as f:
        json.dump(outputs_to_file, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path-student", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base-student", type=str, default=None)
    parser.add_argument("--model-path-teacher", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base-teacher", type=str, default=None)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--conv-mode-student", type=str, default="llava_v1_student_feedback")
    parser.add_argument("--conv-mode-teacher", type=str, default="llava_v1_student_feedback")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--mode", choices=["student", "teacher"], default="student")
    
    args = parser.parse_args()

    eval_model(args)
