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

import copy
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]  
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, model_config,
                 mode="student", num_chunks=1, chunk_id=0):
        logger.info("dataset path = {};num_chunks={}; chunk_id={}", data_path, num_chunks, chunk_id)
        with open(data_path, "r") as f:
            list_data = json.load(f)
        self.num_chunks = num_chunks
        self.chunk_id = chunk_id
        # clear dataset
        dataset = []
        id_set = set()
        with open("data/ignore_imgs.json", "r") as f:
            ignore_images = json.load(f)
       
        for item in list_data:
            if item["id"] in id_set:
                print('WARNING')
                print(item["id"], "duplicated")
            id_set.add(item["id"])
            if item["image"] in ignore_images:
                continue
            chat = item["conversations"]
            instruct = []
            for turn in chat:
                if turn["from"] == "human":
                    instruct.append(copy.deepcopy(turn))
                elif turn["from"] == "gpt":
                    instruct.append(copy.deepcopy(turn))
                    item_new = copy.deepcopy(item)
                    item_new["conversations"] = copy.deepcopy(instruct)
                    # print("append", len(item_new["conversations"]))
                    # time.sleep(10)
                    dataset.append(item_new)
                else:
                    raise ValueError("unknown role")
        self.list_data = dataset 
        if num_chunks > 1:
            self.list_data = get_chunk(self.list_data, num_chunks, chunk_id)
        logger.info("flat dataset {} -> {}", len(list_data), len(self.list_data))
                
        # self.list_data = [i for i in self.list_data if 'image' in i]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.mode = mode
    
    def __getitem__(self, index):
        item = self.list_data[index]
        image_path = "data/FIRE-test"
        image_file = item["image"]
        if "mathverse" in image_file:
            image_file = image_file.replace("mathverse", "mathverse/images")
        image_file = os.path.join(image_path, image_file)
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        conv = conv_templates[args.conv_mode].copy()
        chat = item["conversations"]
        for turn in chat[:-1]:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        
        assert chat[-1]["from"] == "gpt"
        conv.append_message(conv.roles[1], None)
        
        label = chat[-1]["value"]
        return conv.get_prompt(), image_tensor, image.size, label, item["id"]

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
    prompts, image_tensors, image_sizes, labels, item_ids = zip(*batch)
    image_tensors = torch.stack(image_tensors, dim=0)
    return prompts, image_tensors, image_sizes, labels, item_ids

# DataLoader
def create_data_loader(
    data_path, 
    tokenizer, 
    image_processor, 
    model_config, 
    batch_size=4, 
    num_workers=4,
    num_chunks=1,
    chunk_id=0
    ):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(data_path, tokenizer, image_processor, model_config, mode=args.mode, num_chunks=num_chunks, chunk_id=chunk_id)
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
        batch_size=args.batch_size,
        num_chunks=args.num_chunks,
        chunk_id=args.chunk_id)
    from collections import OrderedDict
    output_data = OrderedDict()
    for prompts, image_tensor, image_sizes, labels, item_ids in tqdm(data_loader, total=len(data_loader)):
        torch.cuda.empty_cache()
        batch_input_ids = []
        for prompt in prompts:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.to(device=model.device)
            batch_input_ids.append(input_ids)
        # batch_input_ids = [pad_sequence_to_max_length(seq, max_len, tokenizer.pad_token_id) for seq in batch_input_ids]
        batch_input_ids = torch.stack(batch_input_ids)
        image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)
        with torch.inference_mode():
            outputs = batch_generate(
                tokenizer,
                model,
                image_sizes,
                batch_input_ids,
                image_tensor,
                max_new_tokens=args.max_new_tokens
            )
        
        
        for item_id, label, pred in zip(item_ids, labels, outputs):
            if args.mode == "student":
                if item_id not in output_data:
                    output_data[item_id] = {
                        "id" : item_id,
                        "predicts": [],
                        "labels": []}
                output_data[item_id]["predicts"].append(pred)
                output_data[item_id]["labels"].append(label)
            else:
                if item_id not in output_data:
                    output_data[item_id] = {
                        "id" : item_id,
                        "score_predictions": [],
                        "score_labels": [],
                        "feedback_predictions": [],
                        "feedback_labels": []
                    }
                from llava.eval.feedback_chat import get_score
                print(pred, label)
                try:
                    pred_score = get_score(pred)
                except Exception as e:
                    logger.error("parse score failed {}", pred)
                    pred_score = -1
                
                try:
                    label_score = get_score(label)
                except Exception as e:
                    logger.error("parse score failed {}", pred)
                    label_score = -1
                
                output_data[item_id]["score_predictions"].append(pred_score)
                output_data[item_id]["score_labels"].append(label_score)
                output_data[item_id]["feedback_predictions"].append(pred)
                output_data[item_id]["feedback_labels"].append(label)
        #$if len(output_data) >= 10:
        #    break
    from datetime import datetime
    today = f"{datetime.now().strftime('%m%d')}"
    output_path = f"data/eval/run_{today}"
    if args.num_chunks == 1:
        output_path = os.path.join(output_path, args.mode, args.model_path + ".json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print("write to", output_path)
        write_dataset = []
        for k, v in output_data.items():
            write_dataset.append(v)
            
        with open(output_path, "w") as f:
            json.dump(write_dataset, f, indent=4, ensure_ascii=False)
    else:
        output_path = os.path.join("data/eval", args.mode, args.model_path, "chunk_" + str(args.chunk_id) + ".json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print("write to", output_path)
        write_dataset = []
        for k, v in output_data.items():
            write_dataset.append(v)
            
        with open(output_path, "w") as f:
            json.dump(write_dataset, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1_student_feedback")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=["student", "teacher"], default="student")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-id", type=int, default=0)
    args = parser.parse_args()
    eval_model(args)
