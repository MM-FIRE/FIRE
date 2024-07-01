from llava.eval.feedback_chat import generate, get_args, get_score, get_feedback
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import image_parser, load_images
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import json
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action="once")
import os
from llava.conversation import conv_templates
from tqdm import tqdm
import random
from argparse import ArgumentParser
from loguru import logger

def prompt(conv_mode, text):
    if conv_mode == "llava_v1_teacher_feedback":
        return text
    
    if conv_mode == "llama_v3_teacher":
        return text
    
    template = """{text}
Can you provide me a feedback based on the groundtruth and the answer?
The feedback is formatted as:
'''
Score: <compare the human's answer with the groundtruth answer in terms of accuracy, relevance, helpfulness, and level of detail, and provide an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.>
Feedback: <provide feedback on the human's answer. Do NOT directly tell the groundtruth answer. The feedback should identify which parts of the human's answer are incorrect, what is missing in the human's answer, and how to improve the human's answer.>"""
    return template.format(text=text)


def eval_feedback_teacher(
    model_path,
    model_base,
    dataset_path,
    conv_mode = "llava_v1_teacher_feedback"
):
    template = conv_templates[conv_mode]
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    image_root = "data/FeedbackReflection/image/"
    args = get_args(model_path, model_base, None, None, conv_mode)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, teacher_context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        load_4bit=args.load_4bit
    )  
    
    results = []
    for item in tqdm(dataset):
        #if random.uniform(0, 1) > 0.01: # subsample
        #    continue
        image_file = os.path.join(image_root, item["image"])
        history = []
        conversations = item['conversations']
        score_predicts = []
        feedback_predicts = []
        score_labels = []
        feedback_labels = []
        
        for turn_idx, turn in enumerate(conversations):
            if turn["from"] == "human":
                if turn_idx == 0:
                    history.append((template.roles[0], prompt(conv_mode, turn["value"])))
                else:
                    history.append((template.roles[0], turn["value"]))
                try:
                    args = get_args(model_path, model_base, None, image_file, conv_mode)
                    output = generate(args, tokenizer, model, image_processor, chat_history=history)
                    logger.info("predict={}", output)
                    score_predict = get_score(output)
                    feedback_predict = get_feedback(output)
                    
                    score_predicts.append(score_predict)
                    feedback_predicts.append(feedback_predict)
                    
                except Exception as e:
                    print("parse failed", repr(e))
                    score_predicts.append(-1)
                    feedback_predicts.append("")
            else:
                history.append((template.roles[1], turn["value"]))
                try:
                    score_label = get_score(turn["value"])
                    feedback_label = get_feedback(turn["value"])
                    score_labels.append(score_label)
                    feedback_labels.append(feedback_label)
                    
                except Exception as e:
                    print("parse failed", repr(e))
                    score_labels.append(-1)
        results.append(
            {
                "id": item['id'],
                "score_predictions": score_predicts,
                "score_labels": score_labels,
                "feedback_predictions": feedback_predicts,
                "feedback_labels": feedback_labels,
                "raw": item["raw"] if "raw" in item else None
            }
        )
        
    job_id = re.findall(r'\d+', model_path)[-1]
    output_path = f"data/eval/run_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"{conv_mode}_{job_id}.json")
    with open(output_path, "w") as f:
        json.dump(results, f)
    
    # analyze("results.json")

def eval_feedback_student(
    model_path,
    model_base,
    dataset_path,
    conv_mode = "llava_v1_student_feedback"
):
    template = conv_templates[conv_mode]
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    image_root = "data/FeedbackReflection/image/"
    args = get_args(model_path, model_base, None, None, conv_mode)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, teacher_context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        load_4bit=args.load_4bit
    )  
    
    results = []
    for item in tqdm(dataset):
        #if random.uniform(0, 1) > 0.01: # subsample
        #    continue
        image_file = os.path.join(image_root, item["image"])
        history = []
        conversations = item['conversations']
        
        answer_predicts = []
        answer_labels = []
        
        for turn_idx, turn in enumerate(conversations):
            # print("history", history)
            if turn["from"] == "human":
                history.append((template.roles[0], turn["value"]))
            else:
                try:
                    args = get_args(model_path, model_base, None, image_file, conv_mode)
                    output = generate(args, tokenizer, model, image_processor, chat_history=history)
                    print("prediction", output)
                    answer_predicts.append(output)
                    
                except Exception as e:
                    print("parse failed", repr(e))
                    answer_predicts.append("")
                    exit()
                history.append((template.roles[1], turn["value"]))
                answer_labels.append(turn["value"])
        print("pred and labels", answer_predicts, answer_labels)        
        results.append(
            {
                "id": item['id'],
                "predicts": answer_predicts,
                "labels": answer_labels,
                "raw": item["raw"] if "raw" in item else None
            }
        )
    
    job_id = re.findall(r'\d+', model_path)[-1]
    output_path = f"data/eval/run_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"{conv_mode}_{job_id}.json")
    with open(output_path, "w") as f:
        json.dump(results, f)

from datetime import datetime               
import numpy as np
import re

def analyze(file_name):
    with open(file_name, "r") as f:
        results = json.load(f)
    
    preds = []
    labels = []
    for item in tqdm(results):
        pred = item["predictions"]
        label = item["labels"]
        preds.extend(pred)
        labels.extend(label)
    preds = np.array(preds)
    labels = np.array(labels)
    
    print("MSE", np.mean((preds-labels)**2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--dataset-path", default="data/FeedbackReflection/json/test/merge_processed_teacher_test.json")
    parser.add_argument("--conv-mode", choices=[
        'llava_v1', 
        'llava_v1_teacher_feedback', 
        "llava_v1_student_feedback",
        "llama_v3",
        "llama_v3_original"
        "llama_v3_student",
        "llama_v3_teacher"])
    parser.add_argument("--mode", 
                        choices=["student", "teacher"])
    args = parser.parse_args()
    if args.mode == "student":
        eval_feedback_student(
            args.model_path,
            args.model_base,
            args.dataset_path,
            conv_mode=args.conv_mode
        )
    else:
        eval_feedback_teacher(
            args.model_path,
            args.model_base,
            args.dataset_path,
            conv_mode=args.conv_mode
        )