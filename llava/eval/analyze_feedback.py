import numpy as np
import json
from tqdm import tqdm
from llava.eval.capeval.bleu.bleu import Bleu
from llava.eval.capeval.cider.cider import Cider
from nltk.tokenize import word_tokenize
from collections import defaultdict
from argparse import ArgumentParser
import pandas as pd

def get_metrics():
    return {
        "bleu": Bleu(),
        "cider": Cider()
    }
    
def analyze_teacher(file_name):
    with open(file_name, "r") as f:
        results = json.load(f)
    
    preds_scores = []
    labels_scores = []
    preds_feedback = []
    labels_feedback = []
    with open("data/ignore_ids.json", "r") as f:
        ignore_ids = json.load(f) 
    for item in tqdm(results):
        if item["id"] in ignore_ids:
            continue
        
        pred_score = item["score_predictions"]
        label_score = item["score_labels"]
        pred_feedback = item["feedback_predictions"]
        label_feedback = item["feedback_labels"]
        preds_feedback.extend([[i] for i in pred_feedback])
        labels_feedback.extend([[i] for i in label_feedback])
        preds_scores.extend(pred_score)
        labels_scores.extend(label_score)
    
    preds_scores = np.array(preds_scores)
    labels_scores = np.array(labels_scores)
    diff = np.abs(preds_scores - labels_scores)
    diff[preds_scores==-1] = 10
    print("MAE", np.round(np.mean(diff),3))
    print("Failed to follow count", np.sum(preds_scores==-1))
    metrics = get_metrics()
    # exit()
    preds_feedback = OrderedDict(enumerate(preds_feedback))
    labels_feedback = OrderedDict(enumerate(labels_feedback))
    # print(preds_feedback)
    for k, v in metrics.items():
        score = v.compute_score(labels_feedback, preds_feedback)[1]
        score = np.array(score)
        if k == "bleu":
            print(k, np.round(score.mean(axis=1), 3))
        else:
            print(k, np.round(score.mean(),3))

def get_answer(text):
    print(text)
    prefix = "The answer is"
    start_idx = text.find(prefix)
    if start_idx == -1:
        raise ValueError(f"invalid input {text}")
    return text[start_idx+len(prefix):].strip()

from collections import OrderedDict
def analyze_student(file_name):
    with open(file_name, "r") as f:
        results = json.load(f)
    
    preds = []
    labels = []
    with open("data/ignore_ids.json", "r") as f:
        ignore_ids = json.load(f) 
    for item in tqdm(results):
        if item["id"] in ignore_ids:
            continue
        
        preds.extend([[i] for i in item["predicts"]])
        labels.extend([[i] for i in item["labels"]])
        
    # print(preds[:5], labels[:5])
    preds = OrderedDict(enumerate(preds))
    labels = OrderedDict(enumerate(labels))
    
    metrics = get_metrics()
    score_str = ""
    score_order = ["bleu", "cider"]
    for k in score_order:
        v = metrics[k]
        score = v.compute_score(labels, preds)[1]
        score = np.array(score)
        if k == "bleu":
            scores = np.round(score.mean(axis=1), 3)
            print(k, scores)
            score_str += ",".join([str(i) for i in scores])
        else:
            score = np.round(score.mean(),3)
            print(k, np.round(score.mean(),3))
            score_str += "," + str(score)
    import os
    file_root = file_name.split("/")[:-1]
    file_root = "/".join(file_root)
    output_file = os.path.join(file_root, "feedback_analyze.csv")
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["file", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "cider"]).to_csv(output_file, index=False)
    
    with open(output_file, "a+") as f:
        f.write(f"{file_name},{score_str}\n")
                
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["student", "teacher"], default="student")
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    if args.mode == "student":
        analyze_student(args.file)
    else:
        analyze_teacher(args.file)