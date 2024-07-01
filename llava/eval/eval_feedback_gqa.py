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
from llava.eval.utils import save
def eval(
    model_path,
    model_base,
    dataset_path,
    conv_mode = "llava_v1_student_feedback",
):
    template = conv_templates[conv_mode]
    import json
    with open("data/FeedbackReflection/image/gqa/questions1.2/testdev_balanced_questions.json", "r") as f:
        annotation = json.load(f)
    print(annotation["201307251"])
    with open(dataset_path, 'r') as json_file:
        json_list = list(json_file)
    
    dataset = [json.loads(i) for i in json_list]
    image_root = "data/FeedbackReflection/image/gqa/images"
    args = get_args(model_path, model_base, None, None, conv_mode)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        load_4bit=args.load_4bit
    )
    results = []
    
    for item in tqdm(dataset):
        question = item["text"]
        groundtruth = annotation[item['question_id']]["answer"]
        image_file = os.path.join(image_root, item["image"])
        # print(question, groundtruth, image_file)
        args = get_args(model_path, model_base, question, image_file, conv_mode)
        history = [(template.roles[0], question)]
        output = generate(args, tokenizer, model, image_processor, history)
        print("prediction", output)
        results.append(
            {
                "image": item["image"],
                "question" : question,
                "groundtruth": groundtruth,
                "prediction": output,
                "question_id": item["question_id"]
            }
        )
        if len(results) == 100:
            break
    correct = 0
    for result in results:
        if result["prediction"].lower() == result["groundtruth"].lower():
            correct += 1
    
    accuracy = round(correct / len(results), 3)
    results = {
        "accuracy": accuracy,
        "predictions": results
    }
    print("accuracy", accuracy)
    save(model_path, conv_mode, results, prefix="gqa")
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--dataset-path", default="playground/data/gqa/llava_gqa_testdev_balanced.jsonl")
    args = parser.parse_args()
    eval(
        args.model_path,
        args.model_base,
        args.dataset_path,
    )