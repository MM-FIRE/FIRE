import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.utils import disable_torch_init
from PIL import Image

import re
from dataclasses import dataclass
import re
from loguru import logger
from typing import List, Any
import time
import torch

@dataclass
class StudentOutput:
    answer: str

@dataclass
class TeacherOutput:
    feedback_raw: str
    feedback: str
    score: float
    
def get_args(model_path, model_base, prompt, image_file, conv_mode = None):
    #print(model_path, model_base)
    return type('Args', (), {
        "model_path": model_path,
        "model_base": model_base,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": conv_mode,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "load_4bit": False
    })()

def generate(args, tokenizer, model, image_processor, chat_history=[]):
    current = time.time()
    # using this conv mode
    conv_mode = "llava_v1"
   
    # print('conv mode', conv_mode)
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        '''print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )'''
        pass
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    for turn_id, history in enumerate(chat_history):
        msg = history[1]
        if turn_id == 0 and "<image>" not in msg and args.image_file is not None:
            msg = "<image>\n" + msg
        conv.append_message(history[0], msg)
    # add completion head
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # logger.info("prompt={}", prompt)
    image = Image.open(args.image_file).convert("RGB")
    image_sizes = [image.size]
    images_tensor = process_images([image],
        image_processor,
        model.config
    )[0]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.to(device=model.device, non_blocking=True)
    input_ids = input_ids.unsqueeze(dim=0)
    images_tensor = images_tensor.unsqueeze(dim=0)
    # print(input_ids.shape, images_tensor.shape)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor.to(
                dtype=torch.float16, device=model.device, non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #print(outputs)
    logger.info("generate consume {} s, input_token_len {}, output_token_len {}", 
                round(time.time() - current, 2),
                len(input_ids[0]),
                len(output_ids[0])
    )
    return outputs

import torch
def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype, device=sequence.device), sequence])
    
def batch_generate(
    tokenizer, 
    model, 
    image_processor, 
    input_prompts,
    input_images):
    current = time.time()
    # using this conv mode
    image_sizes = [image.size for image in input_images]
    images_tensor = process_images(input_images,
        image_processor,
        model.config
    )
    batch_input_ids = []
    
    for prompt in input_prompts:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        # print("len of input_ids", len(input_ids), prompt)
        input_ids = input_ids.to(device=model.device, non_blocking=True)
        batch_input_ids.append(input_ids)
    max_len = max([len(i) for i in batch_input_ids])
    # left or right padding?
    # batch_input_ids = torch.nn.utils.rnn.pad_sequence(
    #         batch_input_ids,
    #         batch_first=True,
    #         padding_value=tokenizer.pad_token_id)
    batch_input_ids = [pad_sequence_to_max_length(seq, max_len, tokenizer.pad_token_id) for seq in batch_input_ids]
    batch_input_ids = torch.stack(batch_input_ids) 
    # print(batch_input_ids.shape, images_tensor.shape)
    with torch.inference_mode():
        output_ids = model.generate(
            batch_input_ids,
            images=images_tensor.to(
                dtype=torch.float16, device=model.device, non_blocking=True),
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    return outputs



class Chat():
    def __init__(
        self,
        student_model_path,
        student_model_base,
        teacher_model_path,
        teacher_model_base,
        student_conv_mode="llava_v1_student_feedback",
        teacher_conv_mode="llava_v1_teacher_feedback"):
        prompt, image_file = None, None # placeholder not used
        # Model
        disable_torch_init()
        # init student
        student_args = get_args(student_model_path, student_model_base, prompt, image_file)
        student_model_name = get_model_name_from_path(student_args.model_path)
        student_tokenizer, student_model, student_image_processor, student_context_len = load_pretrained_model(
            student_args.model_path,
            student_args.model_base,
            student_model_name,
            load_4bit=student_args.load_4bit,
            use_flash_attn=True
        )
        logger.info("compile student and teacher model")
        student_model.generation_config.cache_implementation = "static"
        student_model = torch.compile(student_model, mode="reduce-overhead", fullgraph=True)
        # init teacher
        teacher_args = get_args(teacher_model_path, teacher_model_base, prompt, image_file)

        teacher_model_name = get_model_name_from_path(teacher_args.model_path)
        teacher_tokenizer, teacher_model, teacher_image_processor, teacher_context_len = load_pretrained_model(
            teacher_args.model_path,
            teacher_args.model_base,
            teacher_model_name,
            load_4bit=teacher_args.load_4bit,
            use_flash_attn=True
        )
        teacher_model.generation_config.cache_implementation = "static"
        teacher_model = torch.compile(teacher_model, mode="reduce-overhead", fullgraph=True)
        
        self.student_tokenizer = student_tokenizer
        self.student_model = student_model
        self.student_image_processor = student_image_processor
        self.student_context_len = student_context_len
        self.student_model_path = student_model_path
        self.student_model_base = student_model_base
        self.student_model_name = student_model_name
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_model = teacher_model
        self.teacher_image_processor = teacher_image_processor
        self.teacher_context_len = teacher_context_len
        self.teacher_model_path = teacher_model_path
        self.teacher_model_base = teacher_model_base
        self.teacher_model_name = teacher_model_name
        
        self.student_history = []
        self.teacher_history = []
        
        # set template
        self.student_conv_mode = student_conv_mode
        self.teacher_conv_mode = teacher_conv_mode
        self.student_conv_template = conv_templates[self.student_conv_mode].copy()
        self.teacher_conv_template = conv_templates[self.teacher_conv_mode].copy()
        
    def clear_history(self):
        self.student_history = []
        self.teacher_history = []
    
    def should_stop(self, score: float) -> bool:
        if abs(score - 10) < 1e-4:
            return True
        return False
    
    def invoke(self, question, image_file, groundtruth):        
        student_prompt = question
        student_image_file = image_file
        student_args = get_args(
            self.student_model_path, 
            self.student_model_base, 
            student_prompt, 
            student_image_file,
            conv_mode=self.student_conv_mode
        )
        
        if question is not None:
            self.student_history.append(
                (self.student_conv_template.roles[0], student_prompt)
            )
        student_response = generate(
            student_args, 
            self.student_tokenizer, 
            self.student_model, 
            self.student_image_processor, 
            self.student_history)
        
        teacher_prompt_template = """{question}\nGroundtruth: {groundtruth}\n{student_response}\n"""
        
        teacher_image_file = image_file
        
        # first turn teacher will have template
        teacher_prompt = teacher_prompt_template.format(question=student_prompt, groundtruth=groundtruth, student_response=student_response)
        if question is not None:
            self.teacher_history.append(
                (self.teacher_conv_template.roles[0], teacher_prompt)
            )
        else:
            # add student's response, teacher will generate feedback
            self.teacher_history.append(
                (self.teacher_conv_template.roles[0], student_response)
            )
            
        teacher_args = get_args(
            self.teacher_model_path, 
            self.teacher_model_base, 
            teacher_prompt, 
            teacher_image_file,
            conv_mode=self.teacher_conv_mode
        )
        teacher_feedback = generate(
            teacher_args, 
            self.teacher_tokenizer, 
            self.teacher_model, 
            self.teacher_image_processor, 
            self.teacher_history)
        # logger.info("teacher feedback {}", teacher_feedback)
        student_response = StudentOutput(student_response)
        teacher_feedback = TeacherOutput(
            teacher_feedback,
            self.get_feedback_content(teacher_feedback),
            self.get_score(teacher_feedback),
        )
        # add student response
        self.student_history.append((self.student_conv_template.roles[1], student_response.answer))
        # add teacher's feedback, and student will generate answer in next turn
        self.student_history.append((self.student_conv_template.roles[0], teacher_feedback.feedback))
        
        # add teacher's feedback
        self.teacher_history.append((self.teacher_conv_template.roles[1], teacher_feedback.feedback_raw))
        return student_response, teacher_feedback
    
    def chat(self, question, image_file, groundtruth, num_round=3):
        scores = []
        if IMAGE_PLACEHOLDER not in question and DEFAULT_IMAGE_TOKEN not in question:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        
        #print("after processed", question)
        student_response, teacher_feedback = self.invoke(question, image_file, groundtruth)
        
        if self.should_stop(teacher_feedback.score):
            scores.append(teacher_feedback.score)
            return scores
        
        scores.append(teacher_feedback.score)
        # return scores
        for i in range(1, num_round):
            student_response, teacher_feedback = self.invoke(None, image_file, groundtruth)
            if self.should_stop(teacher_feedback.score):
                scores.append(teacher_feedback.score)
                return scores
            scores.append(teacher_feedback.score)
        return scores
    
    def get_feedback_content(self, text):
        return get_feedback(text)

    def get_score(self, text):
        return get_score(text)
    
def get_feedback(text):
    prefix = "Feedback:"
    prefix_idx = text.find(prefix)
    if prefix_idx == -1:
        raise ValueError(f"cannot find feedback from teacher response {text}")

    feedback = text[prefix_idx+len(prefix):].strip()
    return feedback

def get_score(text):
    score_statement = text.splitlines()[0]
    assert "Score:" in score_statement
    scores = re.findall(r'\d+', score_statement)
    if len(scores) == 0:
        raise ValueError(f"cannot find score from teacher response {text}")
    
    return float(scores[0])