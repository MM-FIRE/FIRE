from typing import Any
from transformers.integrations import WandbCallback
import pandas as pd
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from torch.utils.data import RandomSampler
from torch.utils.data import Subset
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput
import wandb
from transformers import EvalPrediction
from typing import Dict
from transformers import AutoTokenizer

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.run_llava import image_parser, load_images
import random
from collections import defaultdict
from tqdm import tqdm
import torch

# from llava.eval.feedback_chat import generate
IMAGE_TOKEN_PLACEHOLDER = -200

class Metrics():
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, data: EvalPrediction) -> Dict:
        '''print('inputs',data.inputs.shape)
        print('labels',data.label_ids.shape)
        print('predictions', data.predictions.shape)
        n_samples = data.inputs.shape[0]
        for i in range(n_samples):
            input_ids = data.inputs[i].tolist()
            input_ids.remove(IMAGE_TOKEN_PLACEHOLDER)
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            
            print(input_text)
            break
        #inputs = self.tokenizer.batch_decode(data.inputs, skip_special_tokens=True)[0].strip()
        #print(inputs)'''
        return dict()


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=10, freq=2, conv_mode="llava_v1_student_feedback"):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset # truncate
        self.freq = freq
        self.num_samples = num_samples
        self.conv_mode = conv_mode
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.freq != 0:
            return
        
        if self.sample_dataset is None:
            return
        
        self.on_evaluate(args, state, control, **kwargs)
    
    
    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)       
        batch_size = 8
        num_batch = len(self.sample_dataset) // batch_size + 1
        # avoid OOM
        metrics = defaultdict(lambda: float(0))
        pbar = tqdm(total=num_batch)
        for i in range(num_batch):
            sample_indices = [i for i in range(i*batch_size, (i+1) * batch_size)]
            sample_indices = [i for i in sample_indices if i < len(self.sample_dataset)]
            if len(sample_indices) == 0:
                break
            dataset = Subset(self.sample_dataset, sample_indices)
            # this will OOM if dataset is huge
            predictions: PredictionOutput = self.trainer.predict(dataset)
            for k, v in predictions.metrics.items():
                metrics[k] += v
            pbar.update(1)
            
        try:
            for k, v in metrics.items():
                print("eval", k, v /num_batch)
                self._wandb.log({f"eval/{k}": v / num_batch})
        except Exception as e:
            pass
        
import torch

def generate(
    conv_mode,
    model,
    tokenizer,
    dataset
             ):
    template = conv_templates[conv_mode]
    image_processor = model.get_vision_tower().image_processor

    converation = template.copy()
    predictions = []
    labels = []
    image_files = [dataset["image"]]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    class ModelConfig:
        image_aspect_ratio = "anyres"
        image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    images_tensor = process_images(
        images,
        image_processor,
        ModelConfig()
    ).to(model.device, dtype=model.dtype)
    # print(images_tensor.shape)

    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 512
    # print(images_tensor.shape)
    for turn in dataset["conversations"]:
        if turn["from"] == "gpt":
            role = template.roles[1]
            # do inference
            converation.append_message(role, None)
            
            prompt = converation.get_prompt()
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, 
                                        -200, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )
            
            #print("prompt")
            #print("inputs", input_ids.shape, images_tensor.shape)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            predictions.append(outputs)
            labels.append(turn["value"])
            converation.messages[-1][1] = turn["value"] 
        else:
            role = template.roles[0]
            converation.append_message(
                role, 
                turn["value"])
            continue
    return predictions, labels