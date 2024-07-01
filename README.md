# <img src="assets/fire.png" alt="drawing" style="width:35px;margin-bottom:-8px;"/> FIRE: A Dataset for Feedback Integration and Refinement Evaluation of Multimodal Models

## Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/MM-FIRE/FIRE
cd FIRE
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```
# Dataset
Coming soon

# Training

## Student Model

```bash
deepspeed --master_port 60000 llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 256 \
    --lora_modules q_proj,k_proj \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path  Lin-Chen/open-llava-next-llama3-8b \
    --version llama_v3_student \
    --data_path data/FeedbackReflection/json/train/fire_student_train_feedback_sim_351k.json \
    --image_folder ./data/FeedbackReflection/image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-next-llama-3-8b-student-lora-merged \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

## Teacher Model

```bash
deepspeed --master_port 60001 llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 256 \
    --lora_modules q_proj,k_proj \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path  Lin-Chen/open-llava-next-llama3-8b \
    --version llama_v3_teacher \
    --data_path data/FeedbackReflection/json/train/fire_teacher_train_feedback.json \
    --data_path_test data/FeedbackReflection/json/test/merge_processed_teacher_test.json \
    --image_folder ./data/FeedbackReflection/image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-next-llama-3-8b-teacher-lora-merged \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --val_logging_steps 3000 \
    --report_to wandb
```
Training for student and teacher models takes 16 hours on 8xA-100-80GB for every 1 million data points.

# Evaluation
## Instruction Following
We follow the exactly same evaluation script LLaVA repo provided. Please refer to this [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
## Fixed Dialogue
Coming soon
## Free Dialogue
Coming soon
# Acknowledgement
Thanks for their brilliant contributions to the community! Here are the codebases we built upon.
* [LLaVA](https://github.com/lm-sys/FastChat)
* [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
* [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT)