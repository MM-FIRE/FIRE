 
conda activate llava 
 

# evaluation of BLEU-N CIDER for student model lora v1.6
python -m llava.eval.eval_feedback \
   --model-path checkpoints/llava-v1.6-7b-student-instruct-lora-merged-109846/ \
   --model-base /scratch/TecManDep/A_Models/llava-v1.6-vicuna-7b \
   --dataset-path data/FeedbackReflection/json/test/merge_processed_student_test.json \
   --conv-mode llava_v1_student_feedback \
   --mode student
   