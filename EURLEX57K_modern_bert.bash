PROJECT_ROOT="/home/snt/projects_lujun/agentCLS"
MODEL_PATH="/home/snt/llm_models/ModernBERT-base"


CUDA_VISIBLE_DEVICES=0 python script/FT_bert/modern_bert_FT.py \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--num_train_epochs 10 \
--learning_rate 1e-6 \
--project_root $PROJECT_ROOT \
--training_dataset_path assets/training_dataset/EURLEX57K_split_equal_train_1000_val_300.jsonl \
--model_path $MODEL_PATH

sleep 60

CUDA_VISIBLE_DEVICES=0 python script/FT_bert/modern_bert_FT.py \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--num_train_epochs 10 \
--learning_rate 1e-6 \
--project_root $PROJECT_ROOT \
--training_dataset_path assets/training_dataset/EURLEX57K_split_equal_train_100_val_300.jsonl \
--model_path $MODEL_PATH


sleep 60

CUDA_VISIBLE_DEVICES=0 python script/FT_bert/modern_bert_FT.py \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--num_train_epochs 10 \
--learning_rate 1e-6 \
--project_root $PROJECT_ROOT \
--training_dataset_path assets/training_dataset/EURLEX57K_split_equal_train_10_val_300.jsonl \
--model_path $MODEL_PATH

sleep 60

CUDA_VISIBLE_DEVICES=0 python script/FT_bert/modern_bert_FT.py \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--num_train_epochs 10 \
--learning_rate 1e-6 \
--project_root $PROJECT_ROOT \
--training_dataset_path assets/training_dataset/EURLEX57K_split_proportional_train_50_val_300.jsonl \
--model_path $MODEL_PATH


sleep 60

CUDA_VISIBLE_DEVICES=0 python script/FT_bert/modern_bert_FT.py \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--num_train_epochs 10 \
--learning_rate 1e-6 \
--project_root $PROJECT_ROOT \
--training_dataset_path assets/training_dataset/EURLEX57K_split_proportional_train_150_val_300.jsonl \
--model_path $MODEL_PATH


sleep 60

CUDA_VISIBLE_DEVICES=0 python script/FT_bert/modern_bert_FT.py \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--num_train_epochs 10 \
--learning_rate 1e-6 \
--project_root $PROJECT_ROOT \
--training_dataset_path assets/training_dataset/EURLEX57K_split_proportional_train_1500_val_300.jsonl \
--model_path $MODEL_PATH