from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PrefixTuningConfig
import torch
import os
from tqdm import tqdm
import sys
from datetime import datetime
import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets import ClassLabel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback
from peft import PeftModel, PeftConfig
import time
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================== CMD Argument Parser ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using CPT (Continual Pretraining Training)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--project_root", type=str, default="/Users/lujun.li/projects/mt_luxembourgish", help="Path to project root")
    parser.add_argument("--training_dataset_path", type=str, default="data/processed/dataset_merged_llama_fake_targets.jsonl", help="Path to training dataset")
    parser.add_argument("--model_path", type=str, default="/home/llama/Personal_Directories/srb/binary_classfication/Llama-3.2-3B-Instruct", help="Path to model")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="Resume training from checkpoint")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--is_prefix_tuning", action="store_true", help="Whether to use PrefixTuning")
    parser.add_argument("--is_prompt_tuning", action="store_true", help="Whether to use PromptTuning")
    parser.add_argument("--num_virtual_tokens", type=int, default=32, help="Number of virtual tokens")
    return parser.parse_args()

args = parse_args()
per_device_eval_batch_size = args.per_device_eval_batch_size
per_device_train_batch_size = args.per_device_train_batch_size
num_train_epochs = args.num_train_epochs
learning_rate = args.learning_rate
project_root = args.project_root
training_dataset_path = args.training_dataset_path
model_path = args.model_path
resume_from_checkpoint = args.resume_from_checkpoint
resume_checkpoint_path = args.resume_checkpoint_path
is_prefix_tuning = args.is_prefix_tuning
is_prompt_tuning = args.is_prompt_tuning
num_virtual_tokens = args.num_virtual_tokens


print(f"per_device_train_batch_size: {per_device_train_batch_size}")
print(f"per_device_eval_batch_size: {per_device_eval_batch_size}")
print(f"num_train_epochs: {num_train_epochs}")
print(f"learning_rate: {learning_rate}")
print(f"project_root: {project_root}")
print(f"training_dataset_path: {training_dataset_path}")
print(f"model_path: {model_path}")
print(f"resume_from_checkpoint: {resume_from_checkpoint}")
print(f"resume_checkpoint_path: {resume_checkpoint_path}")
print(f"is_prefix_tuning: {is_prefix_tuning}")
print(f"is_prompt_tuning: {is_prompt_tuning}")
print(f"num_virtual_tokens: {num_virtual_tokens}")


# ========================== Constants ==========================
# per_device_train_batch_size = 1
# per_device_eval_batch_size = 1
# num_train_epochs = 10
# learning_rate = 1e-6
# project_root = "/home/snt/projects_lujun/agentCLS"
# training_dataset_path = "assets/training_dataset/EURLEX57K_split_equal_train_1000_val_300.jsonl"
# model_path = "/home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct"
# is_prefix_tuning = True
# is_prompt_tuning = False
# num_virtual_tokens = 128



train_dataset_path = os.path.abspath(os.path.join(project_root, training_dataset_path))
sys.path.append(project_root)

from utils.prompts import (
    prompt_EUR_BASE,
    prompt_EUR_COT,
    prompt_EUR_COD,
    prompt_EUR_FEW_SHOT,
    prompt_LDD_BASE,
    prompt_LDD_COT,
    prompt_LDD_COD,
    prompt_LDD_FEW_SHOT,
    prompt_IE_BASE,
    prompt_IE_COT,
    prompt_IE_COD,
    prompt_IE_FEW_SHOT,
    prompt_SELF_CONSIS,
)


peft_config = None
resume_from_checkpoint = False
resume_checkpoint_path = None
train_ratio = 0.005
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_dataset_name = train_dataset_path.split("/")[-1].split(".")[0]
model_name = model_path.split("/")[-1]
max_length = 4096
train_seed = 3407
logging_steps = 10
eval_steps = 100
eval_strategy = "epoch"
save_strategy = "epoch"
save_total_limit = 2
logging_strategy = "steps"
max_grad_norm = 0.3
label_names = "labels"

if "EURLEX" in training_dataset_path:
    prompt_templates = [prompt_EUR_BASE, prompt_EUR_COT, prompt_EUR_COD, prompt_EUR_FEW_SHOT, prompt_SELF_CONSIS]
    is_EURLEX = True
elif "LDD" in training_dataset_path:
    prompt_templates = [prompt_LDD_BASE, prompt_LDD_COT, prompt_LDD_COD, prompt_LDD_FEW_SHOT, prompt_SELF_CONSIS]
    is_LDD = True
elif "FOYER" in training_dataset_path:
    prompt_templates = [prompt_IE_BASE, prompt_IE_COT, prompt_IE_COD, prompt_IE_FEW_SHOT, prompt_SELF_CONSIS]
    is_IE = True
else:
    raise ValueError(f"Unknown dataset: {training_dataset_path}")



if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


if is_prefix_tuning:
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_virtual_tokens,
        base_model_name_or_path=model_path,
    )

if is_prompt_tuning:
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=prompt_templates[0],
        tokenizer_name_or_path=model_path,
    )



if resume_from_checkpoint and resume_checkpoint_path is None:
    raise ValueError("Please provide a checkpoint path to resume training from")

if resume_from_checkpoint:
    output_dir = resume_checkpoint_path
else:
    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    if "PREFIX_TUNING" in peft_config.peft_type:
        output_dir = f"{project_root}/assets/logs/prompt_tuning/{input_dataset_name}_{train_ratio}_{model_name}_output_{current_time}_PREFIX_TUNING_{peft_config.num_virtual_tokens}"
    elif "PROMPT_TUNING" in peft_config.peft_type:
        output_dir = f"{project_root}/assets/logs/prompt_tuning/{input_dataset_name}_{train_ratio}_{model_name}_output_{current_time}_PROMPT_TUNING_{peft_config.num_virtual_tokens}"
    else:
        raise ValueError("Please provide a valid peft_type")
    
dataset = pd.read_json(train_dataset_path, lines=True)
dataset.rename(columns={"cls_label": "labels"}, inplace=True)

# Compute sample counts for each group based on 'labels' and 'split'
sample_counts = dataset.groupby(['labels', 'split']).size() * train_ratio

filtered_train_data = dataset.groupby('labels', group_keys=False).apply(
    lambda x: x[x['split'] == 'train'].iloc[:int(sample_counts.loc[x.name, 'train'])]
)

filtered_validation_data = dataset.groupby('labels', group_keys=False).apply(
    lambda x: x[x['split'] == 'validation'].iloc[:int(sample_counts.loc[x.name, 'validation'])]
)

filtered_train = filtered_train_data.reset_index(drop=True)
filtered_validation = filtered_validation_data.reset_index(drop=True)


train_dataset = Dataset.from_pandas(filtered_train)
val_dataset = Dataset.from_pandas(filtered_validation)


# Tokenization
def tokenize(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=max_length)

labels =  set(train_dataset['labels'])
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

train_dataset = train_dataset.map(lambda x: {"labels": label2id[x["labels"]]})
val_dataset = val_dataset.map(lambda x: {"labels": label2id[x["labels"]]})

labels_ids = list(set(train_dataset['labels']))
class_label = ClassLabel(num_classes=len(labels_ids), names=labels_ids)
train_dataset = train_dataset.cast_column("labels", class_label)
val_dataset = val_dataset.cast_column("labels", class_label)

# def apply_prompt_template(examples):
#     # Applying the prompt template to the 'content' column
#     return {
#         "content": prompt_templates[0].format(input=examples["content"])
#     }

# train_dataset = train_dataset.map(apply_prompt_template)
# val_dataset = val_dataset.map(apply_prompt_template)

keep_columns = ["labels", "input_ids", "attention_mask"]
tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=[col for col in train_dataset.column_names if col not in keep_columns])
tokenized_val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=[col for col in val_dataset.column_names if col not in keep_columns])
train_dataset.features.keys()


model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, label2id=label2id, id2label=id2label)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())


import json 


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy_score(labels, np.argmax(predictions, axis=-1))
    f1 = f1_score(labels, np.argmax(predictions, axis=-1), average="weighted")
    return {"accuracy": acc, "f1": f1}

def train():
    # Define training args
    training_args = TrainingArguments(
        output_dir= output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=True,
        optim="adamw_torch_fused", 
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        max_grad_norm=max_grad_norm,
        # group_by_length=True,
        # use_mps_device=True,
        metric_for_best_model="eval_loss",
        # push to hub parameters
        # push_to_hub=True,
        # hub_strategy="every_save",
        # hub_token=HfFolder.get_token(),
        report_to="tensorboard",
        disable_tqdm=False,
        seed = train_seed,
        # label_names=list(id2label.values()),
    )
    
    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Finished training SFT.")
    return trainer_stats

def get_last_checkpoints(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [c for c in checkpoints if "checkpoint" in c]
    checkpoints = [int(c.split("-")[-1]) for c in checkpoints]
    last_checkpoint = max(checkpoints)
    return f"{output_dir}/checkpoint-{last_checkpoint}"

def get_all_checkpoints(output_dir):
    # List all checkpoint directories in output_dir
    checkpoints = os.listdir(output_dir)
    # Filter for directories that include 'checkpoint' in their name
    checkpoints = [c for c in checkpoints if "checkpoint" in c]
    # Return the full paths to all the checkpoint directories
    return [os.path.join(output_dir, checkpoint) for checkpoint in checkpoints]


def evaluate():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints_path  = get_last_checkpoints(output_dir)
    config = PeftConfig.from_pretrained(checkpoints_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, label2id=label2id, id2label=id2label, ).to(device)
    model = PeftModel.from_pretrained(model, checkpoints_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    validation_results = []

    # Initialize lists to store true and predicted labels
    true_label_one_hot_list = []
    true_labels = []
    predicted_labels = []
    all_probs = []

    # Start time for measuring inference efficiency
    start_time = time.time()

    # Iterate through validation dataset and make predictions
    for input in tqdm(val_dataset, desc="Processing validation data", unit="sample"):
        sample = input['content']
        true_label_idx = int(input['labels'])
        tokenized_input = tokenizer(sample, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").to(device)
        model_output = model(**tokenized_input)
        logits = model_output.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        predicted_label = id2label[str(predicted_class_idx)]
        
        true_label = id2label[str(true_label_idx)]
        true_label_one_hot = np.zeros(probabilities.size(-1))
        true_label_one_hot[true_label_idx] = 1

        true_labels.append(true_label)
        true_label_one_hot_list.append(true_label_one_hot)
        predicted_labels.append(predicted_label)
        all_probs.append(probabilities.detach().cpu().numpy())  # Store the raw probabilities for AUC
        result = {
            'content': sample,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'true_label_one_hot': true_label_one_hot.tolist(),
            'predicted_class_idx': predicted_class_idx,
            'probabilities': probabilities.detach().cpu().numpy().tolist()  # Convert to list for JSON serialization
        }
        validation_results.append(result)

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    df_validation_results = pd.DataFrame(validation_results)
    jsonl_file_path = os.path.join(checkpoints_path, f'validation_results_{timestamp}.jsonl')
    df_validation_results.to_json(jsonl_file_path, orient='records', lines=True)


    # Calculate accuracy, F1 score, and AUC
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')  # weighted F1 score
    auc = roc_auc_score(np.array(true_label_one_hot_list),  np.squeeze(np.array(all_probs), axis=1), multi_class='ovr', average='weighted')  # for multi-class AUC

    # Calculate inference time (average time per sample)
    end_time = time.time()
    inference_time = (end_time - start_time) / len(train_dataset)

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Average Inference Time per Sample: {inference_time:.4f} seconds")
    return 
        


trainer_stats = None

def main():
    trainer_stats = train()
    return trainer_stats

if __name__ == "__main__":
    trainer_stats = main()
    eval_results = evaluate()

    print("Finished training and evaluation.")



# python script/SPT/prompt_tuning_peft.py \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --num_train_epochs 10 \
#   --learning_rate 1e-5 \
#   --project_root "/home/snt/projects_lujun/agentCLS" \
#   --training_dataset_path "assets/training_dataset/EURLEX57K_split_equal_train_1000_val_300.jsonl" \
#   --model_path "/home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct" \
#   --is_prompt_tuning \
#   --num_virtual_tokens 128

# python script/SPT/prompt_tuning_peft.py \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train_epochs 10 \
#   --learning_rate 1e-5 \
#   --project_root "/home/snt/projects_lujun/agentCLS" \
#   --training_dataset_path "assets/training_dataset/EURLEX57K_split_equal_train_1000_val_300.jsonl" \
#   --model_path "/home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct" \
#   --is_prefix_tuning \
#   --num_virtual_tokens 128