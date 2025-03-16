from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
import torch
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime
import sys
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, LoraConfig
import json
from sklearn.metrics import roc_auc_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import ClassLabel
import os
import torch
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
# Data preparation
import torch
from transformers import EarlyStoppingCallback
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 
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
    parser.add_argument("--qlora", type=bool, default=False, help="Use QLoRA")
    parser.add_argument("--r", type=int, default=16, help="Rank for LoRA")
    return parser.parse_args()

args = parse_args()


print("Arguments passed:")
print(f"Train Batch Size: {args.per_device_train_batch_size}")
print(f"Eval Batch Size: {args.per_device_eval_batch_size}")
print(f"Number of Epochs: {args.num_train_epochs}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Project Root: {args.project_root}")
print(f"Training Dataset Path: {args.training_dataset_path}")
print(f"Model path: {args.model_path}")
print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
print(f"Resume checkpoint path: {args.resume_checkpoint_path}")
print(f"Qlora: {args.qlora}")
print(f"Rank: {args.r}")

per_device_train_batch_size = args.per_device_train_batch_size  # Batch size for training per device
per_device_eval_batch_size = args.per_device_eval_batch_size  # Batch size for evaluation per device
num_train_epochs = args.num_train_epochs  # Number of epochs for training
learning_rate = args.learning_rate # Learning rate for the optimizer
project_root = args.project_root
training_dataset_path = args.training_dataset_path
model_path = args.model_path
resume_from_checkpoint = args.resume_from_checkpoint
resume_checkpoint_path = args.resume_checkpoint_path
qlora = args.qlora
r = args.r


## Data preparation
# per_device_train_batch_size = 8
# per_device_eval_batch_size = 8
# num_train_epochs = 1
# learning_rate = 5e-5
# project_root = "/home/snt/projects_lujun/agentCLS"
# training_dataset_path = "assets/training_dataset/LDD_split.json"
# model_path = "/home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct"
# resume_from_checkpoint = False
# resume_checkpoint_path = None
# qlora = True
# r = 16

train_dataset_path = os.path.abspath(os.path.join(project_root, training_dataset_path))
sys.path.append(project_root)


# Default Parameters
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

train_seed = 3407
train_ratio = 1.0
logging_steps = 10
eval_steps = 100
eval_strategy = "epoch"
save_strategy = "epoch"
save_total_limit = 2
logging_strategy = "steps"
max_grad_norm = 0.3
input_dataset_name = train_dataset_path.split("/")[-1].split(".")[0]
model_name = model_path.split("/")[-1]
max_length = 4096
load_in_4bit = True
bnb_4bit_quant_type = 'nf4'
quantization_config = None


if qlora:
# Quantization with Lora
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit, # enable 4-bit quantization
        bnb_4bit_quant_type = bnb_4bit_quant_type, # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
        bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
    )

    # Lora
    lora_config = LoraConfig(
        r = r, # the dimension of the low-rank matrices
        lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout = 0.05, # dropout probability of the LoRA layers
        bias = 'none', # wether to train bias weights, set to 'none' for attention layers
        task_type = 'SEQ_CLS',
    )


if resume_from_checkpoint and resume_checkpoint_path is None:
    raise ValueError("Please provide a checkpoint path to resume training from")

if resume_from_checkpoint:
    output_dir = resume_checkpoint_path
else:
    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    output_dir = f"{project_root}/assets/logs/{input_dataset_name}_{train_ratio}_{model_name}_output_{current_time}"

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


keep_columns = ["labels", "input_ids", "attention_mask"]
tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=[col for col in train_dataset.column_names if col not in keep_columns])
tokenized_val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=[col for col in val_dataset.column_names if col not in keep_columns])
train_dataset.features.keys()


# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, label2id=label2id, id2label=id2label, quantization_config=quantization_config,)
if qlora:
    model = get_peft_model(prepare_model_for_kbit_training(model), lora_config)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()


train_dataset.features.keys()


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


def evaluate():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def get_last_checkpoints(output_dir):
        checkpoints = os.listdir(output_dir)
        checkpoints = [c for c in checkpoints if "checkpoint" in c]
        checkpoints = [int(c.split("-")[-1]) for c in checkpoints]
        last_checkpoint = max(checkpoints)
        return f"{output_dir}/checkpoint-{last_checkpoint}"
    
    checkpoints_path  = get_last_checkpoints(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoints_path, num_labels=num_labels, label2id=label2id, id2label=id2label, quantization_config=quantization_config,).to(device)
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

# python llama3_FT.py \
# --per_device_train_batch_size 8 \
# --per_device_eval_batch_size 8 \
# --num_train_epochs 10 \
# --learning_rate 1e-6 \
# --project_root /home/llama/Personal_Directories/srb/agentCLS \
# --training_dataset_path assets/training_dataset/LDD_split_equal_train_1000_val_300.jsonl \
# --model_path /home/llama/Personal_Directories/srb/binary_classfication/Llama-3.2-3B-Instruct \
# --resume_from_checkpoint "False" \
# --resume_checkpoint_path "" \
# --qlora False \
# --r 16