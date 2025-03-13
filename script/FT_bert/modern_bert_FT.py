from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import ClassLabel
from datetime import datetime
import warnings
from transformers import logging as transformers_logging
import argparse
import os
import sys
from transformers import AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score
from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.simplefilter("ignore")
transformers_logging.set_verbosity_error()

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


per_device_train_batch_size = args.per_device_train_batch_size  # Batch size for training per device
per_device_eval_batch_size = args.per_device_eval_batch_size  # Batch size for evaluation per device
num_train_epochs = args.num_train_epochs  # Number of epochs for training
learning_rate = args.learning_rate # Learning rate for the optimizer
project_root = args.project_root
training_dataset_path = args.training_dataset_path
model_path = args.model_path
resume_from_checkpoint = args.resume_from_checkpoint
resume_checkpoint_path = args.resume_checkpoint_path

# Data preparation
# per_device_train_batch_size = 8
# per_device_eval_batch_size = 8
# num_train_epochs = 5
# learning_rate = 5e-5
# project_root = "/home/snt/projects_lujun/agentCLS"
# training_dataset_path = "assets/LDD_split.json"
# model_path = "answerdotai/ModernBERT-base"
# resume_from_checkpoint = False
# resume_checkpoint_path = None

train_dataset_path = os.path.abspath(os.path.join(project_root, training_dataset_path))
sys.path.append(project_root)

tokenizer = AutoTokenizer.from_pretrained(model_path)
train_seed = 3407
train_ratio = 0.001
logging_steps = 10
eval_steps = 10
eval_strategy = "steps"
save_strategy = "epoch"
save_total_limit = 2
logging_strategy = "steps"
max_grad_norm = 0.3
max_length = 4096
input_dataset_name = train_dataset_path.split("/")[-1].split(".")[0]
model_name = model_path.split("/")[-1]


if resume_from_checkpoint and resume_checkpoint_path is None:
    raise ValueError("Please provide a checkpoint path to resume training from")

if resume_from_checkpoint:
    output_dir = resume_checkpoint_path
else:
    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    output_dir = f"{project_root}/assets/logs/{input_dataset_name}_{train_ratio}_{model_name}_output_{current_time}"

dataset = pd.read_json(train_dataset_path, lines=True)
dataset.rename(columns={"cls_label": "labels"}, inplace=True)


# if "EURLEX57K_split" in train_dataset_path:
#     if "cls_label" in dataset.columns:
#         dataset.rename(columns={"cls_label": "labels"}, inplace=True)
# else:
#     if "cls_label" in dataset.columns:
#         dataset.rename(columns={"cls_label": "labels"}, inplace=True)


sample_counts = dataset.groupby('labels').size() * train_ratio
filtered_data = dataset.groupby('labels', group_keys=False).apply(lambda x: x.iloc[:int(sample_counts[x.name])])
dataset = filtered_data.reset_index(drop=True)
dataset = Dataset.from_pandas(dataset)

train_dataset = dataset.filter(lambda x: x["split"] == "train")
val_dataset = dataset.filter(lambda x: x["split"] == "validation")

# Prepare model labels - useful for inference
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


def tokenize(examples):
    # Tokenize the content and add the corresponding labels to the output
    tokenized_inputs = tokenizer(examples["content"], padding=True, max_length=max_length, truncation=True, return_tensors="pt")
    # tokenized_inputs["label"] = examples["labels"] 
    return tokenized_inputs


keep_columns = ["labels", "input_ids", "attention_mask"]

# Remove all other columns from the dataset
tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=[col for col in train_dataset.column_names if col not in keep_columns])
tokenized_val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=[col for col in val_dataset.column_names if col not in keep_columns])
train_dataset.features.keys()
    

## Load model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, label2id=label2id, id2label=id2label,)

# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    
    return {"f1": float(score) if score == 1 else score}



def train():
    # Define training args
    training_args = TrainingArguments(
        output_dir= output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=5e-5,
        num_train_epochs=num_train_epochs,
        bf16=True,
        optim="adamw_torch_fused", 
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        max_grad_norm=max_grad_norm,
        # group_by_length=True,
        # use_mps_device=True,
        metric_for_best_model="f1",
        # push to hub parameters
        # push_to_hub=True,
        # hub_strategy="every_save",
        # hub_token=HfFolder.get_token(),
        report_to="tensorboard",
        disable_tqdm=False,
        seed = train_seed
    )
    
    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Finished training SFT.")
    return trainer_stats


trainer_stats = None

def main():
    trainer_stats = train()
    return trainer_stats

if __name__ == "__main__":
    trainer_stats = main()