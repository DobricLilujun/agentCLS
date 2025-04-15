from datasets.arrow_dataset import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
import torch
import os
import numpy as np
from datetime import datetime
import sys
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import EarlyStoppingCallback
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
from transformers import AutoModel, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

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
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for LoRA")
    parser.add_argument("--hidden_layers", type=int, default=2, help="Number of hidden layers for LoRA")
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
print(f"Hidden Dimension: {args.hidden_dim}")
print(f"Hidden Layers: {args.hidden_layers}")
# ========================================================================

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
hidden_dim = args.hidden_dim
hidden_layers = args.hidden_layers
# ========================================================================
# ========================== CMD Argument Parser ==========================


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


if resume_from_checkpoint and resume_checkpoint_path is None:
    raise ValueError("Please provide a checkpoint path to resume training from")

if resume_from_checkpoint:
    output_dir = resume_checkpoint_path
else:
    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    output_dir = f"{project_root}/assets/logs/SFT/{input_dataset_name}_{train_ratio}_{model_name}_{hidden_layers}_output_{current_time}"

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
    label2id[label] = i
    id2label[i] = label

os.makedirs(output_dir, exist_ok=True)
data = {"label2id": label2id, "id2label": id2label}
json_file_path = os.path.join(output_dir, "adapter_labels.json")

with open(json_file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

train_dataset = train_dataset.map( lambda x: {"labels":label2id[x["labels"]]} )
val_dataset = val_dataset.map( lambda x: {"labels": label2id[x["labels"]]})

keep_columns = ["labels", "input_ids", "attention_mask"]
tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=[col for col in train_dataset.column_names if col not in keep_columns])
tokenized_val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=[col for col in val_dataset.column_names if col not in keep_columns])
train_dataset.features.keys()





from transformers import AutoModel, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

config = AutoConfig.from_pretrained(model_path, label2id=label2id, id2label=id2label)
config.num_labels = num_labels
model_body = AutoModel.from_pretrained(model_path, config=config)


class CustomModel(nn.Module):
    def __init__(self, model_body, input_dim, hidden_dim, output_dim, hidden_layers=1):
        super(CustomModel, self).__init__()
        self.model_body = model_body
        self.loss_fn = self.get_loss_fn()

        layers = []
        in_dim = input_dim
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.score = nn.Sequential(*layers)

    def get_loss_fn(self):
        return nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model_body(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        
        cls_embedding = outputs.last_hidden_state
        logits = self.score(cls_embedding)

        batch_size = cls_embedding.shape[0]
        non_pad_mask = (input_ids != self.model_body.config.pad_token_id).to(logits.device, torch.int32)
        token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_fn(pooled_logits, labels)
        
        return {"logits": pooled_logits, "loss": loss}

    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            outputs = self.model_body(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

            cls_embedding = outputs.last_hidden_state
            logits = self.score(cls_embedding)

            batch_size = cls_embedding.shape[0]
            non_pad_mask = (input_ids != self.model_body.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

            return pooled_logits
    
    def predict_proba(self, embeddings):
        logits = self.head(embeddings)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
    

input_dim = config.hidden_size 
hidden_dim = hidden_dim        
output_dim = num_labels  


model = CustomModel(model_body=model_body, input_dim=input_dim,
                                       hidden_dim=hidden_dim,
                                       output_dim=output_dim, 
                                       hidden_layers=hidden_layers)

model.model_body.config.pad_token_id = tokenizer.pad_token_id
model.model_body.config.use_cache = False
model.model_body.config.pretraining_tp = 1
model.model_body.gradient_checkpointing_enable()
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

        


trainer_stats = None

def main():
    trainer_stats = train()
    return trainer_stats

if __name__ == "__main__":
    trainer_stats = main()
    print("Finished training and evaluation.")
    
# python script/FT_llama/llama3_FT_with_header.py \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --num_train_epochs 10 \
#     --learning_rate 1e-6 \
#     --project_root /home/snt/projects_lujun/agentCLS \
#     --training_dataset_path assets/training_dataset/LDD_split_proportional_train_1500_val_300.jsonl \
#     --model_path /home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct \
#     --r 16
#     --hidden_dim 256 \
#     --hidden_layers 2