import pandas as pd
from datetime import datetime
import sys
import requests
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
from tqdm import tqdm
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using CPT (Continual Pretraining Training)")

    parser.add_argument("--project_root", type=str, default="", help="Path to project root")
    parser.add_argument("--training_dataset_path", type=str, default="", help="Path to training dataset")
    parser.add_argument("--model_path", type=str, default="", help="Path to model")
    parser.add_argument("--vllm_url", type=str, default="http://0.0.0.0:8000/v1/chat/completions", help="URL to VLLM API")
    parser.add_argument("--is_BASE", action="store_true", help="Whether to run BASE")
    parser.add_argument("--is_COT", action="store_true", help="Whether to run COT")
    parser.add_argument("--is_COD", action="store_true", help="Whether to run COD")
    parser.add_argument("--is_FEW_SHOT", action="store_true", help="Whether to run FEW_SHOT")
    parser.add_argument("--is_SELF_CONSIS", action="store_true", help="Whether to run SELF_CONSIS")

    return parser.parse_args()

args = parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# project_root = "/home/snt/projects_lujun/agentCLS"
# training_dataset_path = "assets/training_dataset/EURLEX57K_split_proportional_train_1500_val_300.jsonl"
# model_path = "/home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct"
# VLLM_API_URL = "http://0.0.0.0:8000/v1/chat/completions"
# is_BASE = True
# is_COT = True
# is_COD = True
# is_FEW_SHOT = True
# is_SELF_CONSIS = True

project_root = args.project_root

sys.path.append(os.path.abspath(project_root))
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
training_dataset_path = args.training_dataset_path
model_path = args.model_path
VLLM_API_URL = args.vllm_url
is_BASE = args.is_BASE
is_COT = args.is_COT
is_COD = args.is_COD
is_FEW_SHOT = args.is_FEW_SHOT
is_SELF_CONSIS = args.is_SELF_CONSIS

print(f"project_root: {project_root}")
print(f"training_dataset_path: {training_dataset_path}")
print(f"model_path: {model_path}")
print(f"VLLM_API_URL: {VLLM_API_URL}")
print(f"is_BASE: {is_BASE}")
print(f"is_COT: {is_COT}")
print(f"is_COD: {is_COD}")
print(f"is_FEW_SHOT: {is_FEW_SHOT}")
print(f"is_SELF_CONSIS: {is_SELF_CONSIS}")


MAX_LEN = 512
temperature = 0.8
repetition_penalty = 1.1
top_k = 10
top_p= 0.7
prompt_templates = []

model_name = model_path.split("/")[-1]
train_dataset_path = os.path.abspath(os.path.join(project_root, training_dataset_path))
sys.path.append(project_root)

current_time = datetime.now()
formatted_time = current_time.strftime('%m_%d_%H_%M')


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



dataset = pd.read_json(training_dataset_path, lines=True)
dataset = dataset[dataset["split"]=="validation"]
sample_size = len(dataset)
start_path = project_root + "/" + "assets/prompt_testing/"


def create_prompt(sample, prompt_template, model_name):
    prompt = prompt_template.format(input=sample["content"])
    system_message = "You are a helpful AI assistant."
    if "gemma" in model_name:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    
    return messages


def call_vllm(messages, vllm_api_url, max_len=100, temperature=1.0):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_path,
        "messages": messages,
        "max_tokens": max_len,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k
    }
    
    try:
        response = requests.post(vllm_api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Error: Received status code {response.status_code}, response: {response.text}")
            return ""
    except Exception as e:
        print(f"Exception while calling VLLM API: {e}")
        return ""


# Self-consistency prompt
def create_SIS_prompt(sample, prompt_template, path1, path2, path3, model_name):
    prompt = prompt_template.format(question_prompt=sample["content"], path_1=path1, path_2=path2, path_3=path3)
    system_message = "You are a helpful AI assistant."
    if "gemma" in model_name:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    return messages


def parse_classification(prompt_output, list_category):
    default_category = list_category[0]
    prompt_output = prompt_output.strip()
    parts = prompt_output.split('####')
    
    if len(parts) > 1:
        string_last = parts[-1].strip()
        if string_last:
            for category in list_category:
                if category.lower() in string_last.lower():
                    return category
            else:
                category_counts = {}
                for category in list_category:
                    count = string_last.lower().count(category.lower())
                    category_counts[category] = count
                max_count = max(category_counts.values())
                if max_count == 0:
                    category_counts = {}
                    for category in list_category:
                        count = prompt_output.lower().count(category.lower())
                        category_counts[category] = count
                    max_count = max(category_counts.values())
                    max_categories = [k for k, v in category_counts.items() if v == max_count]
                    if max_count == 0:
                        return default_category
                    else:
                        return max_categories[0]
                max_categories = [k for k, v in category_counts.items() if v == max_count]
                return max_categories[0]
    
        else:
            return default_category
    else:
        category_counts = {}
        for category in list_category:
            count = prompt_output.lower().count(category.lower())
            category_counts[category] = count
        max_count = max(category_counts.values())

        if max_count == 0:
            return default_category
            
        max_categories = [k for k, v in category_counts.items() if v == max_count]
        return max_categories[0]
    

def generate_dataset_responses(dataset,prompt_template, eval_output_path):
    list_categories = dataset["cls_label"].unique().tolist()
    df_results = pd.DataFrame()

    y_true = []
    y_pred = []
    for index, sample in tqdm(dataset.iterrows(), total=len(dataset), desc="Generating responses"):
        start_time = datetime.now()
        messages = create_prompt(sample, prompt_template=prompt_template, model_name=model_name)
        llm_response = call_vllm(messages = messages, vllm_api_url= VLLM_API_URL, max_len=MAX_LEN, temperature=temperature).strip()
        ground_truth_cls = sample["cls_label"]
        parsed_cls = parse_classification(llm_response, list_categories)
        y_true.append(ground_truth_cls)
        y_pred.append(parsed_cls)
        if parsed_cls is None:
            is_correct = False
        else:
            is_correct = ground_truth_cls.lower() == parsed_cls.lower()
        end_time = datetime.now()
        time_diff = (end_time - start_time).total_seconds()
        result = {
            "LLM_Input": messages,
            "LLM_Output": llm_response,
            "LLM_Prediciton": parsed_cls,
            "Ground_Truth": ground_truth_cls,
            "Iscorrect": is_correct,
            "Time_Passed": time_diff,
        }
        updated_dataframe = pd.DataFrame([result])
        updated_dataframe.to_json(eval_output_path, orient="records", lines=True, mode="a")
        df_results = pd.concat([df_results, updated_dataframe], axis=0)

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")  # Use 'weighted' for imbalanced classes
    try:
        auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), multi_class="ovr")
    except ValueError:
        auc = "N/A (AUC is not applicable for single-class cases)"

    print(f"ACC: {acc:.4f}\tF1: {f1:.4f}\tAUC: {auc}")
    return df_results

def generate_dataset_responses_SIS(dataset, prompt_template_SIS , prompt_template_COT, eval_output_path):
    list_categories = dataset["cls_label"].unique().tolist()
    df_results = pd.DataFrame()
    rep = 3
    y_true = []
    y_pred = []
    for index, sample in tqdm(dataset.iterrows(), total=len(dataset), desc="Generating responses"):
        ## Generate Three Paths
        start_time = datetime.now()
        messages_COT = create_prompt(sample, prompt_template=prompt_template_COT, model_name=model_name)
        paths = [call_vllm(messages = messages_COT, vllm_api_url= VLLM_API_URL, max_len=MAX_LEN, temperature=temperature).strip() for i in range(rep)]
        messages_SIS = create_SIS_prompt(sample, prompt_template=prompt_template_SIS, path1=paths[0], path2=paths[1], path3=paths[2], model_name=model_name)
        llm_response = call_vllm(messages = messages_SIS, vllm_api_url= VLLM_API_URL, max_len=MAX_LEN, temperature=temperature).strip()
        ground_truth_cls = sample["cls_label"]
        parsed_cls = parse_classification(llm_response, list_categories)
        y_true.append(ground_truth_cls)
        y_pred.append(parsed_cls)
        if parsed_cls is None:
            is_correct = False
        else:
            is_correct = ground_truth_cls.lower() == parsed_cls.lower()
        end_time = datetime.now()
        time_diff = (end_time - start_time).total_seconds()
        messages = {"messages_COT": messages_COT, "messages_SIS": messages_SIS}
        result = {
            "LLM_Input": messages,
            "LLM_Output": llm_response,
            "LLM_Prediciton": parsed_cls,
            "Ground_Truth": ground_truth_cls,
            "Iscorrect": is_correct,
            "Time_Passed": time_diff,
            "Paths": paths
        }
        updated_dataframe = pd.DataFrame([result])
        updated_dataframe.to_json(eval_output_path, orient="records", lines=True, mode="a")
        df_results = pd.concat([df_results, updated_dataframe], axis=0)

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")  # Use 'weighted' for imbalanced classes
    try:
        auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), multi_class="ovr")
    except ValueError:
        auc = "N/A (AUC is not applicable for single-class cases)"

    print(f"ACC: {acc:.4f}\tF1: {f1:.4f}\tAUC: {auc}")
    return df_results


print("[INFO] Starting base prompt testing...")
if is_BASE:
    eval_output_path = start_path + training_dataset_path.split("/")[-1].replace(".jsonl", f"_{formatted_time}_is_base_{model_name}.jsonl")
    testset = dataset[dataset["split"]=="validation"].sample(sample_size)
    generated_outputs_df = generate_dataset_responses(testset, prompt_templates[0], eval_output_path)

print("[INFO] Starting COT prompt testing...")
if is_COT:
    eval_output_path = start_path + training_dataset_path.split("/")[-1].replace(".jsonl", f"_{formatted_time}_is_cot_{model_name}.jsonl")
    testset = dataset[dataset["split"]=="validation"].sample(sample_size)
    generated_outputs_df = generate_dataset_responses(testset, prompt_templates[1], eval_output_path)

print("[INFO] Starting COD prompt testing...")
if is_COD:
    eval_output_path = start_path + training_dataset_path.split("/")[-1].replace(".jsonl", f"_{formatted_time}_is_cod_{model_name}.jsonl")
    testset = dataset[dataset["split"]=="validation"].sample(sample_size)
    generated_outputs_df = generate_dataset_responses(testset, prompt_templates[2], eval_output_path)

print("[INFO] Starting Few-shot prompt testing...")
if is_FEW_SHOT:
    eval_output_path = start_path + training_dataset_path.split("/")[-1].replace(".jsonl", f"_{formatted_time}_is_few_shot_{model_name}.jsonl")
    testset = dataset[dataset["split"]=="validation"].sample(sample_size)
    generated_outputs_df = generate_dataset_responses(testset, prompt_templates[3], eval_output_path)

print("[INFO] Starting Self-consistency prompt testing...")
if is_SELF_CONSIS:
    eval_output_path = start_path + training_dataset_path.split("/")[-1].replace(".jsonl", f"_{formatted_time}_is_self_consis_{model_name}.jsonl")
    testset = dataset[dataset["split"]=="validation"].sample(sample_size)
    generated_outputs_df = generate_dataset_responses_SIS(testset, prompt_templates[4],prompt_templates[1], eval_output_path)

print("[INFO] Prompt testing completed!")



# python script/prompt_eval/prompt_testing.py \
#     --project_root "/home/snt/projects_lujun/agentCLS" \
#     --training_dataset_path "assets/training_dataset/EURLEX57K_split_proportional_train_1500_val_300.jsonl" \
#     --model_path "/home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct" \
#     --vllm_url "http://0.0.0.0:8000/v1/chat/completions" \
#     --is_BASE \
#     --is_COT \
#     --is_COD \
#     --is_FEW_SHOT \
#     --is_SELF_CONSIS
