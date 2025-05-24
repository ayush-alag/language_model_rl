import os
import json
import random
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_smoltalk_sft(text):
    return text["messages"]

IGNORE_TOKEN_ID = -100
def tokenize_text(text, tokenizer, max_length, task, query_column, completion_column):
    # TODO: fix smoltalk_sft
    if task == "smoltalk_sft":
        text = process_smoltalk_sft(text)[0]
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    elif task == "warm_start_sft":
        tokens = tokenizer(text[query_column], text[completion_column], truncation="only_second", padding="max_length", max_length=max_length)

        labels = []
        for prompt, attention_mask, input_ids in zip(text[query_column], tokens["attention_mask"], tokens["input_ids"]):
            prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            lab = input_ids.copy()
            lab[:prompt_len] = [IGNORE_TOKEN_ID] * prompt_len
            lab = [l if m==1 else IGNORE_TOKEN_ID for l,m in zip(lab, attention_mask)]
            labels.append(lab)
        tokens["labels"] = labels
        return tokens
    elif task == "countdown":
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_length)

# special because no test dataset
def load_countdown_dataset():
    train_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    return train_dataset

def tokenize_dataset(dataset, max_length, task, query_column, completion_column):
    # we want to use Qwen 2.5 for all datasets
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    return dataset.map(lambda x: tokenize_text(x, tokenizer, max_length, task, query_column, completion_column), batched=True)

def load_dataset_and_tokenize(task, max_length=1024, batch_size=128):
    # support more datasets in the future
    if task == "smoltalk_sft":
        train_dataset, test_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=["train", "test"])
    elif task == "warm_start_sft":
        train_dataset, test_dataset = load_dataset("Asap7772/cog_behav_all_strategies", split=["train", "test"])
        query_column = "query"
        completion_column = "completion"
    else:
        raise ValueError(f"Task {task} not supported")

    tokenized_train_dataset = tokenize_dataset(train_dataset, max_length, task, query_column, completion_column)
    tokenized_test_dataset = tokenize_dataset(test_dataset, max_length, task, query_column, completion_column)

    tokenized_train_dataset.set_format("torch", ["input_ids", "attention_mask", "labels"])
    tokenized_test_dataset.set_format("torch", ["input_ids", "attention_mask", "labels"])

    return DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True), DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="warm_start_sft", choices=["smoltalk_sft", "warm_start_sft", "countdown"])
    parser.add_argument("--output_dir", type=str, default="data/smoltalk_processed")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    train_dataloader, test_dataloader = load_dataset_and_tokenize(args.task, args.max_length, args.batch_size)