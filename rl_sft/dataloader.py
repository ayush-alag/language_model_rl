import os
import json
import random
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from prompts import WSD_PROMPT_FORMAT

""" WSD dataset """

IGNORE_TOKEN_ID = -100
def tokenize_wsd_dataset(dataset, tokenizer, max_length):
    def tokenize_text(text, tokenizer, max_length):
        tokens = tokenizer(text["query"], text["completion"], truncation="only_second", padding="max_length", max_length=max_length)

        labels = []
        for prompt, attention_mask, input_ids in zip(text["query"], tokens["attention_mask"], tokens["input_ids"]):
            prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            lab = input_ids.copy()
            lab[:prompt_len] = [IGNORE_TOKEN_ID] * prompt_len
            lab = [l if m==1 else IGNORE_TOKEN_ID for l,m in zip(lab, attention_mask)]
            labels.append(lab)
        tokens["labels"] = labels
        return tokens

    tokenized_dataset = dataset.map(lambda x: tokenize_text(x, tokenizer, max_length), batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset

def get_wsd_dataset(tokenizer, max_length=1024, batch_size=128, synthetic_dataset=None):
    print("Loading WSD dataset")
    train_dataset, test_dataset = load_dataset("Asap7772/cog_behav_all_strategies", split=["train", "test"])

    if synthetic_dataset:
        synthetic_dataset = load_synthetic_dataset(synthetic_dataset, max_length)
        train_dataset = concatenate_datasets([train_dataset, synthetic_dataset])

    tokenized_train_dataset = tokenize_wsd_dataset(train_dataset, tokenizer, max_length)
    tokenized_test_dataset = tokenize_wsd_dataset(test_dataset, tokenizer, max_length)

    return DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True), DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=True)

""" Countdown dataset """

def tokenize_countdown_dataset(dataset, tokenizer, max_length):
    def tokenize_text(text, tokenizer, max_length):
        # no answer, just prompt
        return tokenizer(text["prompt"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_dataset = dataset.map(lambda x: tokenize_text(x, tokenizer, max_length), batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "idx"])
    return tokenized_dataset

def load_json_countdown(file_path):
    data = []
    with open(file_path, "r") as json_file:
        for line in json_file:
            line = line.strip()
            if line:
                json_data = json.loads(line)
                data.append((json_data["target"], json_data["num"]))
    print(data[0])
    return data

# special because no test dataset
def load_countdown_dataset(tokenizer, batch_size, max_length, from_json=False):
    SERIALIZED_PATH = "/data/c-aalag/countdown_dataset.pt"
    SERIALIZED_TOKENIZED_PATH = "/data/c-aalag/countdown_tokenized_dataset.pt"

    # this is taken from WSD dataset
    if from_json:
        train_dataset = load_json_countdown("countdown.json")
        processed_train_dataset = []
        for i, example in enumerate(train_dataset):
            prompt = WSD_PROMPT_FORMAT.format(target=example[0], numbers=example[1])
            processed_train_dataset.append({
                "prompt": prompt,
                "idx": i,
                "target": example[0],
                "numbers": example[1]})

        train_dataset = Dataset.from_list(processed_train_dataset)
    elif os.path.exists(SERIALIZED_PATH) and os.path.exists(SERIALIZED_TOKENIZED_PATH):
        train_dataset = torch.load(SERIALIZED_PATH, weights_only=False)
        tokenized_train_dataset = torch.load(SERIALIZED_TOKENIZED_PATH, weights_only=False)
        return DataLoader(tokenized_train_dataset, batch_size=1, shuffle=False), train_dataset
    else:
        train_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

        train_dataset = train_dataset.map(lambda example, idx: {
            "prompt": WSD_PROMPT_FORMAT.format(target=example["target"], numbers=example["nums"]),
            "target": example["target"],
            "numbers": example["nums"],
            "idx": idx},
            with_indices=True,
            batched=False,
        )

        torch.save(train_dataset, SERIALIZED_PATH)

    tokenized_train_dataset = tokenize_countdown_dataset(train_dataset, tokenizer, max_length=max_length)
    if not from_json:
        torch.save(train_dataset, SERIALIZED_TOKENIZED_PATH)

    return DataLoader(tokenized_train_dataset, batch_size=1, shuffle=False), train_dataset

def load_synthetic_dataset(synthetic_dataset, max_length):
    synthetic_dataset = json.load(open(synthetic_dataset))
    prompts = [WSD_PROMPT_FORMAT.format(target=example["target"], numbers=example["nums"]) for example in synthetic_dataset]
    synthetic_dataset = {"query": prompts, "completion": [example["chain_of_thought"] for example in synthetic_dataset]}
    synthetic_dataset = Dataset.from_dict(synthetic_dataset)
    return synthetic_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="warm_start_sft", choices=["warm_start_sft", "countdown"])
    parser.add_argument("--output_dir", type=str, default="data/smoltalk_processed")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    train_dataloader, test_dataloader = get_wsd_dataset(tokenizer, args.max_length, args.batch_size)