from eval_countdown import compute_score
from dataloader import load_dataset_and_tokenize, load_countdown_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_sft import get_device
import argparse
import torch

"""Args:
    solution_str: the solution text
    ground_truth: dictionary containing target number and available numbers
    method: the method to extract the solution
    format_score: the score for correct format but wrong answer
    score: the score for the correct answer
"""

def eval_wsd(model_path, batch_size, max_length):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    # load the test dataset for wsd
    _, test_dataloader = load_dataset_and_tokenize("warm_start_sft", max_length, batch_size)

    scores = []
    for example in test_dataloader:
        print(example.keys())
        input_ids = torch.tensor(example["input_ids"]).to(model.device)
        mask = torch.tensor(example["attention_mask"]).to(model.device)
        generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=128)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(decoded)
        gt = {"target": example["target"], "numbers": example["numbers"]}
        scores.append(compute_score(decoded, gt))
    avg_score = sum(scores) / len(scores)
    print(f"wsd average score: {avg_score:.3f}")

def eval_countdown(model_path, batch_size, max_length):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    dataloader, dataset = load_countdown_dataset(batch_size, max_length)

    scores = []
    i = 0
    for example in dataloader:
        input_ids = torch.tensor(example["input_ids"]).to(model.device)
        mask = torch.tensor(example["attention_mask"]).to(model.device)
        generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=2056)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        gt = {"target": dataset[i]["target"], "numbers": dataset[i]["nums"]}
        print(f"example {i}: {decoded}")
        print(gt)
        scores.append(compute_score(decoded, gt))

        i += 1

    avg_score = sum(scores) / len(scores)
    print(f"countdown average score: {avg_score:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    # eval_wsd(args.model_path, args.batch_size, args.max_length)
    eval_countdown(args.model_path, args.batch_size, args.max_length)