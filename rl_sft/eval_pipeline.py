from eval_countdown import compute_score, extract_solution
from dataloader import load_dataset_and_tokenize, load_countdown_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_sft import get_device
import argparse
import torch
import json
from vllm import LLM, SamplingParams

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
        input_ids = torch.tensor(example["input_ids"]).to(model.device)
        mask = torch.tensor(example["attention_mask"]).to(model.device)
        generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=128)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        gt = {"target": example["target"], "numbers": example["numbers"]}
        scores.append(compute_score(decoded, gt))
    avg_score = sum(scores) / len(scores)
    print(f"wsd average score: {avg_score:.3f}")

def write_output_json(model, tokenizer, dataloader, dataset):
    with open("countdown_outputs.jsonl", "w") as f:
        for example in dataloader:
            input_ids = torch.tensor(example["input_ids"]).to(model.device)
            mask = torch.tensor(example["attention_mask"]).to(model.device)
            generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=1028)
            idx = example["idx"]
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            solution = extract_solution(decoded)

            if not solution:
                solution = ""

            dict_item = {
                "target": dataset[idx]["target"],
                "numbers": dataset[idx]["numbers"][0],
                "solution": solution,
            }
            print(dict_item)

            f.write(json.dumps(dict_item) + "\n")

def get_scores(model, tokenizer, dataloader, dataset):
    scores = []
    # batch size is 1
    for example in dataloader:
        input_ids = torch.tensor(example["input_ids"]).to(model.device)
        mask = torch.tensor(example["attention_mask"]).to(model.device)
        generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=1028)
        idx = example["idx"]

        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(dataset[idx])
        gt = {"target": dataset[idx]["target"], "numbers": dataset[idx]["nums"]}
        # print(f"example {idx}: {decoded}")
        # print(gt)
        score = compute_score(decoded, gt)
        print(f"example {idx}: {score}")
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print(f"countdown average score: {avg_score:.3f}")

def eval_countdown(model_path, batch_size, max_length, from_json=False):
    device = get_device()
    print(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    dataloader, dataset = load_countdown_dataset(batch_size, max_length, from_json)

    if not from_json:
        get_scores(model, tokenizer, dataloader, dataset)
    else:
        write_output_json(model, tokenizer, dataloader, dataset)

def eval_countdown_model_path(model_path, batch_size, max_length, from_json=False):
    device = get_device()
    model = LLM(
        model=model_path,
        tokenizer="Qwen/Qwen2.5-0.5B",
        device=device,
    )

    return eval_countdown_vllm(model, batch_size, max_length, from_json)

def eval_countdown_vllm(vllm_model, batch_size, max_length, from_json=False):
    device = get_device()
    sampling_params = SamplingParams(
        max_tokens=max_length,
        temperature=0.0,
        top_p=1.0,
    )

    _, dataset = load_countdown_dataset(batch_size, max_length, from_json)
    print(dataset[0])
    prompts = [example["prompt"] for example in dataset]
    print(prompts[0])

    outputs = vllm_model.generate(prompts, sampling_params=sampling_params)
    scores = []
    with open("countdown_outputs.json", "w") as f:
        for i, output in enumerate(outputs):
            nums = dataset[i]["numbers"]
            target = dataset[i]["target"]
            response = extract_solution(output.outputs[0].text)
            if not response:
                response = ""

            print(f"target: {target}, nums: {nums}, response: {response}")
            f.write(json.dumps({
                "num": nums,
                "response": response,
                "target": target,
            }, separators=(",", ":")) + "\n")

            score = compute_score(response, {"target": [target], "numbers": [nums]}, is_equation=True)
            scores.append(score)
    avg_score = sum(scores) / len(scores)
    print(f"countdown average score: {avg_score:.3f}")
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--from_json", action="store_true")
    args = parser.parse_args()
    # eval_wsd(args.model_path, args.batch_size, args.max_length)
    eval_countdown_model_path(args.model_path, args.batch_size, args.max_length, args.from_json)