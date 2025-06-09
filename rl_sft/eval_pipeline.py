from eval_countdown import compute_score, extract_solution
from dataloader import get_wsd_dataset, load_countdown_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import get_device
import argparse
import torch
import json
from vllm import LLM, SamplingParams

def eval_wsd_model_path(model_path, eval_dataset, tokenizer, max_length):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True)
    model.to(device).eval()

    # load the test dataset for wsd
    scores = []
    for example in eval_dataset:
        input_ids = torch.tensor(example["input_ids"]).to(model.device)
        mask = torch.tensor(example["attention_mask"]).to(model.device)
        generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=128)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        gt = {"target": example["target"], "numbers": example["numbers"]}
        scores.append(compute_score(decoded, gt))
    avg_score = sum(scores) / len(scores)
    print(f"wsd average score: {avg_score:.3f}")
    return avg_score

def eval_countdown_model_path(model_path, eval_dataset, max_length, eval_sampling_params, json_path):
    device = get_device()
    model = LLM(
        model=model_path,
        tokenizer="Qwen/Qwen2.5-0.5B",
        device=device,
    )

    return eval_countdown_vllm(model, eval_dataset, max_length, eval_sampling_params, json_path)

def eval_countdown_vllm(vllm_model, eval_dataset, max_length, sampling_params, json_path):
    prompts = [example["prompt"] for example in eval_dataset]
    outputs = vllm_model.generate(prompts, sampling_params=sampling_params)
    outputs = sorted(outputs, key=lambda o: int(o.request_id))
    print(outputs[0])
    scores = []
    with open(json_path, "w") as f:
        for i, output in enumerate(outputs):
            nums = eval_dataset[i]["numbers"]
            target = eval_dataset[i]["target"]
            response = extract_solution(output.outputs[0].text)
            if not response:
                response = ""

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

def eval_countdown(model_path, dataloader, dataset, tokenizer, batch_size, max_length):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True)
    model.to(device).eval()

    scores = []
    i = 0
    for example in dataloader:
        print(f"example {i}: {example}")
        input_ids = torch.tensor(example["input_ids"]).to(model.device)
        mask = torch.tensor(example["attention_mask"]).to(model.device)
        generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_length)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(dataset[i])
        gt = {"target": [dataset[i]["target"]], "numbers": [dataset[i]["numbers"]]}
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
    parser.add_argument("--json_path", action="store_true", default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    eval_sampling_params = SamplingParams(
        max_tokens=args.max_length,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    if args.json_path:
        easy_json_path = "countdown.json"
        hard_json_path = "countdown_heldout_prompts.json"
        easy_dataloader, easy_eval_dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length, json_path=easy_json_path)
        hard_dataloader, hard_eval_dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length, json_path=hard_json_path)

        eval_countdown_model_path(args.model_path, easy_eval_dataset, args.max_length, eval_sampling_params, "countdown_outputs.json")
        eval_countdown_model_path(args.model_path, hard_eval_dataset, args.max_length, eval_sampling_params, "countdown_outputs_hard.json")
    else:
        dataloader, eval_dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length)
        eval_countdown_model_path(args.model_path, eval_dataset, args.max_length, eval_sampling_params, "countdown_outputs.json")
    # eval_countdown(args.model_path, dataloader, eval_dataset, tokenizer, args.batch_size, args.max_length)