from eval_countdown import compute_score, extract_solution
from dataloader import get_wsd_dataset, load_countdown_dataset
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

def eval_wsd_model_path(model_path, batch_size, max_length):
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
    _, test_dataloader = get_wsd_dataset(max_length, batch_size)

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
    return avg_score

def eval_countdown_model_path(model_path, eval_dataset, max_length):
    device = get_device()
    model = LLM(
        model=model_path,
        tokenizer="Qwen/Qwen2.5-0.5B",
        device=device,
    )

    return eval_countdown_vllm(model, eval_dataset, max_length)

def eval_countdown_vllm(vllm_model, eval_dataset, max_length):
    device = get_device()
    sampling_params = SamplingParams(
        max_tokens=max_length,
        temperature=0.0,
        top_p=1.0,
    )

    prompts = [example["prompt"] for example in eval_dataset]
    print(prompts[0])
    print(eval_dataset[0])

    outputs = vllm_model.generate(prompts, sampling_params=sampling_params)
    scores = []
    with open("countdown_outputs.json", "w") as f:
        for i, output in enumerate(outputs):
            nums = eval_dataset[i]["numbers"]
            target = eval_dataset[i]["target"]
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
    # eval_wsd_model_path(args.model_path, args.batch_size, args.max_length)

    _, eval_dataset = load_countdown_dataset(args.batch_size, args.max_length, args.from_json)
    eval_countdown_model_path(args.model_path, eval_dataset, args.max_length)