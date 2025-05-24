from eval_countdown import compute_score
from dataloader import load_dataset_and_tokenize, load_countdown_dataset
from transformers import AutoModelForCausalLM
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

def eval_countdown(model_path, batch_size, max_length):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True)

    dataset = load_countdown_dataset(batch_size, max_length)
    model.to(device).eval()
    print("here")

    # now we need to evaluate on the countdown examples
    with torch.no_grad():
        scores = []
        num_samples = 0
        MAX_SAMPLES = 10
        for example in dataset:
            if num_samples >= MAX_SAMPLES:
                break

            num_samples += 1

            print(example.column_names)
            input_ids = torch.tensor(example["input_ids"]).to(model.device)
            mask = torch.tensor(example["attention_mask"]).to(model.device)
            generated = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=128)
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            gt = {"target": example["target"], "numbers": example["numbers"]}
            scores.append(compute_score(decoded, gt))
        avg_score = sum(scores) / len(scores)
        print(f"Countdown average score: {avg_score:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    eval_countdown(args.model_path, args.batch_size, args.max_length)