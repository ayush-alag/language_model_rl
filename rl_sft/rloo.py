import os
import argparse
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from dataloader import load_countdown_dataset
from eval_countdown import compute_score
from torch.amp import GradScaler
from common import get_device, set_seed, init_vllm, load_policy_into_vllm_instance
from train_sft import save_checkpoint
from eval_pipeline import eval_countdown_vllm
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from vllm import LLM, SamplingParams

def log_probs(model, input_ids, num_generated_tokens):
    out = model(input_ids=input_ids).logits[:, :-1]
    labels = input_ids[:, 1:]
    log_softmax = F.log_softmax(out, dim=-1)
    log_probabilities_y = log_softmax.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    log_probabilities_response = log_probabilities_y[:, -num_generated_tokens:]
    # return log_probabilities_response.mean(dim=-1)
    return log_probabilities_response.sum(dim=-1)

def sample_k(model, vllm_model, tokenizer, prompt_ids, k, max_new):
    prompt_texts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
    sampling_params = SamplingParams(
        n=k,
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_new,
    )

    outputs = vllm_model.generate(prompt_texts, sampling_params)
    # vllm_model.llm_engine.reset()
    texts = []
    log_probabilities = []
    for p_idx, res in enumerate(outputs):
        prompt_len = prompt_ids[p_idx].size(0)
        for out in res.outputs:
            # output = out.text
            # full_ids = tokenizer.encode(prompt_texts[p_idx] + output, return_tensors="pt").to(model.device)
            # gen_len = full_ids.size(1) - prompt_len
            # logp = log_probs(model, full_ids, gen_len)[0]

            output_ids = torch.tensor(out.token_ids, device=model.device)
            full_ids = torch.cat([prompt_ids[p_idx], output_ids], dim=0).unsqueeze(0)
            gen_len = full_ids.size(1) - prompt_len
            logp = log_probs(model, full_ids, gen_len)[0]

            texts.append(out.text)
            log_probabilities.append(logp)
    return texts, torch.stack(log_probabilities)

def get_batch_loss(model, vllm_model, batch, device, tokenizer, dataset, k):
    idx = batch["idx"]
    prompt_ids = batch["input_ids"].to(device)
    texts, log_probabilities = sample_k(model, vllm_model, tokenizer, prompt_ids, args.k, args.max_new)
    torch.cuda.empty_cache()
    repeated_idx = idx.repeat_interleave(args.k).tolist()
    rewards = []
    for i, t in enumerate(texts):
        nums = dataset[repeated_idx[i]]["numbers"]
        target = dataset[repeated_idx[i]]["target"]
        # print(t, nums, target)
        rewards.append(compute_score(t, {"target": [target], "numbers": [nums]}))
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    return rloo_loss(rewards, log_probabilities, k)

# def rloo_loss(rewards, log_probabilities):
#     k = rewards.size(0)
#     rewards = rewards.view(-1, k)
#     log_probabilities = log_probabilities.view(-1, k)
#     baseline = (rewards.sum(1, keepdim=True) - rewards) / (k - 1)
#     adv = rewards - baseline

#     # we dont pass gradients to the rewards (they are fixed)
#     loss = -(adv.detach() * log_probabilities).mean()
#     return loss

def rloo_loss(rewards, log_probabilities, k):
    """RLOO loss computation with proper k-sample grouping"""
    batch_size = rewards.size(0) // k

    # Reshape to [batch_size, k]
    rewards = rewards.view(batch_size, k)
    log_probabilities = log_probabilities.view(batch_size, k)

    # RLOO baseline: average of other k-1 samples
    baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (k - 1)
    advantages = rewards - baseline

    # Policy gradient loss
    loss = -(advantages.detach() * log_probabilities).mean()

    # Add debugging info
    if torch.rand(1).item() < 0.01:  # Log 1% of the time
        print(f"RLOO Debug - Batch: {batch_size}, K: {k}")
        print(f"Rewards shape: {rewards.shape}, range: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"Advantages mean: {advantages.mean():.3f}, std: {advantages.std():.3f}")
        print(f"Loss: {loss.item():.6f}")

    return loss

def train(args, device, model, tokenizer, vllm_model, train_loader, dataset, easy_eval_dataset, hard_eval_dataset, max_steps):
    wandb.init(
        project="rl-sft",
        name=args.experiment_name,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps)

    scaler = GradScaler()
    global_step = 0
    max_score = 0
    best_countdown_score = 0
    best_countdown_score_hard = 0
    for epoch in range(args.num_epochs):
        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            load_policy_into_vllm_instance(model, vllm_model)
            loss = get_batch_loss(model, vllm_model, batch, device, tokenizer, dataset, args.k)
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                # reload the model weights to vllm
                load_policy_into_vllm_instance(model, vllm_model)

                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": global_step,
                })

            if (global_step + 1) % args.eval_steps == 0:
                load_policy_into_vllm_instance(model, vllm_model)
                countdown_score = eval_countdown_vllm(vllm_model, easy_eval_dataset, args.max_length, eval_sampling_params, "countdown_outputs_rl.json")
                countdown_score_hard = eval_countdown_vllm(vllm_model, hard_eval_dataset, args.max_length, eval_sampling_params, "countdown_outputs_hard_rl.json")

                wandb.log({
                    "test/countdown_score_easy": countdown_score,
                    "test/countdown_score_hard": countdown_score_hard,
                    "test/step": global_step,
                })

                if countdown_score_hard > best_countdown_score_hard:
                    best_countdown_score_hard = countdown_score_hard
                    # rename the json files
                    os.rename("countdown_outputs_hard_rl.json", "countdown_outputs_hard_rl_best.json")
                    save_checkpoint(model, args.out_dir, f"best_model")
                print(f"Step {global_step}, Eval Score: {countdown_score:.4f}, Hard Eval Score: {countdown_score_hard:.4f}")

            if global_step > max_steps:
                break

            global_step += 1

        save_checkpoint(model, args.out_dir, f"epoch_{epoch+1}")

    wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sft_checkpoint", type=str, default="/data/c-aalag/checkpoints_sft_synth_improved/best_model")
    p.add_argument("--out_dir", type=str, default="/data/c-aalag/checkpoints_rl/best_rloo_model")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_new", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=str, default="countdown")
    p.add_argument("--experiment_name", type=str, default="rloo")
    p.add_argument("--batch_size", type=int, default=1) # one at a time?
    p.add_argument("--max_steps", type=int, default=250)
    args = p.parse_args()
    set_seed(args.seed)

    print("Setting up model")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        trust_remote_code=True,
    ).to(device)

    print("Loading datasets")
    easy_json_path = "countdown.json"
    hard_json_path = "countdown_heldout_prompts.json"
    _, easy_eval_dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length, json_path=easy_json_path)
    _, hard_eval_dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length, json_path=hard_json_path)

    print("Initializing vllm")
    vllm_model = init_vllm("Qwen/Qwen2.5-0.5B", device, args.seed)
    load_policy_into_vllm_instance(model, vllm_model)
    print("Finished initializing vllm")

    eval_sampling_params = SamplingParams(
        max_tokens=args.max_length,
        temperature=0.6,
        top_p=0.95,
        top_k=20
    )

    print("Loading train loader")
    train_loader, dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length, False)

    print("Starting training")
    train(args, device, model, tokenizer, vllm_model, train_loader, dataset, easy_eval_dataset, hard_eval_dataset, args.max_steps)