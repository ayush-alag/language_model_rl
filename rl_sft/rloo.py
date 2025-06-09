import os
import argparse
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from dataloader import load_countdown_dataset
from eval_countdown import compute_score
from torch.amp import GradScaler
from common import get_device, set_seed
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
    return log_probabilities_response.mean(dim=-1)

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
            output = out.text
            full_ids = tokenizer.encode(prompt_texts[p_idx] + output, return_tensors="pt").to(model.device)
            gen_len = full_ids.size(1) - prompt_len
            logp = log_probs(model, full_ids, gen_len)[0]
            texts.append(output)
            log_probabilities.append(logp)
    return texts, torch.stack(log_probabilities)

def get_batch_loss(model, vllm_model, batch, device, tokenizer, dataset):
    idx = batch["idx"]
    prompt_ids = batch["input_ids"].to(device)
    texts, log_probabilities = sample_k(model, vllm_model, tokenizer, prompt_ids, args.k, args.max_new)
    torch.cuda.empty_cache()
    repeated_idx = idx.repeat_interleave(args.k).tolist()
    rewards = []
    for i, t in enumerate(texts):
        nums = dataset[repeated_idx[i]]["nums"]
        target = dataset[repeated_idx[i]]["target"]
        print(t, nums, target)
        rewards.append(compute_score(t, {"target": [target], "numbers": [nums]}))
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    return rloo_loss(rewards, log_probabilities)

def rloo_loss(rewards, log_probabilities):
    baseline = (rewards.sum() - rewards) / (rewards.size(0) - 1)
    adv = rewards - baseline

    # we dont pass gradients to the rewards (they are fixed)
    loss = -(adv.detach() * log_probabilities).mean()
    return loss

def train(args):
    wandb.init(
        project="rl-sft",
        name=args.experiment_name,
    )

    # this is all the same as SFT: load the model, dataset, tokenize
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True, use_fast=True)

    # vllm is for quick rollouts
    vllm_model = LLM(
        model=args.sft_checkpoint,
        tokenizer="Qwen/Qwen2.5-0.5B",
        device=device,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
    )

    train_loader, dataset = load_countdown_dataset(tokenizer, 1, args.max_length, False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps)

    scaler = GradScaler()
    global_step = 0
    max_score = 0
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            loss = get_batch_loss(model, vllm_model, batch, device, tokenizer, dataset)
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                # reload the model weights to vllm
                llm_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(model.state_dict())

            if (global_step + 1) % args.log_interval == 0:
                wandb.log({"loss": loss.item() * args.gradient_accumulation_steps, "step": global_step + 1})

            if (global_step + 1) % args.eval_steps == 0:
                eval_score = eval_countdown_vllm(vllm_model, 16, args.max_length, from_json=True)
                if eval_score > max_score:
                    max_score = eval_score
                    save_checkpoint(model, args.out_dir, f"best_model")
                print(f"Step {global_step}, Eval Score: {eval_score:.4f}")

            global_step += 1

        save_checkpoint(model, args.out_dir, f"epoch_{epoch+1}")

    wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sft_checkpoint", type=str, default="checkpoints/best_model")
    p.add_argument("--out_dir", type=str, default="checkpoints/best_rloo_model")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max_new", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=str, default="countdown")
    p.add_argument("--experiment_name", type=str, default="rloo")
    args = p.parse_args()
    set_seed(args.seed)
    train(args)