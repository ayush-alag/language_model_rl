import os
import argparse
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from dataloader import get_wsd_dataset, load_synthetic_dataset, load_countdown_dataset
from torch.amp import autocast, GradScaler
import wandb
import sys
import inspect
import tempfile

from eval_pipeline import eval_countdown_vllm
from common import get_device, set_seed, init_vllm, load_policy_into_vllm_instance
from vllm import SamplingParams

def get_batch_loss(model, batch, device):
    with autocast(device_type=device.type, dtype=torch.float16):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    return loss

def train_step(model, batch, device, optimizer, scheduler, scaler, args, global_step,
               test_dataloader, easy_eval_dataset, hard_eval_dataset, eval_vllm_model,
               eval_sampling_params, step_num, total_loss, min_test_loss):
    loss = get_batch_loss(model, batch, device)
    loss = loss / args.gradient_accumulation_steps
    scaler.scale(loss).backward()

    if (step_num + 1) % args.gradient_accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        wandb.log({
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/step": global_step,
        })

        if global_step % args.eval_steps == 0:
            eval_loss = evaluate_test_loss(model, test_dataloader, device, global_step)
            load_policy_into_vllm_instance(model, eval_vllm_model)
            countdown_score = eval_countdown_vllm(eval_vllm_model, easy_eval_dataset, args.max_length, eval_sampling_params, "countdown_outputs.json")
            countdown_score_hard = eval_countdown_vllm(eval_vllm_model, hard_eval_dataset, args.max_length, eval_sampling_params, "countdown_outputs_hard.json")
            wandb.log({
                "test/loss": eval_loss,
                "test/countdown_score_easy": countdown_score,
                "test/countdown_score_hard": countdown_score_hard,
                "test/step": global_step,
            })

            if eval_loss < min_test_loss:
                min_test_loss = eval_loss
                save_checkpoint(model, args.output_dir, f"best_model")
            print(f"Step {global_step}, Eval Loss: {eval_loss:.4f}")
    return global_step

def train(args, device, train_dataloader, test_dataloader, easy_eval_dataset, hard_eval_dataset, tokenizer, eval_vllm_model, eval_sampling_params, hard_train_dataloader=None):
    print("Initializing wandb")
    wandb.init(
        project="rl-sft",
        name=args.experiment_name,
    )
    print("Finished initializing wandb")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = num_training_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler()

    global_step = 0
    min_test_loss = float("inf")
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch + 1}")
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            if step == 0:
                print("Easy batch: ", batch)
            global_step = train_step(model, batch, device, optimizer, scheduler, scaler, args,
                                     global_step, test_dataloader, easy_eval_dataset, hard_eval_dataset,
                                     eval_vllm_model, eval_sampling_params, step, total_loss, min_test_loss)

        if hard_train_dataloader:
            print("Training on hard dataset")
            progress_bar = tqdm(hard_train_dataloader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                if step == 0:
                    print("Hard batch: ", batch)
                global_step = train_step(model, batch, device, optimizer, scheduler, scaler, args,
                                         global_step, test_dataloader, easy_eval_dataset, hard_eval_dataset,
                                         eval_vllm_model, eval_sampling_params, step, total_loss, min_test_loss)

        print(f"Epoch {epoch + 1} completed. avg loss: {total_loss / len(train_dataloader):.4f}")

    wandb.finish()

def evaluate_test_loss(model, test_dataloader, device, global_step):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in test_dataloader:
            loss = get_batch_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def save_checkpoint(model, output_dir, checkpoint_name):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, checkpoint_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dataset", action="store_true")
    parser.add_argument("--output_dir", type=str, default="/data/c-aalag/checkpoints_sft_synth_improved")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--experiment_name", type=str, default="wsd")
    parser.add_argument("--curriculum", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )

    args.synthetic_dataset = "countdown_warmstart_cot_100_improved.json" if args.synthetic_dataset else None

    easy_json_path = "countdown.json"
    hard_json_path = "countdown_heldout_prompts.json"
    _, easy_eval_dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length, json_path=easy_json_path)
    _, hard_eval_dataset = load_countdown_dataset(tokenizer, args.batch_size, args.max_length, json_path=hard_json_path)

    device = get_device()
    print(f"Using device: {device}")

    eval_vllm_model = init_vllm("Qwen/Qwen2.5-0.5B", device, args.seed)
    print("Finished initializing vllm")
    set_seed(args.seed)

    eval_sampling_params = SamplingParams(
        max_tokens=args.max_length,
        temperature=0.6,
        top_p=0.95,
        top_k=20
    )

    if args.curriculum:
        train_dataloader, test_dataloader = get_wsd_dataset(tokenizer, args.max_length, args.batch_size, args.synthetic_dataset, num_elements=3)
        hard_train_dataloader, _ = get_wsd_dataset(tokenizer, args.max_length, args.batch_size, args.synthetic_dataset, num_elements=4)
        train(args, device, train_dataloader, test_dataloader, easy_eval_dataset, hard_eval_dataset, tokenizer, eval_vllm_model, eval_sampling_params, hard_train_dataloader)
    else:
        train_dataloader, test_dataloader = get_wsd_dataset(tokenizer, args.max_length, args.batch_size, args.synthetic_dataset)
        train(args, device, train_dataloader, test_dataloader, easy_eval_dataset, hard_eval_dataset, tokenizer, eval_vllm_model, eval_sampling_params)