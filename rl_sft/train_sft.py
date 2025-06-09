import os
import argparse
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from dataloader import get_wsd_dataset, load_synthetic_dataset, load_countdown_dataset
from torch.amp import autocast, GradScaler
import wandb
from eval_pipeline import eval_countdown_vllm
from common import get_device, set_seed

def get_batch_loss(model, batch, device):
    with autocast(device_type=device.type, dtype=torch.float16):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    return loss

def train(args, eval_dataset):
    wandb.init(
        project="rl-sft",
        name="wsd",
        config={
            "batch_size": args.batch_size,
            "max_lr": args.learning_rate,
            "epochs": args.num_epochs,
            "max_length": args.max_length,
        }
    )

    device = get_device()
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True).to(device)

    train_dataloader, test_dataloader = get_wsd_dataset(args.max_length, args.batch_size, args.synthetic_dataset)

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
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            loss = get_batch_loss(model, batch, device)
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
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
                    countdown_score = eval_countdown_vllm(model, eval_dataset, args.max_length)
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/countdown_score": countdown_score,
                        "eval/step": global_step,
                    })

                    if eval_loss < min_test_loss:
                        min_test_loss = eval_loss
                        save_checkpoint(model, args.output_dir, f"best_model")
                    print(f"Step {global_step}, Eval Loss: {eval_loss:.4f}")

                # if global_step % args.save_steps == 0:
                #     save_checkpoint(model, args.output_dir, f"checkpoint-{global_step}")

            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix(loss=loss.item())

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
    parser.add_argument("--output_dir", type=str, default="checkpoints_sft")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    if args.synthetic_dataset:
        args.synthetic_dataset = "countdown_warmstart_cot_100.json"

    _, eval_dataset = load_countdown_dataset(args.batch_size, args.max_length, from_json=True)

    set_seed(args.seed)
    train(args, eval_dataset)