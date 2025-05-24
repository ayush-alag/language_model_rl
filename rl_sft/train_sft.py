import os
import argparse
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from dataloader import load_dataset_and_tokenize, set_seed
from torch.amp import autocast, GradScaler
import wandb

def get_batch_loss(model, batch, device):
    with autocast(device_type=device.type, dtype=torch.float16):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    return loss

# run on bare metal if we can
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

def train(args):
    wandb.init(
        project="rl-sft",
        name=args.task,
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

    model = torch.compile(model)

    # TODO: this reads the entire dataset; we could cache this
    train_dataloader, test_dataloader = load_dataset_and_tokenize(args.task, args.max_length, args.batch_size)

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
                    eval_loss = evaluate(model, test_dataloader, device, global_step)
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

def evaluate(model, test_dataloader, device, global_step):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in test_dataloader:
            loss = get_batch_loss(model, batch, device)
            total_loss += loss.item()

    wandb.log({
        "test/loss": total_loss / len(test_dataloader),
        "test/step": global_step,
    })

    return total_loss / len(test_dataloader)

def save_checkpoint(model, output_dir, checkpoint_name):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, checkpoint_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="warm_start_sft", choices=["smoltalk_sft", "warm_start_sft"])
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)