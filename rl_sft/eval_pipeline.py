from eval_countdown import compute_score
from dataloader import load_dataset_and_tokenize
from transformers import AutoModelForCausalLM
from train_sft import get_device

def get_trained_model(model_path="checkpoints/best_model"):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True).to(device)
    return model

"""Args:
    solution_str: the solution text
    ground_truth: dictionary containing target number and available numbers
    method: the method to extract the solution
    format_score: the score for correct format but wrong answer
    score: the score for the correct answer
"""

def get_countdown_test_dataset():
    dataset, _ = load_dataset_and_tokenize("countdown", max_length=1024, batch_size=16)
    return dataset

def eval_countdown(model, dataset):
    for batch in dataset:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        print(loss)