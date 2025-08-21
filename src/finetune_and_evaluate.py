import json
import time
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

# --- Configuration & Hyperparameters ---
BASE_MODEL = "gpt2-medium"
DATASET_PATH = "qapairs/medtronic_qa_training_data.json"
OUTPUT_DIR = "./gpt2-medium-finetuned"

# Hyperparameters for fine-tuning
LEARNING_RATE = 5e-5 # A common learning rate for fine-tuning GPT-2
BATCH_SIZE = 4
NUM_EPOCHS = 3

# --- Logging Setup ---
LOG_FILE = "training_log_gpt2_medium.txt"

def log_message(message):
    """Logs a message to both the console and a log file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_and_prepare_data():
    """Loads the dataset, using all data for training and a subset for evaluation."""
    log_message(f"Loading dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, "r") as f:
        qa_data = json.load(f)
    
    # Use all data for training as requested
    train_data = qa_data
    
    # Use the first 10 questions for benchmarking/evaluation
    test_data = qa_data[:10]

    log_message(f"Using all {len(train_data)} samples for training.")
    log_message(f"Using the first {len(test_data)} samples for benchmarking (note: this is a subset of the training data).")
    return train_data, test_data

def generate_response(model, tokenizer, question, device):
    """Generates a response from a model for a given question."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )
    end_time = time.time()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up the response to only show the answer part
    answer = response.replace(prompt, "").strip().split('\n')[0]
    
    inference_speed = end_time - start_time
    return answer, inference_speed

def benchmark_model(model, tokenizer, test_questions, device, model_name="Model"):
    """Evaluates a model on test questions and logs the results."""
    log_message("\n" + "="*50)
    log_message(f"  BENCHMARKING: {model_name}")
    log_message("="*50)
    
    total_inference_time = 0
    
    for i, qa_pair in enumerate(test_questions):
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        generated_answer, inference_speed = generate_response(model, tokenizer, question, device)
        total_inference_time += inference_speed
        
        log_message(f"\n--- Test Question {i+1} ---")
        log_message(f"Q: {question}")
        log_message(f"Expected A: {expected_answer}")
        log_message(f"Generated A: {generated_answer}")
        log_message(f"Inference Speed: {inference_speed:.4f} seconds")

    avg_inference_speed = total_inference_time / len(test_questions)
    log_message("\n--- Benchmark Summary ---")
    log_message(f"Average Inference Speed: {avg_inference_speed:.4f} seconds")
    log_message("Note: Accuracy and confidence are qualitative. Please review the generated vs. expected answers.")
    log_message("="*50 + "\n")

def fine_tune(train_data):
    """Fine-tunes the gpt2-medium model using LoRA."""
    log_message("\n" + "="*50)
    log_message("  STARTING FINE-TUNING PROCESS")
    log_message("="*50)

    # Log hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"Compute Setup: {device.upper()}")
    log_message(f"Model: {BASE_MODEL}")
    log_message(f"Learning Rate: {LEARNING_RATE}")
    log_message(f"Batch Size: {BATCH_SIZE}")
    log_message(f"Number of Epochs: {NUM_EPOCHS}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"], # Target modules for GPT-2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    log_message("\nTrainable Parameters:")
    model.print_trainable_parameters()

    # Prepare dataset
    dataset = Dataset.from_list(train_data)
    def preprocess_function(examples):
        formatted_texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
        tokenized = tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=256)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # Set up Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_total_limit=2,
        fp16=True if device == "cuda" else False,
        dataloader_pin_memory=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Start training
    log_message("\nStarting training...")
    trainer.train()
    
    # Save the final model
    log_message(f"Saving fine-tuned model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log_message("Fine-tuning complete!")

def main():
    """Orchestrates the entire benchmark and fine-tuning pipeline."""
    # Clear log file for a fresh run
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    train_data, test_data = load_and_prepare_data()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Baseline Benchmarking (Pre-Fine-Tuning) ---
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    benchmark_model(base_model, base_tokenizer, test_data, device, f"Pre-trained {BASE_MODEL}")
    del base_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Fine-Tuning ---
    fine_tune(train_data)
    
    # --- Post-Fine-Tuning Evaluation ---
    tuned_model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    from peft import PeftModel
    tuned_model = PeftModel.from_pretrained(tuned_model_base, OUTPUT_DIR)
    tuned_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    benchmark_model(tuned_model, tuned_tokenizer, test_data, device, f"Fine-Tuned {BASE_MODEL} with LoRA")

if __name__ == "__main__":
    main()
