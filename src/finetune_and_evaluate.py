# finetune_and_evaluate.py

import json
import time
import torch
import os
import pickle
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel

# --- Configuration & Hyperparameters ---
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "qapairs/medtronic_qa_training_data.json"
OUTPUT_DIR = "./tinyllama-finetuned-adapter-cpu"
MERGED_MODEL_DIR = "./tinyllama-finetuned-merged" 

# Path to save generated Q&A pairs for the app
GENERATED_QA_OUTPUT_PATH = "generated_qa.pkl"

# Hyperparameters for fine-tuning
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
NUM_EPOCHS = 10

# --- Logging Setup ---
LOG_FILE = "training_log_tinyllama_cpu.txt"

def log_message(message):
    """Logs a message to both the console and a log file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_and_prepare_data():
    """Loads the dataset and uses the first 10 for evaluation."""
    log_message(f"Loading dataset from: {DATASET_PATH}")
    try:
        with open(DATASET_PATH, "r") as f:
            qa_data = json.load(f)
    except FileNotFoundError:
        log_message(f"Error: Dataset not found at {DATASET_PATH}")
        return None, None

    train_data = qa_data
    test_data = qa_data[:10]

    log_message(f"Using {len(train_data)} samples for training.")
    log_message(f"Using the first {len(test_data)} samples for benchmarking.")
    return train_data, test_data

def generate_response(model, tokenizer, question, device):
    """Generates a response from a transformers model."""
    start_time = time.time()

    prompt = (
        f"<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    end_time = time.time()
    inference_speed = end_time - start_time
    return answer, inference_speed

def benchmark_model(model_path, test_questions, model_name="Model"):
    """Evaluates a transformers model on test questions."""
    log_message("\n" + "="*50)
    log_message(f"  BENCHMARKING MODEL: {model_name}")
    log_message("="*50)

    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_inference_time = 0
    generated_qa_pairs = {}

    for i, qa_pair in enumerate(test_questions):
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"]

        generated_answer, inference_speed = generate_response(model, tokenizer, question, device)
        total_inference_time += inference_speed
        generated_qa_pairs[question] = generated_answer

        log_message(f"\n--- Test Question {i+1} ---")
        log_message(f"Q: {question}")
        log_message(f"Expected A: {expected_answer}")
        log_message(f"Generated A: {generated_answer}")
        log_message(f"Inference Speed: {inference_speed:.4f} seconds")

    avg_inference_speed = total_inference_time / len(test_questions)
    log_message("\n--- Benchmark Summary ---")
    log_message(f"Average Inference Speed: {avg_inference_speed:.4f} seconds")
    log_message("="*50 + "\n")

    return generated_qa_pairs

def fine_tune(train_data):
    """Fine-tunes the TinyLlama model using LoRA on the CPU."""
    log_message("\n" + "="*50)
    log_message("  STARTING CPU FINE-TUNING PROCESS (LoRA)")
    log_message("="*50)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = Dataset.from_list(train_data)
    def preprocess_function(examples):
        formatted_texts = [
            f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{q}</s>\n<|assistant|>\n{a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        return tokenizer(formatted_texts, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE, logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10, save_steps=50, optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset,
        tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    log_message("\nStarting training...")
    trainer.train()

    log_message(f"Saving fine-tuned adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log_message("Fine-tuning complete!")

def main():
    """Orchestrates the entire benchmark and fine-tuning pipeline."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    train_data, test_data = load_and_prepare_data()
    if not train_data:
        return

    # --- Fine-Tuning ---
    fine_tune(train_data)

    # --- Merge and Save Model ---
    log_message("\n" + "="*50)
    log_message("  MERGING AND SAVING MODEL")
    log_message("="*50)

    device = "cpu"
    log_message("\nLoading base model to apply fine-tuned adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    log_message("Merging adapter with the base model...")
    merged_model = peft_model.merge_and_unload()

    log_message(f"Saving merged model to {MERGED_MODEL_DIR}...")
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    log_message("Model merged and saved.")

    # --- Post-Fine-Tuning Evaluation ---
    generated_qa = benchmark_model(
        model_path=MERGED_MODEL_DIR,
        test_questions=test_data,
        model_name=f"Fine-Tuned {BASE_MODEL}"
    )

    # --- Save the generated Q&A pairs ---
    log_message(f"\nSaving {len(generated_qa)} generated Q&A pairs to {GENERATED_QA_OUTPUT_PATH}...")
    try:
        with open(GENERATED_QA_OUTPUT_PATH, 'wb') as f:
            pickle.dump(generated_qa, f)
        log_message("Successfully saved generated Q&A pairs.")
    except Exception as e:
        log_message(f"Error saving generated Q&A pairs: {e}")

if __name__ == "__main__":
    main()