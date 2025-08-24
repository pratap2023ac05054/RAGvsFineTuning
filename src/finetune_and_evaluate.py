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
BASE_MODEL = "openai-community/gpt2-medium"
DATASET_PATH = "qapairs/medtronic_qa_training_data.json"
ADAPTER_OUTPUT_DIR = "./gpt2-medium-finetuned-adapter"
FINETUNED_MODEL_PATH = "./gpt2-medium-finetuned" # Directory to save the final merged model

# Path to save generated Q&A pairs for the app
GENERATED_QA_OUTPUT_PATH = "generated_qa.pkl"

# Hyperparameters for fine-tuning
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
NUM_EPOCHS = 10

# --- Logging Setup ---
LOG_FILE = "training_log_gpt2_medium.txt"

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
    
    # Simple Q&A prompt format
    prompt = f"Question: {question}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    answer = generated_text.replace(prompt, "").strip()

    end_time = time.time()
    inference_speed = end_time - start_time
    return answer, inference_speed

def benchmark_model(model_path, test_questions, model_name="Model"):
    """Evaluates a transformers model on test questions."""
    log_message("\n" + "="*50)
    log_message(f"  BENCHMARKING MODEL: {model_name}")
    log_message("="*50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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
    """Fine-tunes the gpt2-medium model using LoRA."""
    log_message("\n" + "="*50)
    log_message("  STARTING FINE-TUNING PROCESS (LoRA)")
    log_message("="*50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    # LoRA config targeted for GPT-2's attention mechanism
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = Dataset.from_list(train_data)
    def preprocess_function(examples):
        # Format the text with a clear separator and an end-of-sequence token
        formatted_texts = [
            f"Question: {q}\n\nAnswer: {a}{tokenizer.eos_token}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        return tokenizer(formatted_texts, truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=ADAPTER_OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE, logging_dir=f"{ADAPTER_OUTPUT_DIR}/logs",
        logging_steps=10, save_steps=50, optim="adamw_torch",
        use_cpu=True if device == "cpu" else False, # Ensure Hugging Face Trainer uses CPU if needed
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset,
        tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    log_message("\nStarting training...")
    trainer.train()

    log_message(f"Saving fine-tuned adapter to {ADAPTER_OUTPUT_DIR}")
    trainer.save_model(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
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

    # --- Merge and Save the Final Model ---
    log_message("\n" + "="*50)
    log_message("  MERGING ADAPTER AND SAVING FINAL MODEL")
    log_message("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"\nLoading base model ({BASE_MODEL}) on {device} to merge adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    
    # Load the LoRA adapter
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_OUTPUT_DIR)
    
    log_message("Merging adapter with the base model...")
    merged_model = peft_model.merge_and_unload()
    
    log_message(f"Saving final merged model to {FINETUNED_MODEL_PATH}...")
    merged_model.save_pretrained(FINETUNED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_PATH)
    log_message("Model merged and saved successfully.")

    # --- Post-Fine-Tuning Evaluation ---
    generated_qa = benchmark_model(
        model_path=FINETUNED_MODEL_PATH,
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