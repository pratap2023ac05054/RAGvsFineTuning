# finetune_and_evaluate.py

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
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split

# Import the guardrail functions
from guardrails import validate_query, validate_response

# --- Configuration & Hyperparameters ---
BASE_MODEL = "HuggingFaceH4/zephyr-7b-beta"
DATASET_PATH = "qapairs/medtronic_qa_training_data.json"
OUTPUT_DIR = "./zephyr-finetuned-fast"

# Hyperparameters for fine-tuning
LEARNING_RATE = 2e-5
BATCH_SIZE = 1 # Keep batch size low for large models
NUM_EPOCHS = 1 # Reduced for faster training

# --- Logging Setup ---
LOG_FILE = "training_log_zephyr_fast.txt"

def log_message(message):
    """Logs a message to both the console and a log file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_and_prepare_data():
    """Loads the dataset, splits it, and returns train/test sets."""
    log_message(f"Loading dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, "r") as f:
        qa_data = json.load(f)
    
    train_data, test_data = train_test_split(
        qa_data, 
        test_size=max(0.1, min(1.0, 10 / len(qa_data))), 
        random_state=42
    )
    
    if len(test_data) < 10 and len(qa_data) >= 10:
        test_data = qa_data[:10]
        train_data = qa_data[10:]

    log_message(f"Dataset split: {len(train_data)} training samples, {len(test_data)} test samples.")
    return train_data, test_data

def generate_response(model, tokenizer, question, device):
    """Generates a response from a model for a given question."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {question}\nAnswer:"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("<|assistant|>")[1].strip() if "<|assistant|>" in response else response
    
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
        
        is_valid, message = validate_query(question)
        
        log_message(f"\n--- Test Question {i+1} ---")
        log_message(f"Q: {question}")
        log_message(f"Input Guardrail: {'PASSED' if is_valid else f'FAILED ({message})'}")

        if not is_valid:
            continue

        generated_answer, inference_speed = generate_response(model, tokenizer, question, device)
        total_inference_time += inference_speed
        
        validation_result = validate_response(generated_answer, [expected_answer])

        log_message(f"Expected A: {expected_answer}")
        log_message(f"Generated A: {generated_answer}")
        log_message(f"Inference Speed: {inference_speed:.4f} seconds")
        if validation_result["pass"]:
            log_message("Output Guardrail: ✅ PASSED")
        else:
            log_message(f"Output Guardrail: ⚠️ FLAGGED - Reason: {validation_result['reason']}")
            log_message(f"   Details: {validation_result.get('details', 'N/A')}")

    avg_inference_speed = total_inference_time / len(test_questions)
    log_message("\n--- Benchmark Summary ---")
    log_message(f"Average Inference Speed: {avg_inference_speed:.4f} seconds")
    log_message("="*50 + "\n")

def fine_tune(train_data):
    """Fine-tunes the Zephyr-7B model using LoRA."""
    log_message("\n" + "="*50)
    log_message("  STARTING FINE-TUNING PROCESS")
    log_message("="*50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"Compute Setup: {device.upper()}")
    log_message(f"Learning Rate: {LEARNING_RATE}")
    log_message(f"Batch Size: {BATCH_SIZE}")
    log_message(f"Number of Epochs: {NUM_EPOCHS}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        log_message("Warning: Running on CPU. Training will be extremely slow.")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    log_message("\nTrainable Parameters:")
    model.print_trainable_parameters()

    dataset = Dataset.from_list(train_data)
    def preprocess_function(examples):
        questions = examples["question"]
        answers = examples["answer"]
        texts = []
        for q, a in zip(questions, answers):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {q}\nAnswer: {a}"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False) + tokenizer.eos_token
            texts.append(text)
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs", logging_steps=10,
        save_total_limit=2,
        fp16=True if device == "cuda" else False,
        dataloader_pin_memory=False
    )
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    log_message("\nStarting training...")
    trainer.train()
    
    log_message(f"Saving fine-tuned model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log_message("Fine-tuning complete!")

def main():
    """Orchestrates the entire benchmark and fine-tuning pipeline."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    train_data, test_data = load_and_prepare_data()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Baseline Benchmarking ---
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", load_in_4bit=True)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    benchmark_model(base_model, base_tokenizer, test_data, device, f"Pre-trained {BASE_MODEL}")
    del base_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Fine-Tuning ---
    fine_tune(train_data)
    
    # --- Post-Fine-Tuning Evaluation ---
    if device == "cuda":
        tuned_model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", load_in_4bit=True)
    else:
        tuned_model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
        
    from peft import PeftModel
    tuned_model = PeftModel.from_pretrained(tuned_model_base, OUTPUT_DIR)
    tuned_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    benchmark_model(tuned_model, tuned_tokenizer, test_data, device, f"Fine-Tuned {BASE_MODEL} with LoRA")

if __name__ == "__main__":
    main()
