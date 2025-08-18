import os, json, math, argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType

TEMPLATE = (
    "### Instruction\n"
    "Answer the question concisely. If unknown, say \"I don't know\".\n\n"
    "### Question\n{q}\n\n"
    "### Response\n"
)

def build_prompt(q: str) -> str:
    return TEMPLATE.format(q=q)

def tokenize_and_mask(examples, tokenizer, max_len=512):
    prompts = [build_prompt(q) for q in examples["question"]]
    answers = [a + tokenizer.eos_token for a in examples["answer"]]

    model_inputs = tokenizer(prompts, add_special_tokens=False, truncation=True, max_length=max_len)
    with tokenizer.as_target_tokenizer():
        label_tokens = tokenizer(answers, add_special_tokens=False, truncation=True, max_length=max_len)

    input_ids, attention_mask, labels = [], [], []
    for p_ids, l_ids in zip(model_inputs["input_ids"], label_tokens["input_ids"]):
        inp = p_ids + l_ids
        att = [1] * len(inp)
        lab = [-100] * len(p_ids) + l_ids  # mask prompt, learn only answer
        input_ids.append(inp[:max_len])
        attention_mask.append(att[:max_len])
        labels.append(lab[:max_len])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class Collator:
    def __init__(self, tokenizer):
        self.tk = tokenizer
    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        def pad(seq, pad_id):
            return seq + [pad_id] * (max_len - len(seq))
        input_ids = [pad(x["input_ids"], self.tk.pad_token_id) for x in batch]
        attention_mask = [pad(x["attention_mask"], 0) for x in batch]
        labels = [pad(x["labels"], -100) for x in batch]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2")  # or "gpt2-medium"
    ap.add_argument("--train_path", default="data/train.jsonl")
    ap.add_argument("--output_dir", default="out/gpt2-lora-qa")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Hyperparameters ===")
    print(json.dumps({
        "model_name": args.model_name,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_len": args.max_len,
        "compute": device
    }, indent=2))

    ds = load_dataset("json", data_files={"train": args.train_path})
    tk = AutoTokenizer.from_pretrained(args.model_name)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token

    tokenized = ds["train"].map(
        tokenize_and_mask, fn_kwargs={"tokenizer": tk, "max_len": args.max_len},
        batched=True, remove_columns=ds["train"].column_names
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tk))

    # PEFT LoRA
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention proj layers
        bias="none"
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    collator = Collator(tk)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=torch.cuda.is_available() and not (torch.cuda.is_bf16_supported()),
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tk.save_pretrained(args.output_dir)
    print("Saved LoRA-tuned adapter to:", args.output_dir)

if __name__ == "__main__":
    main()