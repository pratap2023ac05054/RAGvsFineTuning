import os, json, argparse, random
from pathlib import Path
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

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
        lab = [-100] * len(p_ids) + l_ids
        input_ids.append(inp[:max_len])
        attention_mask.append(att[:max_len])
        labels.append(lab[:max_len])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class Collator:
    def __init__(self, tokenizer): self.tk = tokenizer
    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        def pad(seq, pad_id): return seq + [pad_id]*(max_len - len(seq))
        import torch
        return {
            "input_ids": torch.tensor([pad(x["input_ids"], self.tk.pad_token_id) for x in batch]),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
            "labels": torch.tensor([pad(x["labels"], -100) for x in batch]),
        }

def sample_replay(ds_path: str, frac: float = 0.3, seed: int = 13):
    ds = load_dataset("json", data_files={"old": ds_path})["old"]
    n = max(1, int(len(ds) * frac))
    random.seed(seed)
    idx = random.sample(range(len(ds)), n)
    return ds.select(idx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="gpt2")
    ap.add_argument("--old_adapter_dir", default="out/gpt2-lora-qa")  # from first fine-tune
    ap.add_argument("--new_data_path", default="data/finance_new.jsonl")
    ap.add_argument("--replay_from", default="data/train.jsonl")
    ap.add_argument("--replay_frac", type=float, default=0.3)
    ap.add_argument("--output_dir", default="out/gpt2-lora-finance")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--new_adapter_name", default="finance_v1")
    args = ap.parse_args()

    # Load tokenizer
    tk = AutoTokenizer.from_pretrained(args.old_adapter_dir)
    if tk.pad_token is None: tk.pad_token = tk.eos_token

    # Base + load old adapter
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.old_adapter_dir)
    model.print_trainable_parameters()  # only LoRA params trainable
    # Add a new adapter head
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"], bias="none"
    )
    model.add_adapter(args.new_adapter_name, peft_cfg)
    model.set_adapter(args.new_adapter_name)  # activate the new adapter for training

    # Datasets: new finance data + replay slice of old set
    new_ds = load_dataset("json", data_files={"new": args.new_data_path})["new"]
    rep_ds = sample_replay(args.replay_from, frac=args.replay_frac)
    mix = concatenate_datasets([new_ds, rep_ds]).shuffle(seed=42)

    tokenized = mix.map(
        tokenize_and_mask, fn_kwargs={"tokenizer": tk, "max_len": args.max_len},
        batched=True, remove_columns=mix.column_names
    )
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
    )

    trainer = Trainer(
        model=model, args=train_args, train_dataset=tokenized, data_collator=collator
    )
    trainer.train()

    # Save the multi-adapter model (contains old adapter + new adapter)
    trainer.save_model(args.output_dir)
    tk.save_pretrained(args.output_dir)
    print("Saved incremental adapter to:", args.output_dir)
    print("Adapters available:", model.peft_config.keys())
    print("Tip: at inference you can call model.set_adapter('finance_v1') or the old adapter name.")

if __name__ == "__main__":
    main()