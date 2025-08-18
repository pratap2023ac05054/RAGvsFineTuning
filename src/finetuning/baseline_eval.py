import json, time, math, argparse, re
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# tiny text normalization for EM/F1
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = _WS.sub(" ", s)
    s = _PUNCT.sub("", s)
    return s.strip()

def f1_score(pred, gold):
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for w in g:
        common[w] = common.get(w, 0) + 1
    hit = 0
    for w in p:
        if common.get(w, 0) > 0:
            hit += 1
            common[w] -= 1
    if hit == 0:
        return 0.0
    precision = hit / len(p)
    recall = hit / len(g)
    return 2 * precision * recall / (precision + recall)

def build_prompt(q: str) -> str:
    return (
        "### Instruction\n"
        "Answer the question concisely. If unknown, say \"I don't know\".\n\n"
        f"### Question\n{q}\n\n"
        "### Response\n"
    )

def load_test(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return Dataset.from_list(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2")  # or "gpt2-medium"
    ap.add_argument("--test_path", default="data/test.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = load_test(Path(args.test_path))
    tk = AutoTokenizer.from_pretrained(args.model_name)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=tk.eos_token_id,
        pad_token_id=tk.pad_token_id,
    )

    ems, f1s, confs, times = [], [], [], []
    for ex in ds:
        prompt = build_prompt(ex["question"])
        inputs = tk(prompt, return_tensors="pt").to(args.device)

        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                generation_config=gen_cfg,
                return_dict_in_generate=True,
                output_scores=True
            )
        dt = time.perf_counter() - start
        times.append(dt)

        # decode
        full = tk.decode(out.sequences[0], skip_special_tokens=True)
        resp = full[len(prompt):].strip()

        # confidence: mean softmax prob of chosen tokens
        # out.scores: list[T=(1,vocab)] per generated step
        # chosen token ids are out.sequences[:, -k:] after prompt length
        gen_len = len(out.scores)
        if gen_len > 0:
            gen_token_ids = out.sequences[0, -gen_len:]
            step_probs = []
            for step_idx, logits in enumerate(out.scores):
                probs = torch.softmax(logits[0], dim=-1)
                tok_id = gen_token_ids[step_idx].item()
                step_probs.append(probs[tok_id].item())
            conf = sum(step_probs) / len(step_probs)
        else:
            conf = 0.0
        confs.append(conf)

        em = 1.0 if normalize(resp) == normalize(ex["answer"]) else 0.0
        ems.append(em)
        f1s.append(f1_score(resp, ex["answer"]))

    print("\n=== Baseline (pre-finetune) ===")
    print(f"Model: {args.model_name}")
    print(f"Test size: {len(ds)}")
    print(f"Accuracy (EM): {sum(ems)/len(ems):.3f}")
    print(f"F1: {sum(f1s)/len(f1s):.3f}")
    print(f"Avg confidence: {sum(confs)/len(confs):.3f}")
    print(f"Avg latency per Q: {sum(times)/len(times):.3f}s")

if __name__ == "__main__":
    main()