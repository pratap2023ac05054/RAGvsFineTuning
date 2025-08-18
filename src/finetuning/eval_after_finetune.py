import argparse, json, time, re
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

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
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common = {}
    for w in g: common[w] = common.get(w,0)+1
    hit=0
    for w in p:
        if common.get(w,0)>0:
            hit+=1; common[w]-=1
    if hit==0: return 0.0
    prec=hit/len(p); rec=hit/len(g)
    return 2*prec*rec/(prec+rec)

def build_prompt(q: str) -> str:
    return (
        "### Instruction\n"
        "Answer the question concisely. If unknown, say \"I don't know\".\n\n"
        f"### Question\n{q}\n\n"
        "### Response\n"
    )

def load_test(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="gpt2")
    ap.add_argument("--adapter_dir", default="out/gpt2-lora-qa")
    ap.add_argument("--test_path", default="data/test.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tk.pad_token is None: tk.pad_token = tk.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model).to(args.device)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=False,
        eos_token_id=tk.eos_token_id, pad_token_id=tk.pad_token_id
    )

    data = load_test(Path(args.test_path))
    ems=f1s=confs=times=0.0
    for ex in data:
        prompt = build_prompt(ex["question"])
        inputs = tk(prompt, return_tensors="pt").to(args.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg,
                                 return_dict_in_generate=True, output_scores=True)
        dt = time.perf_counter()-t0
        times += dt

        full = tk.decode(out.sequences[0], skip_special_tokens=True)
        resp = full[len(prompt):].strip()

        # confidence
        gen_len = len(out.scores)
        if gen_len>0:
            gen_ids = out.sequences[0, -gen_len:]
            step_probs=[]
            for step, logits in enumerate(out.scores):
                p = torch.softmax(logits[0], dim=-1)
                step_probs.append(p[gen_ids[step]].item())
            conf = sum(step_probs)/len(step_probs)
        else:
            conf=0.0
        confs += conf

        ems += 1.0 if normalize(resp)==normalize(ex["answer"]) else 0.0
        f1s += f1_score(resp, ex["answer"])

    n=len(data)
    print("\n=== After Fine-Tuning ===")
    print(f"Adapter: {args.adapter_dir}")
    print(f"Test size: {n}")
    print(f"Accuracy (EM): {ems/n:.3f}")
    print(f"F1: {f1s/n:.3f}")
    print(f"Avg confidence: {confs/n:.3f}")
    print(f"Avg latency per Q: {times/n:.3f}s")

if __name__ == "__main__":
    main()