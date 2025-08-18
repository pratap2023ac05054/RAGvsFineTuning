import json, random, os
from pathlib import Path

random.seed(13)

SRC = Path("data/qa.jsonl")
OUT_DIR = Path("data")
TRAIN_OUT = OUT_DIR / "train.jsonl"
TEST_OUT  = OUT_DIR / "test.jsonl"

def main():
    assert SRC.exists(), f"Missing {SRC}. Put ~50 Q/A pairs there."
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    # light validation
    cleaned = []
    for r in rows:
        q = (r.get("question","") or "").strip()
        a = (r.get("answer","") or "").strip()
        if q and a:
            cleaned.append({"question": q, "answer": a})
    if len(cleaned) < 20:
        raise ValueError("Need at least ~20 Q/A pairs; you said ~50. Add more to qa.jsonl.")

    random.shuffle(cleaned)
    # 80/20 split
    n = len(cleaned)
    n_test = max(10, n // 5)
    test = cleaned[:n_test]
    train = cleaned[n_test:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with TEST_OUT.open("w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train and {len(test)} test examples.")

if __name__ == "__main__":
    main()