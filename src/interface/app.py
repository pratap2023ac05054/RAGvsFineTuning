import os
import sys
import time
import importlib
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st


# --- detect whether we're under `streamlit run` ---
def _in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        try:
            import streamlit.runtime as rt

            return getattr(rt, "exists", lambda: False)()
        except Exception:
            return False


IN_STREAMLIT = _in_streamlit_runtime()

# --- make `src/` importable and import rag_mod from src/rag_mod.py ---
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    rag_mod = importlib.import_module("rag_mod")
except Exception as e:
    if IN_STREAMLIT:
        st.set_page_config(page_title="RAG / Fine-Tuned QA", page_icon="🔎", layout="wide")
        st.error(f"Could not import module 'rag_mod' from {SRC_DIR}:\n{e}")
        st.stop()
    else:
        print(f"[ERROR] Could not import rag_mod: {e}", file=sys.stderr)
        sys.exit(1)


# ----------------- compute setup helper -----------------
def get_compute_label() -> str:
    """
    Returns a short label of the current compute setup: GPU/CUDA, Apple MPS, Intel XPU, or CPU.
    Safe to call even if torch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA GPU"
            try:
                count = torch.cuda.device_count()
                suffix = f" x{count}" if count and count > 1 else ""
            except Exception:
                suffix = ""
            return f"GPU (CUDA) • {name}{suffix}"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "GPU (Apple MPS)"
        # Intel GPU via XPU (oneAPI)
        if hasattr(torch, "xpu") and callable(getattr(torch, "xpu", None)) and torch.xpu.is_available():
            return "GPU (Intel XPU)"
        # DirectML (Windows) if present
        if hasattr(torch.backends, "dml") and torch.backends.dml.is_available():  # type: ignore[attr-defined]
            return "GPU (DirectML)"
        return "CPU"
    except Exception:
        return "CPU"


# ----------------- model/adapter resolution helpers -----------------
def _has_full_model(path: Path) -> bool:
    return (path / "config.json").exists()


def _has_adapter(path: Path) -> bool:
    if (path / "adapter_config.json").exists():
        return True
    return any(path.glob("*adapter_model*.safetensors"))


def _scan_for_model_or_adapter(root: Path, max_depth: int = 2) -> Tuple[Optional[Path], Optional[str]]:
    """
    Search root (and subdirs up to max_depth) for:
      - full model dir (contains config.json)
      - or PEFT adapter dir (contains adapter_config.json or *adapter_model*.safetensors)
    Returns (path, kind) where kind is 'full' or 'adapter', else (None, None).
    """
    if not root.exists() or not root.is_dir():
        return None, None

    # Depth-0 checks
    if _has_full_model(root):
        return root, "full"
    if _has_adapter(root):
        return root, "adapter"

    # BFS-style scan up to max_depth
    frontier = [(root, 0)]
    seen = {root}
    best_full: Optional[Path] = None
    best_adapter: Optional[Path] = None

    while frontier:
        cur, d = frontier.pop(0)
        if d > max_depth:
            continue
        for child in cur.iterdir():
            if not child.is_dir() or child in seen:
                continue
            seen.add(child)
            if _has_full_model(child):
                best_full = best_full or child  # prefer first found at shallow depth
            elif _has_adapter(child):
                best_adapter = best_adapter or child
            if d + 1 <= max_depth:
                frontier.append((child, d + 1))

    if best_full is not None:
        return best_full, "full"
    if best_adapter is not None:
        return best_adapter, "adapter"
    return None, None


# ------------- shared helpers -------------
def answer_with_rag(query: str, use_gen: bool):
    vec, tfidf, fnames, docs = rag_mod.load_data()

    t0 = time.perf_counter()
    result = rag_mod.adaptive_retrieve_and_answer(query, vec, tfidf, fnames, docs)
    elapsed = time.perf_counter() - t0

    extractive = (result.get("best") or {}).get("answer", "") or ""
    score = (result.get("best") or {}).get("score", None)
    gen_answer = (result.get("gen") or "").strip()
    answer_text = gen_answer if (use_gen and gen_answer) else extractive

    method_str = (
        "RAG (Hybrid TF-IDF+BM25 → chunk merge → extractive QA"
        + (" + generative synthesis)" if (use_gen and gen_answer) else ")")
    )
    iters = result.get("iterations", None)
    if iters:
        method_str += f" • iterations={iters}"

    contexts = result.get("contexts", []) or []
    # For RAG, "confidence" = extractive QA score (if available)
    return answer_text, score, method_str, elapsed, contexts


# ---- Fine-tuned model answering with confidence ----
def _mean_gen_token_prob(out, sequences, gen_len, device):
    import torch

    if gen_len <= 0:
        return 0.0
    gen_ids = sequences[0, -gen_len:].to(device)
    step_probs = []
    for step_idx, logits in enumerate(out.scores):
        probs = torch.softmax(logits[0], dim=-1)
        tok_id = gen_ids[step_idx].item()
        step_probs.append(float(probs[tok_id]))
    return sum(step_probs) / len(step_probs)


def answer_with_finetuned(query: str):
    """
    Loads either:
      - a full fine-tuned model (local dir or HF Hub id), or
      - a PEFT adapter + its BASE_MODEL.
    Default root is src/finetuning, but we auto-scan inside it for a valid subfolder.
    """
    import json
    import torch
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        GenerationConfig,
    )
    from peft import PeftModel

    # Resolve model ID/path
    local_root = SRC_DIR / "finetuning"
    env_model = os.environ.get("FINETUNED_MODEL", "").strip()
    model_id = env_model or str(local_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new = int(os.environ.get("FT_MAX_NEW", "128"))
    prompt = f"You are concise.\n\nQuestion: {query}\nAnswer:"

    # Try to resolve local paths (full model or adapter), otherwise treat as HF Hub id
    kind = None
    resolved_path = None
    p = Path(model_id)

    if p.exists():
        # Use it directly if valid, else scan inside for a valid child folder
        if _has_full_model(p):
            resolved_path, kind = p, "full"
        elif _has_adapter(p):
            resolved_path, kind = p, "adapter"
        else:
            found, k = _scan_for_model_or_adapter(p, max_depth=2)
            if found is not None:
                resolved_path, kind = found, k

    # helper fns
    def _gen_and_conf_seq2seq(model, tk, inputs):
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=tk.eos_token_id,
            pad_token_id=tk.pad_token_id,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                generation_config=gen_cfg,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - t0
        text = tk.decode(out.sequences[0], skip_special_tokens=True).strip()
        conf = _mean_gen_token_prob(out, out.sequences, len(out.scores), device)
        return text, conf, elapsed

    def _gen_and_conf_causal(model, tk, inputs):
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=tk.eos_token_id,
            pad_token_id=tk.pad_token_id,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                generation_config=gen_cfg,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - t0
        full = tk.decode(out.sequences[0], skip_special_tokens=True)
        anchor = "Answer:"
        pos = full.rfind(anchor)
        text = full[pos + len(anchor):].strip() if pos >= 0 else full.strip()
        conf = _mean_gen_token_prob(out, out.sequences, len(out.scores), device)
        return text, conf, elapsed

    # CASE A: we resolved a local FULL model folder
    if resolved_path is not None and kind == "full":
        model_id = str(resolved_path)
        cfg = AutoConfig.from_pretrained(model_id)
        tk = AutoTokenizer.from_pretrained(model_id)
        if tk.pad_token is None:
            tk.pad_token = tk.eos_token or tk.cls_token

        if getattr(cfg, "is_encoder_decoder", False):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
            inputs = tk(prompt, return_tensors="pt").to(device)
            text, confidence, elapsed = _gen_and_conf_seq2seq(model, tk, inputs)
            method = "Fine-Tuned (seq2seq); no retrieval"
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            inputs = tk(prompt, return_tensors="pt").to(device)
            text, confidence, elapsed = _gen_and_conf_causal(model, tk, inputs)
            method = "Fine-Tuned (causal LM); no retrieval"

        return text, confidence, method, elapsed

    # CASE B: we resolved a local ADAPTER folder
    if resolved_path is not None and kind == "adapter":
        adapter_dir = str(resolved_path)
        base_id = os.environ.get("BASE_MODEL", "").strip()

        # Try to read base from adapter_config if not provided
        if not base_id and (resolved_path / "adapter_config.json").exists():
            try:
                base_id = json.loads((resolved_path / "adapter_config.json").read_text(encoding="utf-8")).get(
                    "base_model_name_or_path", ""
                )
            except Exception:
                base_id = ""

        if not base_id:
            base_id = "gpt2"  # default fallback; override with BASE_MODEL for other bases

        tk = AutoTokenizer.from_pretrained(base_id)
        if tk.pad_token is None:
            tk.pad_token = tk.eos_token or tk.cls_token
        base_cfg = AutoConfig.from_pretrained(base_id)

        if getattr(base_cfg, "is_encoder_decoder", False):
            base = AutoModelForSeq2SeqLM.from_pretrained(base_id).to(device)
            from peft import PeftModel  # local import to avoid hard dep in RAG-only mode
            model = PeftModel.from_pretrained(base, adapter_dir)
            inputs = tk(prompt, return_tensors="pt").to(device)
            text, confidence, elapsed = _gen_and_conf_seq2seq(model, tk, inputs)
            method = f"Fine-Tuned (PEFT adapter on {base_id}, seq2seq); no retrieval"
        else:
            base = AutoModelForCausalLM.from_pretrained(base_id).to(device)
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, adapter_dir)
            inputs = tk(prompt, return_tensors="pt").to(device)
            text, confidence, elapsed = _gen_and_conf_causal(model, tk, inputs)
            method = f"Fine-Tuned (PEFT adapter on {base_id}, causal LM); no retrieval"

        return text, confidence, method, elapsed

    # CASE C: treat model_id as a HF Hub id
    from transformers import AutoConfig as _AutoConfig  # local import for clarity
    try:
        cfg = _AutoConfig.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not load config from '{model_id}'. "
            f"If this is a LoRA/PEFT adapter, set FINETUNED_MODEL to the adapter dir "
            f"and BASE_MODEL=<base model id>, or point FINETUNED_MODEL to a valid subfolder "
            f"inside src/finetuning (e.g., a checkpoint directory containing config.json).\n\nDetails: {e}"
        )

    tk = AutoTokenizer.from_pretrained(model_id)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token or tk.cls_token

    if getattr(cfg, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        inputs = tk(prompt, return_tensors="pt").to(device)
        text, confidence, elapsed = _gen_and_conf_seq2seq(model, tk, inputs)
        method = "Fine-Tuned (seq2seq); no retrieval"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        inputs = tk(prompt, return_tensors="pt").to(device)
        text, confidence, elapsed = _gen_and_conf_causal(model, tk, inputs)
        method = "Fine-Tuned (causal LM); no retrieval"

    return text, confidence, method, elapsed


# ------------- CLI fallback -------------
def _cli_main():
    print("=== RAG / Fine-Tuned QA (CLI) ===")
    print("Tip: for full UI, run:  streamlit run src/interface/app.py\n")
    mode = (os.environ.get("MODE") or "RAG").upper()
    if mode not in ("RAG", "FINE-TUNED"):
        mode = "RAG"
    try:
        query = input("Enter your query: ").strip()
    except EOFError:
        print("[ERROR] No query provided.", file=sys.stderr)
        sys.exit(1)
    if not query:
        print("Empty query. Exiting.")
        sys.exit(0)

    compute = get_compute_label()

    if mode == "RAG":
        use_gen = os.environ.get("USE_GEN", "1") not in ("0", "false", "False")
        ans, score, method, secs, ctxs = answer_with_rag(query, use_gen)
        confidence = score
    else:
        ans, confidence, method, secs = answer_with_finetuned(query)
        ctxs = []

    print("\n--- Result ---")
    print(f"Method: {method}")
    print(f"Inference time: {secs*1000:.0f} ms")
    print(f"Confidence: {'—' if confidence is None else f'{confidence:.3f}'}")
    print(f"Compute: {compute}")
    print("\nAnswer:\n" + (ans or "[no answer]"))

    if ctxs:
        print("\n--- Retrieved Contexts ---")
        for i, c in enumerate(ctxs, 1):
            span = f"{c.get('start','?')}–{c.get('end','?')}"
            preview = (c.get("text", "") or "").replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + " …"
            print(f"[{i}] {c.get('filename','?')} [{span}]  {preview}")


# ------------- Streamlit UI -------------
def _st_main():
    st.set_page_config(page_title="RAG / Fine-Tuned QA", page_icon="🔎", layout="wide")
    st.title("🔎 RAG / Fine-Tuned QA")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["RAG", "Fine-Tuned"], index=0)
       

    query = st.text_input("Enter your query", placeholder="Ask a question…")

    # ---- custom CSS for the main "Ask" button (smaller + color #170f5f) ----
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #170f5f;
            color: white;
            padding: 10px 20px;   /* smaller height & width */
            font-size: 8px;          /* smaller text */
            border-radius: 6px;
        }
        div.stButton > button:first-child:hover {
            background-color: #2a1b8d;   /* subtle hover shade */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Smaller button (no full-width)
    ask = st.button("Ask")  # intentionally not use_container_width


    if not ask:
        return
    if not query.strip():
        st.warning("Please enter a non-empty query.")
        return

    if mode == "RAG":
        with st.spinner("Running RAG…"):
            answer, score, method, seconds, contexts = answer_with_rag(query, True)
            confidence = score
    else:
        with st.spinner("Running Fine-Tuned model…"):
            answer, confidence, method, seconds = answer_with_finetuned(query)
            contexts = []

    # Summary
    compute_label = get_compute_label()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Method", method)
    c2.metric("Confidence", "—" if confidence is None else f"{confidence:.3f}")
    c3.metric("Inference time", f"{seconds*1000:.0f} ms")
    c4.metric("Compute", compute_label)

    st.subheader("Answer")
    st.code(answer or "[no answer]", language=None)

    if mode == "RAG" and contexts:
        st.subheader("Retrieved Contexts")
        for i, ctx in enumerate(contexts, start=1):
            with st.expander(
                f"Context {i}: {ctx.get('filename','')} [{ctx.get('start','?')}–{ctx.get('end','?')}]"
            ):
                txt = ctx.get("text", "") or ""
                st.write(txt if len(txt) <= 1200 else txt[:1200] + " …")


if __name__ == "__main__":
    if _in_streamlit_runtime():
        _st_main()
    else:
        _cli_main()
