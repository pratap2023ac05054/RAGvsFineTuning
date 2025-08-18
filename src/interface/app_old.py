import os
import sys
import time
import importlib
from pathlib import Path

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
SRC_DIR = THIS_FILE.parents[1]            # .../src
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
    return answer_text, score, method_str, elapsed, contexts

def answer_with_finetuned(query: str):
    from transformers import pipeline
    model_id = os.environ.get("FINETUNED_MODEL", "").strip()
    if not model_id:
        raise RuntimeError("FINETUNED_MODEL is not set (expected a HF model id or local path).")

    # prefer text2text first
    try:
        task = "text2text-generation"
        pipe = pipeline(task, model=model_id)
        prompt = f"You are concise.\n\nQuestion: {query}\nAnswer:"
        t0 = time.perf_counter()
        out = pipe(prompt, max_new_tokens=256)
        text = out[0]["generated_text"]
        elapsed = time.perf_counter() - t0
    except Exception:
        # fallback to causal LM
        task = "text-generation"
        pipe = pipeline(task, model=model_id)
        prompt = f"You are concise.\n\nQuestion: {query}\nAnswer:"
        t0 = time.perf_counter()
        out = pipe(prompt, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.7)
        text = out[0]["generated_text"]
        anchor = "Answer:"
        pos = text.rfind(anchor)
        text = text[pos + len(anchor):].strip() if pos >= 0 else text.strip()
        elapsed = time.perf_counter() - t0

    return text, None, f"Fine-Tuned ({task}); no retrieval", elapsed

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

    if mode == "RAG":
        use_gen = os.environ.get("USE_GEN", "1") not in ("0", "false", "False")
        ans, score, method, secs, ctxs = answer_with_rag(query, use_gen)
    else:
        ans, score, method, secs = answer_with_finetuned(query)
        ctxs = []

    print("\n--- Result ---")
    print(f"Method: {method}")
    print(f"Response time: {secs*1000:.0f} ms")
    print(f"Retrieval confidence: {'—' if score is None else f'{score:.3f}'}")
    print("\nAnswer:\n" + (ans or "[no answer]"))

    if ctxs:
        print("\n--- Retrieved Contexts ---")
        for i, c in enumerate(ctxs, 1):
            span = f"{c.get('start','?')}–{c.get('end','?')}"
            preview = (c.get("text","") or "").replace("\n", " ")
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
        use_gen_in_rag = st.checkbox("Use generative synthesis (RAG)", value=True)
        st.caption("Env:")
        st.code(
            f"USE_GPU={os.environ.get('USE_GPU','0')}\n"
            f"GEN_MODEL={os.environ.get('GEN_MODEL','distilgpt2')}\n"
            f"FINETUNED_MODEL={os.environ.get('FINETUNED_MODEL','<unset>')}",
            language="bash",
        )

    query = st.text_input("Enter your query", placeholder="Ask a question…")
    ask = st.button("Ask", type="primary", use_container_width=True)
    st.caption("Tip: In RAG mode, artifacts should be next to src/rag_mod.py.")

    if not ask:
        return
    if not query.strip():
        st.warning("Please enter a non-empty query.")
        return

    if mode == "RAG":
        with st.spinner("Running RAG…"):
            answer, score, method, seconds, contexts = answer_with_rag(query, use_gen_in_rag)
    else:
        with st.spinner("Running Fine-Tuned model…"):
            answer, score, method, seconds = answer_with_finetuned(query)
            contexts = []

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Method", method)
    c2.metric("Retrieval confidence", "—" if score is None else f"{score:.3f}")
    c3.metric("Response time", f"{seconds*1000:.0f} ms")

    st.subheader("Answer")
    st.code(answer or "[no answer]", language=None)

    if mode == "RAG" and contexts:
        st.subheader("Retrieved Contexts")
        for i, ctx in enumerate(contexts, start=1):
            with st.expander(f"Context {i}: {ctx.get('filename','')} [{ctx.get('start','?')}–{ctx.get('end','?')}]"):
                txt = ctx.get("text","") or ""
                st.write(txt if len(txt) <= 1200 else txt[:1200] + " …")

if __name__ == "__main__":
    if _in_streamlit_runtime():
        _st_main()
    else:
        _cli_main()