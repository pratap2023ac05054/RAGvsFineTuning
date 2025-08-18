import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer

# ==================== Tunables ====================
TOP_K_CONTEXTS = 3                 # contexts passed to QA at each attempt
DOC_CANDIDATES = 25                # #docs shortlisted in coarse phase
RETR_CHUNK_CHARS = 600             # chunk size for retrieval
RETR_STRIDE = 120                  # retrieval overlap
MERGE_GAP_CHARS = 240              # max gap to merge adjacent hits from same doc
MERGED_MAX_CHARS = 2000            # cap merged context size

QA_CHARS_PER_CHUNK = 1200          # chunk size for QA (token-limit safety)
QA_STRIDE = 200
QA_MODEL = "deepset/roberta-base-squad2"

ADAPTIVE_BASE_K = 6                # start with top-K chunks (post-fusion)
ADAPTIVE_STEP = 6                  # expand by this many if confidence low
ADAPTIVE_MAX_K = 36                # hard cap
CONF_THRESHOLD = 0.35              # stop once best QA score >= threshold

# ---- Generative synthesis (small OS model) ----
GEN_MODEL = os.environ.get("GEN_MODEL", "distilgpt2")  # e.g., "gpt2"
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "160"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "0.95"))
GEN_REPEAT_PENALTY = float(os.environ.get("GEN_REPEAT_PENALTY", "1.05"))
# =================================================


# ==================== Loaders ====================
def _joblib_load(path: Path) -> Optional[object]:
    """Try to load with joblib; return None on failure."""
    try:
        from joblib import load as joblib_load
        return joblib_load(path)
    except Exception:
        return None

def _pickle_load(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_any(path: str) -> object:
    """
    Load artifact saved via joblib OR pickle, trying joblib first (covers files
    that trigger `_pickle.UnpicklingError: invalid load key, 'x'` when using pickle).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    obj = _joblib_load(p)
    if obj is not None:
        return obj
    # Attempt pickle next
    try:
        return _pickle_load(p)
    except Exception as e_pickle:
        # As a last attempt, try joblib again (in case of transient import issues)
        obj2 = _joblib_load(p)
        if obj2 is not None:
            return obj2
        raise RuntimeError(f"Failed to load '{path}' via joblib or pickle: {e_pickle}")

def load_data():
    """
    Robustly load artifacts. If tfidf_matrix is unavailable, try to compute from
    existing vectorizer; if that fails, refit a fresh TfidfVectorizer.
    Returns: vectorizer, tfidf_matrix, filenames, documents
    """
    vectorizer = _load_any("vectorizer.pkl")
    filenames  = _load_any("filenames.pkl")
    documents  = _load_any("documents.pkl")

    # Try to load tfidf_matrix; else compute
    try:
        tfidf_matrix = _load_any("tfidf_matrix.pkl")
    except Exception:
        try:
            tfidf_matrix = vectorizer.transform(documents)
        except Exception:
            # As a last resort, refit
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)

    n_rows = getattr(tfidf_matrix, "shape", (0, 0))[0]
    if not (n_rows == len(documents) == len(filenames)):
        raise ValueError(
            f"Data misalignment: tfidf rows={n_rows}, documents={len(documents)}, filenames={len(filenames)}"
        )
    return vectorizer, tfidf_matrix, filenames, documents
# =================================================


# ==================== Utilities ====================
def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

def alpha_for_query(q_tokens: List[str]) -> float:
    """Adaptive fusion weight: shorter queries lean on BM25 a bit more."""
    return 0.35 if len(q_tokens) <= 3 else 0.5
# ====================================================


# ==================== Coarse retrieval (docs) ====================
def build_bm25_docs(vectorizer, documents: List[str]):
    analyzer = vectorizer.build_analyzer()
    tokenized_docs = [analyzer(d) for d in documents]
    return BM25Okapi(tokenized_docs), analyzer

def coarse_doc_retrieve(query: str,
                        vectorizer,
                        tfidf_matrix,
                        bm25_docs: BM25Okapi,
                        analyzer,
                        k_docs: int = DOC_CANDIDATES):
    n_docs = tfidf_matrix.shape[0]
    k = min(k_docs, n_docs)
    if k == 0:
        return np.array([], dtype=int), np.array([])

    q_vec = vectorizer.transform([query])
    cos = cosine_similarity(q_vec, tfidf_matrix).ravel()

    q_tokens = analyzer(query) if query else []
    bm25_scores = np.array(bm25_docs.get_scores(q_tokens)) if q_tokens else np.zeros(n_docs)

    cos_n = minmax_norm(cos)
    bm25_n = minmax_norm(bm25_scores)
    alpha = alpha_for_query(q_tokens)
    combined = alpha * cos_n + (1 - alpha) * bm25_n

    top_doc_idx = np.argpartition(combined, -k)[-k:]
    top_doc_idx = top_doc_idx[np.argsort(combined[top_doc_idx])[::-1]]
    return top_doc_idx, combined
# ===============================================================


# ==================== Chunking ====================
def chunk_document(text: str, max_len=RETR_CHUNK_CHARS, stride=RETR_STRIDE):
    if not text:
        return [(0, 0, "")]
    if len(text) <= max_len:
        return [(0, len(text), text)]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        out.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(0, end - stride)
    return out

def build_chunks_for_docs(documents: List[str], filenames: List[str], doc_indices: List[int]):
    chunks = []  # dicts: {doc_idx, filename, start, end, text}
    for di in doc_indices:
        text = documents[di]
        fname = filenames[di]
        for (s, e, t) in chunk_document(text):
            chunks.append({"doc_idx": di, "filename": fname, "start": s, "end": e, "text": t})
    return chunks
# =================================================


# ==================== Fine retrieval (chunks) ====================
def build_bm25_chunks(vectorizer, chunks: List[Dict]):
    analyzer = vectorizer.build_analyzer()
    tokenized_chunks = [analyzer(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_chunks), analyzer, tokenized_chunks

def score_chunks(query: str,
                 vectorizer,
                 chunks: List[Dict],
                 bm25_chunk: BM25Okapi,
                 chunk_analyzer):
    if not chunks:
        return np.array([]), np.array([]), np.array([])

    q_vec = vectorizer.transform([query])                 # (1, n_terms)
    chunk_texts = [c["text"] for c in chunks]
    chunk_tfidf = vectorizer.transform(chunk_texts)       # (n_chunks, n_terms)
    cos = cosine_similarity(q_vec, chunk_tfidf).ravel()

    q_tokens = chunk_analyzer(query) if query else []
    bm25_scores = np.array(bm25_chunk.get_scores(q_tokens)) if q_tokens else np.zeros(len(chunks))

    cos_n = minmax_norm(cos)
    bm25_n = minmax_norm(bm25_scores)
    alpha = alpha_for_query(q_tokens)
    combined = alpha * cos_n + (1 - alpha) * bm25_n
    return combined, cos, bm25_scores
# ================================================================


# ==================== Chunk merging ====================
def merge_adjacent_hits(chunks: List[Dict],
                        hit_indices: List[int],
                        gap_tol: int = MERGE_GAP_CHARS,
                        max_chars: int = MERGED_MAX_CHARS):
    """Group nearby chunk hits from the SAME doc into merged contexts."""
    by_doc: Dict[int, List[int]] = {}
    for idx in hit_indices:
        di = chunks[idx]["doc_idx"]
        by_doc.setdefault(di, []).append(idx)

    merged = []
    for di, idxs in by_doc.items():
        idxs.sort(key=lambda i: chunks[i]["start"])
        cur_start, cur_end = None, None
        cur_text_parts: List[str] = []
        cur_members: List[int] = []

        def flush_group():
            if not cur_members:
                return
            text = "".join(cur_text_parts)
            merged.append({
                "doc_idx": di,
                "filename": chunks[cur_members[0]]["filename"],
                "start": cur_start,
                "end": cur_end,
                "text": text[:max_chars],
                "members": cur_members.copy()
            })

        for i in idxs:
            s = chunks[i]["start"]
            e = chunks[i]["end"]
            t = chunks[i]["text"]

            if cur_members and (s - cur_end) > gap_tol:
                flush_group()
                cur_start, cur_end = s, e
                cur_text_parts = [t]
                cur_members = [i]
            else:
                if not cur_members:
                    cur_start, cur_end = s, e
                    cur_text_parts = [t]
                    cur_members = [i]
                else:
                    gap = s - cur_end
                    if gap > 0:
                        # small bridge to maintain continuity
                        cur_text_parts.append(chunks[cur_members[-1]]["text"][-min(gap, 50):])
                    cur_text_parts.append(t)
                    cur_end = max(cur_end, e)
                    cur_members.append(i)

            # enforce size limit early
            if sum(len(p) for p in cur_text_parts) >= max_chars:
                cur_text_parts = ["".join(cur_text_parts)[:max_chars]]
                cur_end = e
                flush_group()
                cur_start, cur_end, cur_text_parts, cur_members = None, None, [], []

        flush_group()

    return merged
# ======================================================


# ==================== QA (extractive) ====================
def chunk_for_qa(text: str, max_len=QA_CHARS_PER_CHUNK, stride=QA_STRIDE) -> List[str]:
    if len(text) <= max_len:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        out.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - stride)
    return out

def _make_qa_pipeline():
    use_gpu = os.environ.get("USE_GPU", "").strip() == "1"
    device = 0 if use_gpu else -1
    return pipeline("question-answering", model=QA_MODEL, device=device)

def answer_over_contexts(query: str, contexts: List[Dict], qa):
    """Run QA over each merged context (with QA-safe subchunking); return best."""
    best = {"answer": "", "score": 0.0, "context_idx": -1}
    per_context_best = []
    for ci, ctx in enumerate(contexts):
        local_best = {"answer": "", "score": 0.0, "context_idx": ci}
        for sub in chunk_for_qa(ctx["text"]):
            try:
                out = qa(question=query, context=sub)
                if out["score"] > local_best["score"]:
                    local_best = {"answer": out["answer"], "score": float(out["score"]), "context_idx": ci}
            except Exception:
                continue
        per_context_best.append(local_best)
        if local_best["score"] > best["score"]:
            best = local_best
    return best, per_context_best
# =======================================================


# ==================== Generative synthesis ====================
def _make_gen_pipeline():
    """Create a small open-source text-generation pipeline and tokenizer."""
    use_gpu = os.environ.get("USE_GPU", "").strip() == "1"
    device = 0 if use_gpu else -1
    gen = pipeline("text-generation", model=GEN_MODEL, device=device)
    tok = gen.tokenizer if getattr(gen, "tokenizer", None) is not None else AutoTokenizer.from_pretrained(GEN_MODEL)
    return gen, tok

def _truncate_to_budget(texts: List[str], tokenizer: AutoTokenizer, max_input_tokens: int) -> str:
    """
    Greedily add texts until the token budget is hit. Preserves order.
    """
    kept = []
    used = 0
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        n = len(ids)
        if used + n <= max_input_tokens:
            kept.append(t)
            used += n
        else:
            remaining = max_input_tokens - used
            if remaining > 0:
                kept.append(tokenizer.decode(ids[:remaining]))
            break
    return "\n".join(kept)

def build_generation_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Instruction-style prompt to constrain small causal LMs (e.g., GPT-2 family)
    to only use provided passages.
    """
    header = (
        "You are a concise, factual assistant.\n"
        "Use ONLY the passages to answer the question.\n"
        "If the answer is not in the passages, say you don't know.\n"
    )
    parts = [header, "Passages:"]
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"[{i}] From {ctx['filename']} ({ctx['start']}–{ctx['end']}):\n{ctx['text']}\n")
    parts.append(f"Question: {query}\nAnswer:")
    return "\n".join(parts)

def generate_answer_from_contexts(query: str, contexts: List[Dict]) -> str:
    """
    Concatenate retrieved passages + query, keep within context window,
    and produce a generative answer.
    """
    if not contexts:
        return ""

    try:
        gen, tok = _make_gen_pipeline()
    except Exception:
        return ""

    model_ctx = getattr(tok, "model_max_length", 1024)
    # Reserve room for generation (roughly)
    input_budget = max(128, model_ctx - int(GEN_MAX_NEW_TOKENS * 1.2))

    prompt_full = build_generation_prompt(query, contexts)
    prompt = _truncate_to_budget([prompt_full], tok, input_budget)

    try:
        out = gen(
            prompt,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            repetition_penalty=GEN_REPEAT_PENALTY,
            eos_token_id=tok.eos_token_id
        )
        text = out[0]["generated_text"]
        # Return only the text after the last "Answer:"
        anchor = "Answer:"
        pos = text.rfind(anchor)
        return text[pos + len(anchor):].strip() if pos >= 0 else text.strip()
    except Exception:
        return ""
# ============================================================


# ==================== Adaptive loop ====================
def adaptive_retrieve_and_answer(query: str,
                                 vectorizer,
                                 tfidf_matrix,
                                 filenames: List[str],
                                 documents: List[str]):
    # Coarse: shortlist docs
    bm25_docs, analyzer = build_bm25_docs(vectorizer, documents)
    top_doc_idx, _doc_scores = coarse_doc_retrieve(
        query, vectorizer, tfidf_matrix, bm25_docs, analyzer, k_docs=DOC_CANDIDATES
    )
    if top_doc_idx.size == 0:
        return {"best": {"answer": "", "score": 0.0}, "contexts": [], "iterations": 0, "gen": ""}

    # Build chunks for shortlisted docs
    chunks = build_chunks_for_docs(documents, filenames, top_doc_idx.tolist())
    if not chunks:
        return {"best": {"answer": "", "score": 0.0}, "contexts": [], "iterations": 0, "gen": ""}

    bm25_chunk, chunk_analyzer, _ = build_bm25_chunks(vectorizer, chunks)
    combined, _cos, _bm25 = score_chunks(query, vectorizer, chunks, bm25_chunk, chunk_analyzer)
    if combined.size == 0:
        return {"best": {"answer": "", "score": 0.0}, "contexts": [], "iterations": 0, "gen": ""}

    # Prepare QA
    try:
        qa = _make_qa_pipeline()
    except Exception:
        qa = None

    iterations = 0
    k = min(ADAPTIVE_BASE_K, combined.size)
    best_overall = {"answer": "", "score": 0.0, "context_idx": -1}
    chosen_contexts_snapshot: List[Dict] = []

    while True:
        iterations += 1
        # pick top-k chunk hits
        top_k_idx = np.argpartition(combined, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(combined[top_k_idx])[::-1]]

        # merge adjacent chunk hits into contexts
        merged_contexts = merge_adjacent_hits(
            chunks, top_k_idx.tolist(), gap_tol=MERGE_GAP_CHARS, max_chars=MERGED_MAX_CHARS
        )

        # rank merged contexts by sum of member scores
        def group_score(ctx):
            return float(np.sum([combined[i] for i in ctx["members"]]))
        merged_contexts.sort(key=group_score, reverse=True)

        # pick top contexts
        contexts = merged_contexts[:TOP_K_CONTEXTS]
        chosen_contexts_snapshot = contexts

        # QA
        if qa is None or not contexts:
            # retrieval-only fallback
            break

        best, _per_context = answer_over_contexts(query, contexts, qa)

        # stop if confident enough or budget exhausted
        if best["score"] >= CONF_THRESHOLD or k >= min(ADAPTIVE_MAX_K, combined.size):
            best_overall = best
            break

        best_overall = best if best["score"] > best_overall["score"] else best_overall
        k = min(k + ADAPTIVE_STEP, combined.size)

    # Generative synthesis on chosen contexts
    gen_answer = generate_answer_from_contexts(query, chosen_contexts_snapshot)

    return {
        "best": best_overall,
        "contexts": chosen_contexts_snapshot,
        "iterations": iterations,
        "gen": gen_answer
    }
# ======================================================


# ==================== CLI ====================
def main():
    try:
        vectorizer, tfidf_matrix, filenames, documents = load_data()
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        query = input("Enter your query: ").strip()
    except EOFError:
        print("[ERROR] No query provided (EOF).", file=sys.stderr)
        sys.exit(1)

    if not query:
        print("Please enter a non-empty query.")
        return

    result = adaptive_retrieve_and_answer(query, vectorizer, tfidf_matrix, filenames, documents)

    best = result["best"]
    contexts = result["contexts"]
    print(f"\n[Adaptive Retrieval] Iterations: {result['iterations']}, "
          f"Best QA score: {best.get('score', 0.0):.4f}")
    print(f"Extractive Answer: {best.get('answer','')!r}")

    if result.get("gen"):
        print("\nGenerative Answer:")
        print(result["gen"])

    for i, ctx in enumerate(contexts, start=1):
        preview = ctx["text"][:200].replace("\n", " ") + ("..." if len(ctx["text"]) > 200 else "")
        span = f"{ctx['start']}–{ctx['end']}"
        print(f"\nContext {i}: {ctx['filename']} [{span}]")
        print(f"Preview: {preview}")

if __name__ == "__main__":
    main()