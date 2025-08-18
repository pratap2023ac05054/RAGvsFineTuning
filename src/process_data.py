import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional: use joblib for better compression of large sklearn objects
try:
    from joblib import dump as joblib_dump
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


def preprocess_text(text: str) -> str:
    """
    Light normalization: strip, collapse whitespace.
    (Avoid heavy tokenization here; let TfidfVectorizer handle it.)
    """
    if not text:
        return ""
    # Replace any newlines/tabs with spaces and collapse multiple spaces
    cleaned = " ".join(text.replace("\t", " ").replace("\n", " ").split())
    return cleaned


def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """
    Word-count based chunking. Skips empty input.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def collect_chunks(input_dir: Path, chunk_size: int = 500) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Returns:
        chunks: list[str] of chunk texts
        meta: list[(filename, chunk_index)]
    """
    chunks: List[str] = []
    meta: List[Tuple[str, int]] = []

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".txt"):
            continue
        file_path = input_dir / filename
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception as e:
            # Skip unreadable files but continue pipeline
            print(f"[WARN] Failed to read {file_path}: {e}")
            continue

        pre = preprocess_text(raw)
        file_chunks = split_into_chunks(pre, chunk_size=chunk_size)
        # Filter out empty / near-empty chunks
        file_chunks = [c for c in file_chunks if c.strip()]

        start_idx = len(chunks)
        chunks.extend(file_chunks)
        for i in range(len(file_chunks)):
            meta.append((filename, i))  # (source filename, chunk index)

    return chunks, meta


def save_artifacts(vectorizer, X, meta, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep your original pickle filenames
    vec_path = out_dir / "vectorizer.pkl"
    mat_path = out_dir / "tfidf_matrix.pkl"
    filenames_path = out_dir / "filenames.pkl"
    documents_path = out_dir / "documents.pkl"
    meta_json = out_dir / "chunks_meta.json"

    # Save vectorizer (joblib if available for better compatibility/compression)
    if _HAS_JOBLIB:
        joblib_dump(vectorizer, vec_path, compress=3)
    else:
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)

    # Save TF-IDF matrix
    if _HAS_JOBLIB:
        joblib_dump(X, mat_path, compress=3)
    else:
        with open(mat_path, "wb") as f:
            pickle.dump(X, f)

    # Back-compat: filenames + documents like your original (but not duplicated unnecessarily)
    # filenames.pkl: list of filenames aligned to each chunk
    filenames = [fn for (fn, _idx) in meta]
    with open(filenames_path, "wb") as f:
        pickle.dump(filenames, f)

    # documents.pkl: the chunk texts
    # NOTE: if you don’t need this twice, you can delete this file entirely.
    # Keeping it for drop-in compatibility.
    # We won't have 'documents' in scope here; better pass chunks in if needed.
    # For compatibility, we re-create documents from X’s row count and separate storage.
    # Easiest: accept chunks as param (see small tweak below).
    pass  # <-- this is addressed in the variant below

    # Also save richer metadata as JSON
    # Example: [{"filename": "fileA.txt", "chunk_index": 0}, ...]
    meta_records = [{"filename": fn, "chunk_index": idx} for (fn, idx) in meta]
    meta_json.write_text(json.dumps(meta_records, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    input_dir = Path("data/processedfiles")
    out_dir = Path(".")  # current folder

    # 1) Collect chunks + metadata
    chunks, meta = collect_chunks(input_dir, chunk_size=500)

    if not chunks:
        raise ValueError(
            f"No text chunks found under {input_dir}. "
            "Ensure there are .txt files with content."
        )

    # 2) Fit TF-IDF
    vectorizer = TfidfVectorizer(
        # keep defaults mostly; add two small niceties
        lowercase=True,
        strip_accents="unicode",
        # If your corpus is noisy, consider min_df=2 or max_df=0.9
        # min_df=2,
        # max_df=0.95,
    )
    X = vectorizer.fit_transform(chunks)

    # 3) Persist artifacts (slight tweak to also store documents)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _HAS_JOBLIB:
        joblib_dump(vectorizer, out_dir / "vectorizer.pkl", compress=3)
        joblib_dump(X, out_dir / "tfidf_matrix.pkl", compress=3)
    else:
        with open(out_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open(out_dir / "tfidf_matrix.pkl", "wb") as f:
            pickle.dump(X, f)

    # filenames aligned to chunk rows
    filenames = [fn for (fn, _idx) in meta]
    with open(out_dir / "filenames.pkl", "wb") as f:
        pickle.dump(filenames, f)

    # documents aligned to chunk rows (your original behavior)
    with open(out_dir / "documents.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # richer metadata, useful later
    meta_json = out_dir / "chunks_meta.json"
    meta_records = [{"filename": fn, "chunk_index": idx} for (fn, idx) in meta]
    meta_json.write_text(json.dumps(meta_records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
