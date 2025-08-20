import os
import re
import pickle
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

# --- Configuration ---
DOCS_DIR = "out"  # Directory containing your text files
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PARENT_CHUNK_SIZE = 400  # Tokens
CHILD_CHUNK_SIZE = 100   # Tokens
CHUNK_OVERLAP = 20

# --- Artifact file paths ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"

def simple_tokenize(text: str) -> list[str]:
    """A simple tokenizer for BM25."""
    return re.findall(r'\b\w+\b', text.lower())

def process_and_chunk_documents():
    """
    Reads documents, creates parent and child chunks, and returns them.
    """
    # Use a tokenizer to accurately count tokens
    tokenizer = AutoTokenizer.from_pretrained('gpt2') # Using gpt2 tokenizer for token counting
    nltk.download('punkt', quiet=True)

    parent_chunks = []
    child_chunks = []
    chunk_id_counter = 0

    print(f"Reading documents from '{DOCS_DIR}'...")
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Create parent chunks
        parent_tokens = tokenizer.encode(text)
        for i in range(0, len(parent_tokens), PARENT_CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_end = i + PARENT_CHUNK_SIZE
            parent_chunk_tokens = parent_tokens[i:chunk_end]
            parent_text = tokenizer.decode(parent_chunk_tokens)

            parent_id = f"{filename}-{len(parent_chunks)}"
            parent_chunks.append({
                "parent_id": parent_id,
                "text": parent_text,
                "filename": filename
            })

            # Create child chunks from the parent chunk
            child_tokens = tokenizer.encode(parent_text)
            for j in range(0, len(child_tokens), CHILD_CHUNK_SIZE - CHUNK_OVERLAP):
                child_end = j + CHILD_CHUNK_SIZE
                child_chunk_tokens = child_tokens[j:child_end]
                child_text = tokenizer.decode(child_chunk_tokens)

                child_chunks.append({
                    "chunk_id": chunk_id_counter,
                    "parent_id": parent_id,
                    "text": child_text,
                })
                chunk_id_counter += 1

    print(f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks.")
    return parent_chunks, child_chunks

def build_indices(child_chunks):
    """
    Builds and saves the FAISS (dense) and BM25 (sparse) indices.
    """
    child_chunk_texts = [chunk['text'] for chunk in child_chunks]

    # --- Build Dense Index (FAISS) ---
    print(f"Building dense index with '{EMBED_MODEL_NAME}'...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embed_model.encode(
        child_chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}'.")

    # --- Build Sparse Index (BM25) ---
    print("Building sparse index with BM25...")
    tokenized_corpus = [simple_tokenize(doc) for doc in child_chunk_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 index saved to '{BM25_INDEX_PATH}'.")


def main():
    """Main function to orchestrate the indexing process."""
    parent_chunks, child_chunks = process_and_chunk_documents()
    build_indices(child_chunks)

    # Save the chunk data for the RAG script to use
    with open(CHUNK_DATA_PATH, 'wb') as f:
        pickle.dump({"parents": parent_chunks, "children": child_chunks}, f)
    print(f"Chunk data saved to '{CHUNK_DATA_PATH}'.")
    print("\nIndexing complete!")

if __name__ == "__main__":
    main()
