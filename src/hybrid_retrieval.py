import sys
import os


# --- MODIFICATION END ---

import argparse
import pickle
import string
import time
import json
import subprocess

# --- MODIFICATION START ---
# Add the local 'packages' folder to the Python path.
# This forces the script to look for imports in this directory first.
# NOTE: Assumes you have a folder named 'packages' in the same directory as this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'package')))
# The faiss-cpu library has a hard dependency on NumPy.
# We import it only where needed to interact with the faiss index.
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

import nltk
from response_generator import ResponseGenerator

# --- NLTK Data Download Logic ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data packages...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Configuration ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_N = 10
RRF_K = 60

# --- Artifact file paths ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"

# --- Pre-computation ---
STOPWORDS = set(stopwords.words('english'))

def preprocess_query(query: str) -> tuple[str, list[str]]:
    """Cleans, lowercases, and removes stopwords from a query."""
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(query)
    filtered_tokens = [word for word in tokens if word not in STOPWORDS and word.isalnum()]
    return " ".join(filtered_tokens), filtered_tokens

def reciprocal_rank_fusion(retrieved_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Combines multiple ranked lists using Reciprocal Rank Fusion."""
    fused_scores = {}
    for doc_list in retrieved_lists:
        for rank, doc_id in enumerate(doc_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
    return fused_scores

def retrieve(query: str, embed_model, faiss_index, bm25_index, chunk_data):
    """Performs the full hybrid retrieval pipeline."""
    preprocessed_query, bm25_tokens = preprocess_query(query)
    print(f"Preprocessed Query: '{preprocessed_query}'")
    
    # NumPy is required by the faiss-cpu library to correctly format the input vector.
    import numpy as np
    query_embedding = embed_model.encode([preprocessed_query]).astype(np.float32)

    _, dense_indices = faiss_index.search(query_embedding, k=TOP_N)
    dense_retrieved_ids = dense_indices[0].tolist()

    bm25_scores = bm25_index.get_scores(bm25_tokens)
    
    # REPLACEMENT FOR np.argsort: Use Python's sorted() with enumerate
    # to get the indices of the top N scores in descending order.
    sparse_retrieved_ids = [
        item[0] for item in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    ][:TOP_N]

    print(f"Dense Retriever found IDs: {dense_retrieved_ids}")
    print(f"Sparse Retriever found IDs: {sparse_retrieved_ids}")

    fused_scores = reciprocal_rank_fusion([dense_retrieved_ids, sparse_retrieved_ids])
    sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

    child_chunks = chunk_data["children"]
    final_results = [{"chunk_id": cid, **child_chunks[cid], "score": fused_scores[cid]} for cid in sorted_chunk_ids]
    return final_results

def main():
    """Main function to load indices and run the full RAG pipeline for the UI."""
    parser = argparse.ArgumentParser(description="Full RAG Pipeline")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    parser.add_argument("--mode", type=str, choices=["RAG", "Fine-Tuned"], required=True, help="The mode of operation.")
    args = parser.parse_args()

    start_time = time.time()

    # Load retrieval components
    print("Loading retrieval indices and data...")
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(BM25_INDEX_PATH, 'rb') as f:
            bm25_index = pickle.load(f)
        with open(CHUNK_DATA_PATH, 'rb') as f:
            chunk_data = pickle.load(f)
    except FileNotFoundError:
        print(json.dumps({"error": "Index files not found. Please run 'build_indices.py' first."}))
        return

    # Initialize the Response Generator
    generator = ResponseGenerator()

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(args.query, embed_model, faiss_index, bm25_index, chunk_data)

    # Generate the final answer
    print("Generating final answer...")
    final_answer = generator.generate(args.query, retrieved_chunks)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    # Use the highest score from the retrieved chunks as the confidence score
    confidence_score = 0
    if retrieved_chunks:
        confidence_score = retrieved_chunks[0].get("score", 0)


    # Prepare and Print Results as JSON for the Streamlit app
    results = {
        "answer": final_answer,
        "confidence_score": confidence_score,
        "method_used": args.mode,
        "response_time": response_time
    }
    print(json.dumps(results))

if __name__ == "__main__":
    main()