# hybrid_retrieval.py

import argparse
import pickle
import re
import string
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk

# Import the new components
from response_generator import ResponseGenerator
from guardrails import validate_query, validate_response

# --- NLTK Data Download ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data packages...")
    nltk.download('punkt', quiet=False)
    nltk.download('stopwords', quiet=False)

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
    print(f"Preprocessed Query: '{preprocessed_query}'\n")

    query_embedding = embed_model.encode([preprocessed_query]).astype(np.float32)

    _, dense_indices = faiss_index.search(query_embedding, k=TOP_N)
    dense_retrieved_ids = dense_indices[0].tolist()

    bm25_scores = bm25_index.get_scores(bm25_tokens)
    sparse_retrieved_ids = np.argsort(bm25_scores)[::-1][:TOP_N].tolist()

    print(f"Dense Retriever found IDs: {dense_retrieved_ids}")
    print(f"Sparse Retriever found IDs: {sparse_retrieved_ids}")

    fused_scores = reciprocal_rank_fusion([dense_retrieved_ids, sparse_retrieved_ids])
    sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

    child_chunks = chunk_data["children"]
    final_results = [{"chunk_id": cid, **child_chunks[cid], "score": fused_scores[cid]} for cid in sorted_chunk_ids]
    return final_results

def main():
    """Main function to load indices and run the full RAG pipeline."""
    parser = argparse.ArgumentParser(description="Full RAG Pipeline with Guardrails")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    args = parser.parse_args()

    # --- Step 1: Input Guardrail ---
    is_valid, message = validate_query(args.query)
    if not is_valid:
        print(f"Input Error: {message}")
        return

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
        print("Error: Index files not found. Please run 'build_indices.py' first.")
        return

    # Initialize the Response Generator
    generator = ResponseGenerator()

    # --- Step 2: Retrieve relevant chunks ---
    retrieved_chunks = retrieve(args.query, embed_model, faiss_index, bm25_index, chunk_data)

    # --- Step 3: Generate the final answer ---
    print("Generating final answer...")
    final_answer = generator.generate(args.query, retrieved_chunks)

    # --- Step 4: Output Guardrail ---
    context_texts = [chunk['text'] for chunk in retrieved_chunks[:5]] # Use top 5 for validation
    validation_result = validate_response(final_answer, context_texts)

    print("\n" + "="*50)
    print("                 FINAL GENERATED ANSWER")
    print("="*50 + "\n")
    print(final_answer)
    print("\n" + "="*50)
    print("                 GUARDRAIL VALIDATION")
    print("="*50 + "\n")
    if validation_result["pass"]:
        print("✅ Status: PASSED")
    else:
        print("⚠️ Status: FLAGGED")
        print(f"   Reason: {validation_result['reason']}")
        print(f"   Details: {validation_result.get('details', 'N/A')}")


if __name__ == "__main__":
    main()
