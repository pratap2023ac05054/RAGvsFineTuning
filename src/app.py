# app.py

import streamlit as st
import time
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

from response_generator import ResponseGenerator
from guardrails import validate_query
from hybrid_retrieval import retrieve

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# --- Caching ---
@st.cache_resource
def load_components():
    """Loads all necessary models and data."""
    print("Loading components...")
    components = {}
    try:
        components["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
        components["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)
        with open(BM25_INDEX_PATH, 'rb') as f:
            components["bm25_index"] = pickle.load(f)
        with open(CHUNK_DATA_PATH, 'rb') as f:
            components["chunk_data"] = pickle.load(f)
        components["generator"] = ResponseGenerator(model_name=GENERATOR_MODEL)
        print("All components loaded.")
        return components
    except FileNotFoundError as e:
        st.error(f"Error loading components: {e}. Please run 'build_indices.py' first.")
        return None

# --- CORRECTED FUNCTION ---
def generate_directly(generator: ResponseGenerator, query: str):
    """
    Generates a response directly from the ctransformers model.
    This function now correctly passes a string prompt to the model.
    """
    start_time = time.time()
    model = generator.model
    # The prompt format should match the model's fine-tuning
    prompt = f"<s>[INST] {query} [/INST]"

    # ctransformers models are called directly with the prompt string.
    # It handles tokenization and decoding internally.
    answer = model(
        prompt,
        max_new_tokens=250,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        stop=["</s>"] # Use the end-of-sequence token to stop generation
    )

    end_time = time.time()
    inference_time = end_time - start_time
    return answer.strip(), inference_time

# --- Main App UI ---
st.set_page_config(page_title="Advanced Q&A System", layout="wide")
st.title("Advanced Question-Answering System 🤖")
st.markdown("This interface allows you to ask questions using either a RAG pipeline or by querying the model directly.")
components = load_components()

with st.sidebar:
    st.header("Configuration")
    mode = st.radio(
        "Choose the operational mode:",
        ("RAG (Retrieval-Augmented Generation)", "Direct Generation (Mistral 7B)")
    )
    st.markdown("---")
    st.info(
        "**RAG mode** finds relevant documents first and generates an answer based on them.\n\n"
        "**Direct Generation** uses the powerful Mistral-7B model to answer directly from its own knowledge."
    )

if components:
    query = st.text_input("Enter your question here:", key="query_input")
    if st.button("Ask Question", key="ask_button"):
        if not query:
            st.warning("Please enter a question.")
        else:
            is_valid, message = validate_query(query)
            if not is_valid:
                st.error(f"Input Error: {message}")
            else:
                if mode == "RAG (Retrieval-Augmented Generation)":
                    with st.spinner("Processing your query with the RAG pipeline..."):
                        start_time = time.time()
                        retrieved_chunks = retrieve(
                            query,
                            components["embed_model"],
                            components["faiss_index"],
                            components["bm25_index"],
                            components["chunk_data"]
                        )
                        final_answer = components["generator"].generate(query, retrieved_chunks)
                        end_time = time.time()
                        response_time = end_time - start_time

                        st.subheader("Generated Answer")
                        st.markdown(final_answer)
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Retrieval Confidence", value=f"{retrieved_chunks[0]['score']:.4f}" if retrieved_chunks else "N/A")
                        with col2:
                            st.metric(label="Method", value="RAG (Hybrid)")
                        with col3:
                            st.metric(label="Response Time", value=f"{response_time:.2f} s")

                elif mode == "Direct Generation (Mistral 7B)":
                    with st.spinner("Querying the Mistral-7B model directly..."):
                        answer, response_time = generate_directly(
                            components["generator"],
                            query
                        )
                        st.subheader("Generated Answer")
                        st.markdown(answer)
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Confidence Score", value="N/A (Direct)")
                        with col2:
                            st.metric(label="Method", value="Direct Generation (Mistral 7B)")
                        with col3:
                            st.metric(label="Inference Time", value=f"{response_time:.2f} s")
else:
    st.error("Application components could not be loaded. Please check the console for errors.")