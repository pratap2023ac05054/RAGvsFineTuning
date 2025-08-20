# app.py

import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from peft import PeftModel

# Import the components from your other files
from response_generator import ResponseGenerator
from guardrails import validate_query, validate_response
from hybrid_retrieval import retrieve 

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RAG_GENERATOR_MODEL = "HuggingFaceH4/zephyr-7b-beta"
FINETUNED_BASE_MODEL = "distilgpt2" # The base model used for fine-tuning
FINETUNED_ADAPTER_DIR = "./distilgpt2-finetuned-fast" # The directory of your fine-tuned adapters

# --- Caching ---
# Use Streamlit's caching to load models and data only once.
@st.cache_resource
def load_components():
    """
    Loads all the necessary models and data for the RAG and Fine-Tuned pipelines.
    Returns a dictionary of components.
    """
    print("Loading components...")
    components = {}
    try:
        # Load RAG components
        components["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
        components["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)
        with open(BM25_INDEX_PATH, 'rb') as f:
            components["bm25_index"] = pickle.load(f)
        with open(CHUNK_DATA_PATH, 'rb') as f:
            components["chunk_data"] = pickle.load(f)
        components["rag_generator"] = ResponseGenerator(model_name=RAG_GENERATOR_MODEL)
        
        # Load Fine-Tuned Model components
        if os.path.exists(FINETUNED_ADAPTER_DIR):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            base_model = AutoModelForCausalLM.from_pretrained(FINETUNED_BASE_MODEL).to(device)
            
            tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_DIR)
            tuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_ADAPTER_DIR)
            
            components["finetuned_model"] = tuned_model
            components["finetuned_tokenizer"] = tuned_tokenizer
            print("Fine-tuned model loaded successfully.")
        else:
            print(f"Warning: Fine-tuned model directory not found at '{FINETUNED_ADAPTER_DIR}'.")
            components["finetuned_model"] = None

        print("All components loaded.")
        return components
        
    except FileNotFoundError as e:
        st.error(f"Error loading components: {e}. Please make sure 'build_indices.py' has been run successfully.")
        return None

def generate_from_finetuned(model, tokenizer, query):
    """Generates a response directly from the fine-tuned model."""
    start_time = time.time()
    device = next(model.parameters()).device
    
    prompt = f"Question: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response_text.replace(prompt, "").strip().split('\n')[0]
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    return answer, inference_time

# --- Main App UI ---

st.set_page_config(page_title="Advanced Q&A System", layout="wide")

st.title("Advanced Question-Answering System �")
st.markdown("This interface allows you to ask questions using either a RAG pipeline or a fine-tuned model.")

# Load all components
components = load_components()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    mode = st.radio(
        "Choose the operational mode:",
        ("RAG (Retrieval-Augmented Generation)", "Fine-Tuned Model")
    )
    st.markdown("---")
    st.info("The RAG mode finds relevant documents first and then generates an answer. The Fine-Tuned mode uses a specialized model directly.")

# Main panel for interaction
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
                        final_answer = components["rag_generator"].generate(query, retrieved_chunks)
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
                            
                elif mode == "Fine-Tuned Model":
                    if components.get("finetuned_model") is None:
                        st.error("Fine-tuned model is not available. Please run the fine-tuning script first.")
                    else:
                        with st.spinner("Querying the fine-tuned model..."):
                            answer, response_time = generate_from_finetuned(
                                components["finetuned_model"],
                                components["finetuned_tokenizer"],
                                query
                            )
                            
                            st.subheader("Generated Answer")
                            st.markdown(answer)
                            st.markdown("---")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(label="Confidence Score", value="N/A")
                            with col2:
                                st.metric(label="Method", value="Fine-Tuned Model")
                            with col3:
                                st.metric(label="Inference Time", value=f"{response_time:.2f} s")
else:
    st.error("Application components could not be loaded. Please check the console for errors.")