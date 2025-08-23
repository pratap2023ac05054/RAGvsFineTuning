# app.py

import streamlit as st
import time
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from response_generator import ResponseGenerator
from guardrails import validate_query
from hybrid_retrieval import retrieve

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BASE_GENERATOR_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FINETUNED_ADAPTER_PATH = "./tinyllama-finetuned-adapter-cpu" # Path to your fine-tuned adapter

# --- Caching ---
@st.cache_resource
def load_components():
    """Loads all necessary models and data."""
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
        
        # Load Base GGUF Model for RAG
        components["base_generator"] = ResponseGenerator(model_name=BASE_GENERATOR_MODEL)

        # Load Fine-Tuned Model
        if os.path.exists(FINETUNED_ADAPTER_PATH):
            print(f"Loading fine-tuned model from {FINETUNED_ADAPTER_PATH}...")
            base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)
            tuned_model = tuned_model.merge_and_unload() # Merge adapter for faster inference
            
            components["tuned_model"] = tuned_model
            components["tuned_tokenizer"] = AutoTokenizer.from_pretrained(FINETUNED_ADAPTER_PATH)
            print("Fine-tuned model loaded.")
        else:
            st.warning(f"Fine-tuned model adapter not found at '{FINETUNED_ADAPTER_PATH}'. The fine-tuned option will be disabled.")
            components["tuned_model"] = None

        print("All components loaded.")
        return components
    except FileNotFoundError as e:
        st.error(f"Error loading components: {e}. Please run 'build_indices.py' first.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading components: {e}")
        return None

def generate_from_finetuned(model, tokenizer, query: str):
    """Generates a response and calculates a confidence score."""
    start_time = time.time()
    device = "cpu"
    model.to(device)
    
    prompt = (
        f"<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{query}</s>\n"
        f"<|assistant|>"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            output_scores=True, # Request scores for confidence calculation
            return_dict_in_generate=True
        )
    
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = response.split("<|assistant|>")[-1].strip()

    # Calculate confidence score as the average probability of the generated tokens
    token_probs = []
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[-1]:]
    scores = outputs.scores
    
    for i, token_id in enumerate(generated_ids):
        # scores is a tuple of tensors, one for each generation step
        step_scores = scores[i]
        # Convert logits to probabilities
        step_probs = torch.softmax(step_scores, dim=-1)
        # Get the probability of the chosen token
        token_prob = step_probs[0, token_id].item()
        token_probs.append(token_prob)

    avg_confidence = sum(token_probs) / len(token_probs) if token_probs else 0

    end_time = time.time()
    inference_time = end_time - start_time
    return answer, inference_time, avg_confidence

def display_results(answer, method, response_time, confidence_score="N/A"):
    """Displays the generated answer and performance metrics in a consistent format."""
    st.subheader("Generated Answer")
    st.markdown(answer)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Answer", value="") # Placeholder for alignment
    with col2:
        st.metric(label="Confidence Score", value=confidence_score)
    with col3:
        st.metric(label="Method", value=method)
    with col4:
        st.metric(label="Inference Time", value=f"{response_time:.2f} s")


# --- Main App UI ---
st.set_page_config(page_title="RAG vs. Fine-Tuning", layout="wide")
st.title("RAG vs. Fine-Tuning Comparison 🤖")
st.markdown("This interface allows you to compare responses from RAG and a fine-tuned model.")
components = load_components()

with st.sidebar:
    st.header("Configuration")
    
    # Dynamically create radio options
    radio_options = ["RAG (Retrieval-Augmented Generation)"]
    if components and components.get("tuned_model"):
        radio_options.append("Fine-Tuned Model (TinyLlama)")

    mode = st.radio(
        "Choose the operational mode:",
        radio_options
    )
    st.markdown("---")
    st.info(
        "**RAG**: Finds relevant documents and uses them to answer.\n\n"
        "**Fine-Tuned Model**: Answers from knowledge specialized on your Q&A data."
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
                    with st.spinner("Processing with the RAG pipeline..."):
                        start_time = time.time()
                        retrieved_chunks = retrieve(
                            query,
                            components["embed_model"],
                            components["faiss_index"],
                            components["bm25_index"],
                            components["chunk_data"]
                        )
                        final_answer = components["base_generator"].generate(query, retrieved_chunks)
                        end_time = time.time()
                        response_time = end_time - start_time
                        confidence = f"{retrieved_chunks[0]['score']:.4f}" if retrieved_chunks else "N/A"
                        display_results(final_answer, "RAG", response_time, confidence)

                elif mode == "Fine-Tuned Model (TinyLlama)":
                    with st.spinner("Querying the fine-tuned model..."):
                        answer, response_time, confidence = generate_from_finetuned(
                            components["tuned_model"],
                            components["tuned_tokenizer"],
                            query
                        )
                        display_results(answer, "Fine-Tuned", response_time, confidence_score=f"{confidence:.4f}")
else:
    st.error("Application components could not be loaded. Please check the console for errors.")
