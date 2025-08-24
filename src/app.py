# app.py

import streamlit as st
import time
import faiss
import pickle
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from response_generator import ResponseGenerator
from guardrails import validate_query
from hybrid_retrieval import retrieve

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BASE_GENERATOR_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"

# Paths for the fine-tuned model
FINETUNED_MODEL_PATH = "./tinyllama-finetuned-merged"

# --- Caching ---
@st.cache_resource
def load_components():
    """Loads all necessary models and data for all modes."""
    print("Loading application components...")
    components = {}
    try:
        # Load RAG components
        components["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
        components["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)
        with open(BM25_INDEX_PATH, 'rb') as f:
            components["bm25_index"] = pickle.load(f)
        with open(CHUNK_DATA_PATH, 'rb') as f:
            components["chunk_data"] = pickle.load(f)
        components["base_generator"] = ResponseGenerator(model_name=BASE_GENERATOR_MODEL)
        print("RAG components loaded successfully.")

        # Load the fine-tuned model directly from transformers
        if os.path.exists(FINETUNED_MODEL_PATH):
            print(f"Loading fine-tuned model from {FINETUNED_MODEL_PATH}...")
            device = "cpu"
            tuned_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH).to(device)
            tuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
            if tuned_tokenizer.pad_token is None:
                tuned_tokenizer.pad_token = tuned_tokenizer.eos_token

            components["tuned_model"] = tuned_model
            components["tuned_tokenizer"] = tuned_tokenizer
            print("Fine-tuned model loaded successfully.")
        else:
            st.warning(f"Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'. The fine-tuned option will be disabled.")
            components["tuned_model"] = None

        return components

    except FileNotFoundError as e:
        st.error(f"Error loading components: {e}. Please ensure index files are present.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading components: {e}")
        return None

def generate_from_finetuned(model, tokenizer, query: str):
    """Generates a response from the fine-tuned transformers model."""
    start_time = time.time()
    device = "cpu"

    prompt = (
        f"<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{query}</s>\n"
        f"<|assistant|>"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the response
    with torch.no_grad():
        # Add parameters to get token scores
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250, 
            temperature=0.7, 
            top_p=0.9,
            do_sample=True,
            output_scores=True, # Request scores
            return_dict_in_generate=True # Return a dictionary
        )
    
    # Decode the generated text
    answer = tokenizer.decode(outputs.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # --- Calculate Confidence Score ---
    # Get the scores for each generated token
    scores = outputs.scores
    # Get the IDs of the generated tokens
    generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
    
    # Calculate the average probability of the generated sequence
    token_probs = []
    for i, score in enumerate(scores):
        # Get the probability distribution for the current token
        prob_dist = torch.softmax(score, dim=-1)
        # Get the probability of the token that was actually chosen
        token_prob = prob_dist[0, generated_ids[i]].item()
        token_probs.append(token_prob)

    # The confidence is the average probability
    confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
    # --- End of Confidence Calculation ---

    end_time = time.time()
    inference_time = end_time - start_time
    
    # Return the calculated confidence score instead of None
    return answer, inference_time, confidence

def display_results(answer, method, response_time, confidence_score="N/A", confidence_label="Score"):
    """Displays the generated answer and performance metrics."""
    st.subheader("Generated Answer")
    st.markdown(answer)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Method", value=method)
    with col2:
        st.metric(label="Inference Time", value=f"{response_time:.2f} s")
    with col3:
        st.metric(label=confidence_label, value=confidence_score)

# --- Main App UI ---
st.set_page_config(page_title="RAG vs. Fine-Tuning", layout="wide")
st.title("RAG vs. Fine-Tuning Comparison 🤖")
st.markdown("This application allows you to compare responses from a RAG pipeline and a fine-tuned model on Medtronic's financial data.")
components = load_components()

with st.sidebar:
    st.header("Configuration")

    radio_options = ["RAG (Retrieval-Augmented Generation)"]
    if components and components.get("tuned_model"):
        radio_options.append("Fine-Tuned Model")

    mode = st.radio(
        "Choose the operational mode:",
        radio_options,
        key="mode_radio"
    )
    st.markdown("---")
    st.info(
        "**RAG**: Finds relevant documents and uses them as context to answer. Best for specific, factual questions.\n\n"
        "**Fine-Tuned Model**: Generates an answer live from knowledge specialized during its training."
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
                        top_score = f"{retrieved_chunks[0]['score']:.4f}" if retrieved_chunks else "N/A"
                        display_results(final_answer, "RAG", response_time, top_score, confidence_label="Top Document Score")

                        st.subheader("Sources")
                        for i, chunk in enumerate(retrieved_chunks[:3]):
                            with st.expander(f"Source {i+1} (Score: {chunk['score']:.4f})"):
                                st.markdown(chunk['text'])

                elif mode == "Fine-Tuned Model":
                    with st.spinner("Querying the fine-tuned model..."):
                        answer, response_time, confidence = generate_from_finetuned(
                            components["tuned_model"],
                            components["tuned_tokenizer"],
                            query
                        )
                        # Use the 'confidence' variable and format it
                        display_results(answer, "Fine-Tuned", response_time, confidence_score=f"{confidence:.4f}", confidence_label="Avg. Token Prob.")
else:
    st.error("Application components could not be loaded. Please check the terminal for errors.")