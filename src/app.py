# app.py

import streamlit as st
import time
import faiss
import pickle
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from response_generator import ResponseGenerator
from guardrails import validate_query
from hybrid_retrieval import retrieve

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BASE_GENERATOR_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FINETUNED_ADAPTER_PATH = "./tinyllama-finetuned-adapter-cpu" 
ADAPTER_NAME = "medtronic_qa" # The name given to the adapter during training

# --- Caching ---
@st.cache_resource
def load_components():
    """Loads all necessary models and data for both RAG and Fine-Tuned modes."""
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

        # Load Fine-Tuned Adapter Model
        if os.path.exists(FINETUNED_ADAPTER_PATH):
            print(f"Loading fine-tuned adapter from {FINETUNED_ADAPTER_PATH}...")
            base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            # Load the adapter into the base model
            model_with_adapter = base_model.load_adapter(FINETUNED_ADAPTER_PATH)
            
            components["tuned_model"] = model_with_adapter
            components["tuned_tokenizer"] = AutoTokenizer.from_pretrained(FINETUNED_ADAPTER_PATH)
            components["adapter_name"] = ADAPTER_NAME
            print("Fine-tuned adapter loaded successfully.")
        else:
            st.warning(f"Fine-tuned adapter not found at '{FINETUNED_ADAPTER_PATH}'. The fine-tuned option will be disabled.")
            components["tuned_model"] = None

        return components
        
    except FileNotFoundError as e:
        st.error(f"Error loading components: {e}. Please ensure index files are present.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading components: {e}")
        return None

def generate_from_finetuned(model, tokenizer, query: str, adapter_name: str):
    """Generates a response from the fine-tuned adapter model and calculates a confidence score."""
    start_time = time.time()
    device = "cpu"
    model.to(device)
    model.active_adapters = adapter_name # Activate the adapter

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
            output_scores=True,
            return_dict_in_generate=True
        )
    
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = response.split("<|assistant|>")[-1].strip()

    # Calculate confidence score as the average probability of the generated tokens
    token_probs = []
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[-1]:]
    scores = outputs.scores
    
    for i, token_id in enumerate(generated_ids):
        step_scores = scores[i]
        step_probs = torch.softmax(step_scores, dim=-1)
        token_prob = step_probs[0, token_id].item()
        token_probs.append(token_prob)

    avg_confidence = sum(token_probs) / len(token_probs) if token_probs else 0
    end_time = time.time()
    inference_time = end_time - start_time
    return answer, inference_time, avg_confidence

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
        radio_options.append("Fine-Tuned Model (Adapter)")

    mode = st.radio(
        "Choose the operational mode:",
        radio_options,
        key="mode_radio"
    )
    st.markdown("---")
    st.info(
        "**RAG**: Finds relevant documents from Medtronic's 10-K reports and uses them as context to answer. Best for specific, factual questions.\n\n"
        "**Fine-Tuned Model**: Answers from knowledge specialized during its training on the Q&A dataset. Best for questions similar to the training data."
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
                        for i, chunk in enumerate(retrieved_chunks[:3]): # Show top 3 sources
                            with st.expander(f"Source {i+1} (Score: {chunk['score']:.4f})"):
                                st.markdown(chunk['text'])

                elif mode == "Fine-Tuned Model (Adapter)":
                    with st.spinner("Querying the fine-tuned model..."):
                        answer, response_time, confidence = generate_from_finetuned(
                            components["tuned_model"],
                            components["tuned_tokenizer"],
                            query,
                            components["adapter_name"]
                        )
                        display_results(answer, "Fine-Tuned", response_time, confidence_score=f"{confidence:.4f}", confidence_label="Confidence Score")
else:
    st.error("Application components could not be loaded. Please check the terminal for errors.")