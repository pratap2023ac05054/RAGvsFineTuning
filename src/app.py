# app.py

import streamlit as st
import time
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

from response_generator import ResponseGenerator
from guardrails import validate_query, validate_response
from hybrid_retrieval import retrieve

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_DATA_PATH = "chunk_data.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BASE_GENERATOR_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"

# --- Caching ---
@st.cache_resource
def load_components():
    """Loads all necessary models and data for the RAG pipeline."""
    print("Loading RAG components...")
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

        print("All RAG components loaded successfully.")
        return components
        
    except FileNotFoundError as e:
        st.error(f"Error loading components: {e}. Please ensure index files are present and pushed to your repository.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading components: {e}")
        return None

def display_results(answer, response_time, top_score, validation_result):
    """Displays the generated answer and performance metrics."""
    st.subheader("Generated Answer")
    
    # Display Guardrail status
    if validation_result["pass"]:
        st.success(f"✅ Guardrail Passed: {validation_result['reason']}")
    else:
        st.warning(f"⚠️ Guardrail Flagged: {validation_result['reason']}")
        st.caption(validation_result.get('details', ''))

    st.markdown(answer)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Method", value="RAG")
    with col2:
        st.metric(label="Inference Time", value=f"{response_time:.2f} s")
    with col3:
        st.metric(label="Top Document Score", value=top_score)

# --- Main App UI ---
st.set_page_config(page_title="Medtronic 10-K RAG System", layout="wide")
st.title("Medtronic 10-K RAG System 🤖")
st.markdown("This application uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on Medtronic's financial reports.")
components = load_components()

with st.sidebar:
    st.header("About")
    st.info(
        "This system finds relevant documents from Medtronic's 10-K reports using a hybrid search algorithm and then uses a language model to generate a natural language answer."
    )

if components:
    query = st.text_input("Enter your question here:", key="query_input")
    if st.button("Ask Question", key="ask_button"):
        if not query:
            st.warning("Please enter a question.")
        else:
            # 1. Input Guardrail
            is_valid, message = validate_query(query)
            if not is_valid:
                st.error(f"Input Error: {message}")
            else:
                with st.spinner("Processing with the RAG pipeline..."):
                    start_time = time.time()
                    
                    # 2. Retrieve Documents
                    retrieved_chunks = retrieve(
                        query,
                        components["embed_model"],
                        components["faiss_index"],
                        components["bm25_index"],
                        components["chunk_data"]
                    )
                    
                    # 3. Generate Answer
                    final_answer = components["base_generator"].generate(query, retrieved_chunks)
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # 4. Output Guardrail
                    context_for_validation = [chunk['text'] for chunk in retrieved_chunks[:5]]
                    validation_result = validate_response(final_answer, context_for_validation)
                    
                    top_score = f"{retrieved_chunks[0]['score']:.4f}" if retrieved_chunks else "N/A"
                    
                    display_results(final_answer, response_time, top_score, validation_result)

                    # 5. Display Sources
                    st.subheader("Sources")
                    for i, chunk in enumerate(retrieved_chunks[:3]): # Show top 3 sources
                        with st.expander(f"Source {i+1} (Score: {chunk['score']:.4f})"):
                            st.markdown(chunk['text'])
else:
    st.error("Application components could not be loaded. Please check the terminal for errors.")