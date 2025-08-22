import streamlit as st
import subprocess
import json
import time
import faiss
import sys

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

# --- Page Configuration ---
st.set_page_config(
    page_title="Query Interface",
    page_icon="🤖",
    layout="wide"
)

# --- UI Elements ---
st.title("📄 RAG and Fine-Tuning Query Interface")
st.markdown("Enter your query below and choose a mode to get an answer from the system.")

# Mode selection
mode = st.radio(
    "Select Mode:",
    ("RAG", "Fine-Tuned"),
    horizontal=True,
    help="**RAG**: Retrieves relevant documents to augment the query. **Fine-Tuned**: Uses a model specially trained on a specific domain."
)

# User query input
user_query = st.text_input("Enter your query:", "")

# Submit button
if st.button("Submit Query"):
    if user_query:
        # --- Backend Processing ---
        with st.spinner("Processing your query... This may take a moment as the model loads for the first time."):
            try:
                # --- FIX: Pass the query as a named argument '--query' ---
                command = ["python", "src/hybrid_retrieval.py", "--query", user_query, "--mode", mode]
                
                # Execute the script as a subprocess
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True  # This will raise an error if the script fails
                )
                
                # Find the JSON output from the last line of stdout
                output_lines = process.stdout.strip().split('\n')
                json_output = output_lines[-1]
                
                results = json.loads(json_output)

                # --- Display Results ---
                st.success("Query processed successfully!")
                
                st.subheader("Answer:")
                st.markdown(f"> {results.get('answer', 'No answer provided.')}")

                st.subheader("Response Details:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Response Time", f"{results.get('response_time', 0):.2f} s")
                col2.metric("Confidence Score", f"{results.get('confidence_score', 0) * 100:.2f}%")
                col3.metric("Method Used", results.get('method_used', 'N/A'))

            except subprocess.CalledProcessError as e:
                st.error("An error occurred while running the backend script.")
                st.code(f"Error Output:\n{e.stderr}") # Display the error for debugging
            except (json.JSONDecodeError, IndexError):
                st.error("Failed to decode the JSON response from the backend.")
                st.code(f"Received Output:\n{process.stdout}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a query.")
