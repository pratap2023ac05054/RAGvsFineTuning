# utils.py

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- NLTK Data Download Logic ---
# This block ensures the necessary NLTK data is available before proceeding.
try:
    print("Verifying NLTK data packages...")
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print("NLTK packages are already downloaded.")
except LookupError:
    print("One or more NLTK data packages not found. Downloading...")
    nltk.download('punkt', quiet=False)
    nltk.download('stopwords', quiet=False)
    print("NLTK data download complete.")

# --- Pre-computation ---
STOPWORDS = set(stopwords.words('english'))

# --- Shared Function ---
def preprocess_query(query: str) -> tuple[str, list[str]]:
    """Cleans, lowercases, and removes stopwords from a query."""
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(query)
    filtered_tokens = [word for word in tokens if word not in STOPWORDS and word.isalnum()]
    return " ".join(filtered_tokens), filtered_tokens