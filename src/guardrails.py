# guardrails.py

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Configuration ---
MIN_QUERY_LENGTH = 3
# Using a set for faster lookups
BLOCKLIST = {
    "how to make a bomb", "self harm", "suicide", "illegal activities",
    "create malware", "how to kill", "inappropriate content"
}
# Threshold for the fraction of answer words that must be in the context
COVERAGE_THRESHOLD = 0.35

# --- Helper Functions ---

def simple_tokenize(text: str) -> set[str]:
    """Tokenizes text, removes stopwords, and returns a set of unique words."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return {word for word in tokens if word.isalnum() and word not in stop_words}

def extract_numbers(text: str) -> set[str]:
    """Extracts and normalizes numbers, percentages, and currency values from text."""
    # This regex finds integers, decimals, percentages, and currency amounts
    pattern = r'(\$?\d[\d,]*\.?\d*\s*%?)'
    matches = re.findall(pattern, text)
    # Normalize by removing commas, dollar signs, and whitespace
    return {re.sub(r'[,\$\s]', '', match) for match in matches}

# --- Input Guardrail ---

def validate_query(query: str) -> tuple[bool, str]:
    """
    Validates a user query against a set of input-side guardrails.

    Args:
        query (str): The user's input query.

    Returns:
        A tuple containing:
        - bool: True if the query is valid, False otherwise.
        - str: An error message if invalid, or an empty string if valid.
    """
    if not query or len(query.strip()) < MIN_QUERY_LENGTH:
        return False, f"Query is too short. Please provide at least {MIN_QUERY_LENGTH} characters."

    query_lower = query.lower()
    for blocked_term in BLOCKLIST:
        if blocked_term in query_lower:
            return False, "This query is not allowed due to safety guidelines."

    return True, ""

# --- Output Guardrail ---

def validate_response(generated_answer: str, retrieved_contexts: list[str]) -> dict:
    """
    Validates a generated answer against the retrieved context to flag hallucinations.

    Args:
        generated_answer (str): The answer produced by the language model.
        retrieved_contexts (list[str]): The list of context strings fed to the model.

    Returns:
        A dictionary containing the validation result and reasons.
    """
    if not generated_answer.strip():
        return {"pass": False, "reason": "Generated answer is empty."}

    full_context = " ".join(retrieved_contexts)

    # 1. Grounding/Coverage Check
    answer_tokens = simple_tokenize(generated_answer)
    context_tokens = simple_tokenize(full_context)
    
    if not answer_tokens: # Handle cases where the answer is only stopwords
        return {"pass": True, "reason": "Answer contains no valid tokens to check."}

    common_tokens = answer_tokens.intersection(context_tokens)
    coverage = len(common_tokens) / len(answer_tokens)

    if coverage < COVERAGE_THRESHOLD:
        return {
            "pass": False,
            "reason": "Low Grounding Score",
            "details": f"The answer is not well-supported by the context. Coverage score: {coverage:.2f} (Threshold: {COVERAGE_THRESHOLD})"
        }

    # 2. Factual Consistency (Numeric Check)
    answer_numbers = extract_numbers(generated_answer)
    context_numbers = extract_numbers(full_context)

    # Check if any number in the answer is NOT in the context
    unsupported_numbers = answer_numbers - context_numbers
    if unsupported_numbers:
        return {
            "pass": False,
            "reason": "Numeric Inconsistency",
            "details": f"The following numbers in the answer were not found in the context: {', '.join(unsupported_numbers)}"
        }

    return {"pass": True, "reason": None}
