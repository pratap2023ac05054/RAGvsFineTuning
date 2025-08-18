import re
import unicodedata
from typing import Dict, Any, List

# ----------------------- Config (tune as needed) -----------------------

MAX_QUERY_CHARS = 300

# For domain-relevance (optional): adjust to your use-case
DOMAIN_KEYWORDS = {
    "medtronic", "fy23", "fy24", "financial", "finance", "revenue", "earnings",
    "operating", "margin", "guidance", "10-k", "10q", "10-q", "10k", "ebitda",
    "cash flow", "free cash", "capex", "opex", "gross margin"
}

# Harmful/abuse/prohibited intents (non-exhaustive)
HARMFUL_KEYWORDS = {
    # illegal / violent / self-harm
    "make a bomb", "explosive", "kill", "harm yourself", "suicide", "self-harm",
    # cybercrime / malware
    "hack", "ddos", "backdoor", "keylogger", "ransomware", "malware",
    "sql injection", "xss payload", "csrf attack",
    # explicit wrongdoing
    "buy stolen", "fake id", "counterfeit", "credit card generator",
}

# Prompt-injection telltales
INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+all\s+prior\s+rules",
    r"reveal\s+system\s+prompt",
    r"act\s+as\s+.*(developer|system|root)",
    r"jailbreak",
    r"bypass\s+(safety|guardrail|filters?)",
]

# Sensitive info / PII markers
SENSITIVE_KEYWORDS = {
    "password", "api key", "apikey", "secret key", "otp", "one-time password",
    "pin", "cvv", "ssn", "aadhaar", "pan", "bank account", "routing number"
}

# ----------------------- Helpers -----------------------

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_IN_RE = re.compile(r"\b(?:\+?91[-\s]?)?[6-9]\d{9}\b")  # India mobile heuristic
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")               # Indian PAN format
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")        # 12 digits (loose)

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _ANSI_RE.sub("", s)
    # drop control chars except newline/tab
    s = "".join(ch for ch in s if ch in ("\n", "\t") or ord(ch) >= 32)
    # collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s).strip()
    return s

def _luhn_check(num: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", num)]
    if len(digits) < 12:  # too short to be a typical card
        return False
    s = 0
    alt = False
    for d in reversed(digits):
        s += d * 2 - 9 if alt and d > 4 else (d * 2 if alt else d)
        alt = not alt
    return s % 10 == 0

def _detect_cards(text: str) -> bool:
    # simple scan for 13-19 digit sequences that pass Luhn
    for m in re.finditer(r"(?:\d[ -]?){13,19}", text):
        if _luhn_check(m.group()):
            return True
    return False

def _contains_any(text: str, keywords: List[str]) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)

def _matches_any(text: str, patterns: List[str]) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)

# ----------------------- Main guardrail -----------------------

def guard_query(query: str) -> Dict[str, Any]:
    """
    Returns:
        {
          "action": "allow" | "warn" | "block",
          "sanitized": <string>,
          "reasons": [<codes>],
          "message": <human-readable summary>
        }
    """
    reasons = []
    sanitized = _normalize(query)

    if not sanitized:
        return {"action": "block", "sanitized": "", "reasons": ["empty"], "message": "Please enter a non-empty query."}

    if len(sanitized) > MAX_QUERY_CHARS:
        return {
            "action": "block",
            "sanitized": sanitized[:MAX_QUERY_CHARS] + "…",
            "reasons": ["too_long"],
            "message": f"Query too long (>{MAX_QUERY_CHARS} chars). Please shorten it."
        }

    # Harmful intent
    if _contains_any(sanitized, list(HARMFUL_KEYWORDS)):
        reasons.append("harmful_intent")

    # Prompt injection
    if _matches_any(sanitized, INJECTION_PATTERNS):
        reasons.append("prompt_injection")

    # Sensitive content / PII
    pii_hits = []
    if _EMAIL_RE.search(sanitized): pii_hits.append("email")
    if _PHONE_IN_RE.search(sanitized): pii_hits.append("phone")
    if _PAN_RE.search(sanitized): pii_hits.append("pan")
    if _AADHAAR_RE.search(sanitized): pii_hits.append("aadhaar")
    if _detect_cards(sanitized): pii_hits.append("card_number")
    if _contains_any(sanitized, list(SENSITIVE_KEYWORDS)): pii_hits.append("secrets")
    if pii_hits:
        reasons.append(f"pii:{','.join(pii_hits)}")

    if reasons:
        return {
            "action": "block",
            "sanitized": sanitized,
            "reasons": reasons,
            "message": "Query blocked due to safety/PII concerns. Please remove sensitive or harmful content."
        }

    # Optional domain relevance warning (soft)
    if DOMAIN_KEYWORDS and not _contains_any(sanitized, list(DOMAIN_KEYWORDS)):
        return {
            "action": "warn",
            "sanitized": sanitized,
            "reasons": ["off_domain"],
            "message": "Query may be off-domain for this assistant (financial/Medtronic focus). Proceeding anyway."
        }

    return {"action": "allow", "sanitized": sanitized, "reasons": [], "message": "OK"}

# ----------------------- Example usage -----------------------

if __name__ == "__main__":
    q = input("Enter your query: ")
    verdict = guard_query(q)
    print(f"[{verdict['action'].upper()}] {verdict['message']}")
    if verdict["reasons"]:
        print("Reasons:", ", ".join(verdict["reasons"]))
    print("Sanitized:", verdict["sanitized"])
