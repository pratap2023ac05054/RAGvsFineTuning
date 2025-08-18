import re
import unicodedata
from typing import Any

# Strip ANSI escape sequences (prevents terminal injection)
_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def filter_response(response: Any, max_len: int = 4000) -> str:
    """Normalize and sanitize a response string for console output."""
    if response is None:
        return ""
    s = str(response)

    # Unicode normalization (helps with look-alikes and consistency)
    s = unicodedata.normalize("NFKC", s)

    # Remove ANSI escapes (safer for terminals)
    s = _ANSI_RE.sub("", s)

    # Drop control chars except newline and tab
    s = "".join(ch for ch in s if ch in ("\n", "\t") or ord(ch) >= 32)

    # Collapse excessive whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s).strip()

    # Clamp length
    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s

def main():
    response = "This is a \x1b[31msample\x1b[0m  response.\n   With   extra    spaces."
    filtered_response = filter_response(response)
    print(filtered_response)

if __name__ == "__main__":
    main()