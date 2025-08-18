import re

# Allow: ASCII letters/digits, SPACE, ?, ., ,, -
# Note: no tabs/newlines allowed; only a literal space.
_ALLOWED_QUERY_RE = re.compile(r"[A-Za-z0-9 ?.,-]+")

def validate_query(query: str) -> bool:
    # Treat pure whitespace as invalid
    if query is None:
        return False
    q = query.strip()
    if not q:
        return False
    return bool(_ALLOWED_QUERY_RE.fullmatch(q))

def main():
    query = input("Enter your query: ")
    if validate_query(query):
        print("Valid query")
    else:
        print("Invalid query")

if __name__ == "__main__":
    main()
