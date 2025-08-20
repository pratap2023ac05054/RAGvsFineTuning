import os
import re
from typing import List, Tuple, Dict

import pdfplumber
from pdf2image import convert_from_path
import pytesseract

# Optional environment overrides
POPPLER_PATH = os.environ.get("POPPLER_PATH")  # e.g., r"C:\poppler-24.07.0\Library\bin"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")  # e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ------------------------------- Extraction -------------------------------

def extract_with_pdfplumber(pdf_path: str) -> List[str]:
    """Return list of page texts using pdfplumber (layout/text-based PDFs)."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages.append(txt)
    except Exception:
        return []
    return pages


def extract_with_ocr(pdf_path: str, dpi: int = 300, lang: str = "eng") -> List[str]:
    """Return list of page texts via OCR (scanned PDFs)."""
    texts: List[str] = []
    try:
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
    except Exception:
        return []

    for img in images:
        try:
            text = pytesseract.image_to_string(img, lang=lang)
        except Exception:
            text = ""
        texts.append(text)
    return texts


def robust_extract_pages(pdf_path: str) -> List[str]:
    """Try pdfplumber first; if low text density, fall back to OCR."""
    pages = extract_with_pdfplumber(pdf_path)
    total_chars = sum(len(p or "") for p in pages)
    if not pages or (len(pages) > 0 and (total_chars / max(len(pages), 1) < 100)):
        pages = extract_with_ocr(pdf_path)
    return pages


# ------------------------------- Cleaning -------------------------------

_PAGE_NUMBER_PATTERNS = [
    r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",
    r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$",
    r"^\s*[ivxlcdm]+\s*$",           # roman numerals
    r"^\s*page\s*[ivxlcdm]+\s*$",
]

def is_page_number(line: str) -> bool:
    s = line.strip().lower()
    for pat in _PAGE_NUMBER_PATTERNS:
        if re.match(pat, s):
            return True
    return False


def _normalize_for_repeat(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    return s


def detect_repeating_headers_footers(
    pages_lines: List[List[str]],
    top_k: int = 2,
    bottom_k: int = 2,
    min_repeat_ratio: float = 0.5,
) -> Tuple[set, set]:
    """Identify repeated top/bottom lines across pages (likely headers/footers)."""
    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}
    num_pages = len(pages_lines)

    for lines in pages_lines:
        if not lines:
            continue
        top = lines[: min(top_k, len(lines))]
        bottom = lines[-min(bottom_k, len(lines)) :] if lines else []

        for l in top:
            key = _normalize_for_repeat(l)
            if key:
                header_counts[key] = header_counts.get(key, 0) + 1
        for l in bottom:
            key = _normalize_for_repeat(l)
            if key:
                footer_counts[key] = footer_counts.get(key, 0) + 1

    thr = max(1, int(min_repeat_ratio * num_pages))
    headers = {k for k, c in header_counts.items() if c >= thr}
    footers = {k for k, c in footer_counts.items() if c >= thr}
    return headers, footers


def clean_pages(pages: List[str]) -> str:
    """Remove headers/footers & page numbers; reflow paragraphs & fix hyphenation."""
    pages_lines: List[List[str]] = [p.splitlines() for p in pages]
    headers, footers = detect_repeating_headers_footers(pages_lines)

    cleaned_pages: List[str] = []
    for lines in pages_lines:
        new_lines: List[str] = []
        for line in lines:
            norm = _normalize_for_repeat(line)
            if not line.strip():
                new_lines.append("")
                continue
            if norm in headers or norm in footers:
                continue
            if is_page_number(line):
                continue
            new_lines.append(line)

        # Reflow simple hard-wrapped paragraphs
        reflowed: List[str] = []
        buf = ""
        for ln in new_lines:
            s = ln.strip()
            if not s:
                if buf:
                    reflowed.append(buf.strip())
                    buf = ""
                reflowed.append("")
                continue

            # Treat likely headings as their own lines
            is_heading = (
                (len(s) <= 80 and s.isupper()) or
                (len(s) <= 80 and re.match(r"^([A-Z][\w\-']+\s+){1,6}[A-Z][\w\-']+$", s or ""))
            )
            if is_heading:
                if buf:
                    reflowed.append(buf.strip())
                    buf = ""
                reflowed.append(s)
                continue

            if buf and not re.search(r"[.!?:\"”’)\]]\s*$", buf):
                buf += " " + s.lstrip("-–— ")
            else:
                if buf:
                    reflowed.append(buf.strip())
                buf = s
        if buf:
            reflowed.append(buf.strip())

        cleaned_pages.append("\n".join(reflowed))

    text = "\n\n".join(cleaned_pages)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # fix word-\nwrap → wordwrap
    text = re.sub(r"[ \t]+", " ", text)            # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text).strip() # collapse blank lines
    return text


# ------------------------------- I/O Pipeline -------------------------------

def pdf_to_clean_text(pdf_path: str) -> str:
    pages = robust_extract_pages(pdf_path)
    if not pages:
        return ""
    return clean_pages(pages)


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    input_root = "data"
    output_root = "out"

    if not os.path.isdir(input_root):
        return

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not filename.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(dirpath, filename)

            # Mirror relative structure under out/
            rel_dir = os.path.relpath(dirpath, input_root)
            base = os.path.splitext(filename)[0]
            out_dir = os.path.join(output_root, rel_dir) if rel_dir != "." else output_root
            out_path = os.path.join(out_dir, f"{base}.txt")

            text = pdf_to_clean_text(pdf_path)
            ensure_parent_dir(out_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
