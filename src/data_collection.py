import pdfplumber
import os

def pdf_to_text(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:                # avoid None -> TypeError
                chunks.append(page_text)
            else:
                # Keep page boundaries even if empty/unextractable
                chunks.append(f"[No extractable text on page {i}]")
    # Join with double newline to separate pages
    return "\n\n".join(chunks)

def save_text(text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    input_dir = "data/rawfiles"
    output_dir = "data/processedfiles"

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):        # handle .PDF, .Pdf, etc.
            pdf_path = os.path.join(input_dir, filename)
            try:
                text = pdf_to_text(pdf_path)
            except Exception as e:
                # Optional: write an error file instead of crashing
                text = f"[Error processing {filename}: {e}]"

            output_name = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_dir, output_name)
            save_text(text, output_path)

if __name__ == "__main__":
    main()
