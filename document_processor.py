
import pymupdf
from pathlib import Path


def load_pdf(file_path):
    """Extract text from a PDF document."""

    document = pymupdf.open(file_path)
    extracted_text = ""

    for page_number, page in enumerate(document):
        page_text = page.get_text()
        extracted_text += page_text + "\n"

    document.close()

    return extracted_text


def split_text(text, chunk_size=500, overlap=100):
    """Split extracted text into smaller chunks."""

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    pdf_folder = Path("data/documents")

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
    else:
        for pdf_file in pdf_files:

            print(f"\nProcessing: {pdf_file.name}")

            text = load_pdf(pdf_file)

            chunks = split_text(text)

            print("Text extracted successfully!")
            print("Total characters:", len(text))
            print("Total chunks:", len(chunks))

            if chunks:
                print("\nFirst chunk:")
                print(chunks[0])
