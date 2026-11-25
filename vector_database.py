
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from document_processor import load_pdf, split_text


# 1. Configure paths
PDF_FOLDER = Path("data/documents")
VECTOR_FOLDER = "vectorstore"


# 2. Initialize embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 3. Initialize ChromaDB
vector_db = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory=VECTOR_FOLDER
)


def store_documents():

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF documents found!")
        return

    all_documents = []
    all_ids = []

    for pdf_file in pdf_files:

        print(f"Processing: {pdf_file.name}")

        text = load_pdf(pdf_file)
        chunks = split_text(text)

        for index, chunk in enumerate(chunks):

            document = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_file.name,
                    "chunk": index
                }
            )

            all_documents.append(document)
            all_ids.append(f"{pdf_file.name}_{index}")

    # 4. Store embeddings and documents
    if all_documents:
        vector_db.add_documents(
            documents=all_documents,
            ids=all_ids
        )

    print("\nDocuments stored successfully!")
    print("Total chunks stored:", len(all_documents))


def search_documents(query):

    results = vector_db.similarity_search(
        query,
        k=3
    )

    return results


if __name__ == "__main__":

    store_documents()

    question = "What is the purpose of the student portfolio website?"

    print("\nSearching for:", question)

    results = search_documents(question)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}:")
        print("Source:", result.metadata["source"])
        print("Content:", result.page_content)
