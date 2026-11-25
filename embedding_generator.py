
from langchain_ollama import OllamaEmbeddings

# Initialize the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Sample text
texts = [
    "Python is a programming language.",
    "Machine learning is a branch of artificial intelligence."
]

# Generate embeddings
vectors = embeddings.embed_documents(texts)

# Display results
print("Embeddings generated successfully!")

for i, vector in enumerate(vectors):
    print(f"\nText {i + 1}")
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])
