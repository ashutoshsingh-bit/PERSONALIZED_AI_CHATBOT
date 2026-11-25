
from langchain_ollama import ChatOllama
from vector_database import search_documents

# Load Llama 3.2
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def ask_question(question):

    # Step 1: Retrieve relevant documents
    documents = search_documents(question)

    if not documents:
        return "I could not find relevant information in the documents."

    # Step 2: Prepare context
    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    # Step 3: Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the
provided document context.

If the answer is not available in the context,
clearly say that the information was not found
in the uploaded documents.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # Step 4: Generate answer
    response = llm.invoke(prompt)

    return response.content


# Test the RAG pipeline
if __name__ == "__main__":

    question = "What is the purpose of the student portfolio website?"

    print("\nQuestion:", question)

    answer = ask_question(question)

    print("\nChatbot Answer:")
    print(answer)
