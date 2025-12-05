from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def create_vector_db(docs):
    qdrant = QdrantClient(url="http://localhost:6333")

    vector_db = QdrantVectorStore.from_documents(
        docs,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name="pdf_rag"
    )
    return vector_db


def ask_question(query, vector_db):
    search_results = vector_db.similarity_search(query=query, k=5)

    context = "\n\n".join([
        f"Content: {doc.page_content}\nPage: {doc.metadata.get('page_label')}"
        for doc in search_results
    ])

    system_prompt = f"""
    You are an expert assistant. Answer strictly from the context with page number in next line after the content.
    Rules:
    1. Answer only questions related to pdf.
    2. if anything not in pdf but is related to pdf then ans This pdf dont contains this content.
    3.If any user asks any question which is not related to pdf then ans Sorry, I can help only with pdf questions.
    4.Follow the above rules very stictly. 
    5.Give content answer only from the current uploaded pdf
    CONTEXT:
    {context}
    """

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content

