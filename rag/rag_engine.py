# rag/rag_engine.py
# Updated: OpenAI v1 + ChromaDB v0.4.24+ compliant

import os
import hashlib
from dotenv import load_dotenv

import chromadb
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAIEmbedding(EmbeddingFunction):
    """Custom embedding function compatible with ChromaDB v0.4.24+"""
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def __call__(self, input):
        # input: list of strings
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [item.embedding for item in response.data]


def get_vector_client_and_collection():
    """Initialize Chroma client and collection with OpenAI v1 embeddings"""

    # Initialize Chroma persistent client
    client = chromadb.PersistentClient(path="data/chroma")

    # Initialize our custom embedding function
    embedding_func = OpenAIEmbedding(model_name="text-embedding-3-small")

    # Get or create collection
    collection = client.get_or_create_collection(
        name="sales_features",
        embedding_function=embedding_func
    )

    return client, collection, embedding_func


# Global instances (used across the app)
client, collection, embedding_func = get_vector_client_and_collection()


def ingest_to_vector_db(vector_client, embedding_func, text: str, metadata: dict):
    """Ingest a single text chunk with metadata into Chroma"""
    doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()

    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )


def query_vector_db(
    vector_client,
    embedding_func,
    query: str,
    customer_filter: str = None,
    n_results: int = 10
):
    """Query the vector database."""
    where = {"customer_name": customer_filter} if customer_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas"]
    )

    if not results.get("documents") or not results["documents"][0]:
        return []

    retrieved = []
    for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text": doc_text,
            "metadata": meta or {}
        })

    return retrieved
