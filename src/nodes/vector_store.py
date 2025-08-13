"""
Vector store utilities for the LangGraph Question Answering System.
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_model():
    """
    Initialize the embeddings model using HuggingFace.
    This approach works locally without requiring API calls.
    """
    # Get embedding model from environment or use default
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings


def create_vector_store(documents: List[Document]) -> Chroma:
    """
    Create a Chroma vector store from documents.

    Args:
        documents (List[Document]): List of documents to store

    Returns:
        Chroma: Initialized vector store
    """
    embeddings = get_embeddings_model()

    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store


def search_relevant_chunks(vector_store: Chroma, question: str, k: int = 4) -> List[Document]:
    """
    Search for relevant chunks based on the question.

    Args:
        vector_store (Chroma): The vector store to search
        question (str): The question to search for
        k (int): Number of relevant chunks to return

    Returns:
        List[Document]: List of relevant documents
    """
    # Search for similar documents
    relevant_docs = vector_store.similarity_search(question, k=k)
    return relevant_docs