"""
Text splitting utility for chunking large documents.

This utility splits documents into chunks that will be embedded and stored
in a vector database for efficient similarity search.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def split_documents(
    documents: List[Document], 
    chunk_size: int = None, 
    chunk_overlap: int = None
) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents (List[Document]): List of Document objects to split
        chunk_size (int, optional): Size of each chunk in characters
        chunk_overlap (int, optional): Overlap between chunks in characters
        
    Returns:
        List[Document]: List of chunked Document objects
    """
    # Use provided values or get from environment variables
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))


    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split documents
    splits = text_splitter.split_documents(documents)
    return splits