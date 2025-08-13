"""
Document loaders for various content sources.
"""
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
import requests


def load_content(input_type: str, content: str) -> List[Document]:
    """
    Load content from various sources into Document objects.

    Args:
        input_type (str): Type of input source ("url", "pdf", "textfile", "text")
        content (str): The actual content identifier (URL, file path, or text)

    Returns:
        List[Document]: List of Document objects containing the content

    Raises:
        ValueError: If input_type is not supported
        Exception: If there's an error loading the content
    """
    if input_type == "url":
        return _load_web_content(content)
    elif input_type == "pdf":
        return _load_pdf_content(content)
    elif input_type == "textfile":
        return _load_text_file_content(content)
    elif input_type == "text":
        return _load_direct_text_content(content)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")


def _load_web_content(url: str) -> List[Document]:
    """Load content from a web URL."""
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        raise Exception(f"Failed to load web content from {url}: {str(e)}")


def _load_pdf_content(file_path: str) -> List[Document]:
    """Load content from a PDF file."""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        raise Exception(f"Failed to load PDF content from {file_path}: {str(e)}")


def _load_text_file_content(file_path: str) -> List[Document]:
    """Load content from a text file."""
    try:
        loader = TextLoader(file_path)
        return loader.load()
    except Exception as e:
        raise Exception(f"Failed to load text file content from {file_path}: {str(e)}")


def _load_direct_text_content(text: str) -> List[Document]:
    """Wrap direct text content into a Document object."""
    return [Document(page_content=text, metadata={"source": "direct_text"})]