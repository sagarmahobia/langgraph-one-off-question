"""
LangGraph implementation for the Question Answering System.

This implementation now uses a vector database (ChromaDB) to:
1. Embed document chunks when loading content
2. Store embeddings in the vector database
3. For a given question, embed the question and search for the most relevant chunks
4. Process only the relevant chunks to answer the question

This approach is more efficient and accurate for large documents compared to the previous
implementation that processed all chunks.
"""
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_core.documents import Document

from src.nodes.loaders import load_content
from src.nodes.text_splitter import split_documents
from src.nodes.vector_store import create_vector_store, search_relevant_chunks
from src.nodes.answer_node import answer_question_node  # Updated import


# Define the state schema for our graph
class GraphState(TypedDict):
    """State schema for the question answering graph."""
    input_type: str
    content: str
    question: str
    documents: List[Document]
    chunks: List[Document]
    relevant_chunks: List[Document]
    # Removed chunk_answers since we don't need to process chunks individually
    final_answer: str
    chunk_size: int
    chunk_overlap: int
    max_answer_length: int
    vector_store: Any  # Chroma vector store

def load_content_node(state: GraphState) -> Dict[str, Any]:
    """Load content from various sources."""
    documents = load_content(state["input_type"], state["content"])
    return {"documents": documents}


def split_documents_node(state: GraphState) -> Dict[str, Any]:
    """Split documents into chunks."""
    chunks = split_documents(
        state["documents"], 
        state["chunk_size"], 
        state["chunk_overlap"]
    )
    return {"chunks": chunks}


def create_vector_store_node(state: GraphState) -> Dict[str, Any]:
    """Create vector store from document chunks."""
    vector_store = create_vector_store(state["chunks"])
    return {"vector_store": vector_store}


def search_relevant_chunks_node(state: GraphState) -> Dict[str, Any]:
    """Search for relevant chunks based on the question."""
    relevant_chunks = search_relevant_chunks(
        state["vector_store"], 
        state["question"], 
        k=4  # Number of relevant chunks to retrieve
    )
    return {"relevant_chunks": relevant_chunks}


# Removed answer_chunks_node_wrapper since it's no longer needed


def answer_question_node_wrapper(state: GraphState) -> Dict[str, Any]:
    """Answer question directly using relevant chunks."""
    final_answer = answer_question_node(
        state["relevant_chunks"],
        state["question"],
        state["max_answer_length"]
    )
    return {"final_answer": final_answer}


def create_graph():
    """Create and compile the LangGraph workflow."""
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("load_content", load_content_node)
    workflow.add_node("split_documents", split_documents_node)
    workflow.add_node("create_vector_store", create_vector_store_node)
    workflow.add_node("search_relevant_chunks", search_relevant_chunks_node)
    workflow.add_node("answer_question", answer_question_node_wrapper)  # Updated node
    
    # Add edges
    workflow.add_edge("load_content", "split_documents")
    workflow.add_edge("split_documents", "create_vector_store")
    workflow.add_edge("create_vector_store", "search_relevant_chunks")
    workflow.add_edge("search_relevant_chunks", "answer_question")  # Updated edge
    
    # Set entry point
    workflow.set_entry_point("load_content")
    
    # Set finish point
    workflow.add_edge("answer_question", END)  # Updated edge
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def answer_question_with_graph(
    input_type: str,
    content: str,
    question: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    max_answer_length: int = None
) -> str:
    """
    Answer a question using the LangGraph workflow with vector database.
    
    Args:
        input_type (str): Type of input source ("url", "pdf", "textfile", "text")
        content (str): The actual content identifier (URL, file path, or text)
        question (str): Question to answer based on the content
        chunk_size (int, optional): Size of each chunk in characters
        chunk_overlap (int, optional): Overlap between chunks in characters
        max_answer_length (int, optional): Maximum number of sentences in the final answer
        
    Returns:
        str: Answer to the question
    """
    # Use defaults from environment variables if not provided
    import os
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
    
    # Create the graph
    app = create_graph()
    
    # Initialize the state
    initial_state = GraphState(
        input_type=input_type,
        content=content,
        question=question,
        documents=[],
        chunks=[],
        relevant_chunks=[],
        # Removed chunk_answers initialization
        final_answer="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_answer_length=max_answer_length,
        vector_store=None
    )
    
    # Execute the graph
    final_state = app.invoke(initial_state)
    
    return final_state["final_answer"]