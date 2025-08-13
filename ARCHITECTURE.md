# Architecture Overview

This document outlines the high-level architecture of the LangGraph Question Answering System.

## Core Components

1.  **Input Handler:**
    *   **Purpose:** Determines the type of input source (URL, PDF, Text File, Direct Text) and the question to be answered.
    *   **Logic:** Routes the input to the appropriate loader based on its type.

2.  **Abstract Loader:**
    *   **Purpose:** Provides a unified interface for loading content from diverse sources.
    *   **Implementations:**
        *   `WebLoader`: Fetches and parses HTML content from a URL.
        *   `PDFLoader`: Extracts text from PDF documents.
        *   `TextFileLoader`: Reads plain text files.
        *   `DirectTextLoader`: Wraps raw text input into a document format.
    *   **Output:** Produces a standardized `Document` object (or list of `Document` objects) containing the text and potentially metadata.

3.  **Text Splitter:**
    *   **Purpose:** Segments large documents into smaller chunks to fit within the LLM's context window.
    *   **Configuration:** Chunk size and overlap are configurable (e.g., via environment variables).
    *   **Output:** A list of `Document` chunks.

4.  **Vector Database:**
    *   **Purpose:** Stores document embeddings for efficient similarity search.
    *   **Implementation:** Uses ChromaDB with HuggingFace embeddings.
    *   **Process:** 
        *   Creates embeddings for all document chunks
        *   Stores embeddings with associated document content
        *   Searches for relevant chunks based on question embeddings

5.  **LLM Manager:**
    *   **Purpose:** Manages the connection and interaction with the LLM via the OpenRouter API.
    *   **Configuration:** Reads `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `LLM_MODEL` from environment variables.
    *   **Wrapper:** Utilizes a LangChain LLM wrapper (e.g., `ChatOpenAI`) configured for OpenRouter compatibility.

6.  **Question Answering Nodes (LangGraph):**
    *   **`Relevant_Chunk_Searcher`:**
        *   **Input:** Vector store and user question.
        *   **Process:** Searches vector database for chunks most relevant to the question.
        *   **Output:** List of relevant document chunks.
    *   **`Answer_Question_Node`:**
        *   **Input:** List of relevant `Document` chunks and the user's question.
        *   **Process:** Combines relevant chunks into context and sends to the LLM with a question answering prompt.
        *   **Output:** Final answer.
    *   **`Output_Node`:**
        *   **Input:** Final answer string.
        *   **Process:** Returns the answer to the caller or handles further output (e.g., printing, saving to file).

7.  **LangGraph Orchestrator:**
    *   **Purpose:** Defines the workflow graph, managing the state and transitions between the components.
    *   **State:** Maintains the `Document` objects, question, relevant chunks, and the final answer as it flows through the pipeline.

## Data Flow

1.  **Input:** User provides source type, identifier, and question.
2.  **Routing:** `Input Handler` directs to the correct loader.
3.  **Loading:** `Abstract Loader` fetches and parses content into `Document`(s).
4.  **Chunking:** `Text Splitter` processes `Document`(s) into manageable chunks.
5.  **Vector Storage:** Chunks are embedded and stored in the vector database.
6.  **Relevant Chunk Search:** `Relevant_Chunk_Searcher` finds chunks most relevant to the question.
7.  **Question Answering:** `Answer_Question_Node` uses `LLM Manager` to answer the question based on relevant chunks.
8.  **Output:** `Output_Node` delivers the final answer.

## Diagram (Conceptual)

```
+-----------------+
|  Input Handler  |
+-----------------+
          |
          v
+------------------+       +------------------+       +----------------------+
| Abstract Loader  |------>|  Text Splitter   |------>| Vector Store Creator |
+------------------+       +------------------+       +----------------------+
                                    |                           |
                                    v                           v
+------------------+       +--------------------------+       +--------------------------+
|  LLM Manager     |<------| Relevant Chunk Searcher  |<------| Vector Store             |
+------------------+       +--------------------------+       +--------------------------+
          ^                            |
          |                            v
          |              +--------------------------+
          |              | Answer Question Node     |
          |              +--------------------------+
          |                            |
          |                            v
          |              +------------------+
          +------------->|   Output Node    |
                         +------------------+

                                |
                                v
                         +-------------+
                         |    User     |
                         +-------------+
```

## Technologies

*   **Python:** Primary programming language.
*   **LangGraph:** Core framework for defining and executing the workflow graph.
*   **LangChain:** Provides `Document` abstractions, text splitters, and LLM wrappers.
*   **Loaders:** `langchain_community.document_loaders` (Web, PDF, Text).
*   **Vector Database:** ChromaDB for similarity search.
*   **Embeddings:** HuggingFace embeddings for document encoding.
*   **OpenRouter API:** Interface for accessing various LLMs.
*   **uv:** Package manager for Python dependencies.