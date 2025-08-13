# LangGraph Pipeline Implementation for Question Answering

## Objective
Create a flexible LangGraph pipeline capable of answering questions based on content from various sources (web URL, PDF, text file, direct text query) using an LLM accessed via the OpenRouter API. The LLM model, API key, and base URL are configured via environment variables.

## Pipeline Stages

1.  **Abstract Content Loading:**
    *   Input: Content source type (URL, PDF path, Text file path, Direct Text) and the corresponding identifier (e.g., the actual URL or file path).
    *   Action: Route to a specific loader based on the type:
        *   **URL:** Use `WebLoader` (e.g., from `langchain_community.document_loaders`) to fetch and parse the web page.
        *   **PDF File:** Use `PyPDFLoader` or similar to extract text from the PDF.
        *   **Text File:** Use `TextLoader` to read the file's content.
        *   **Direct Text:** Directly wrap the text into a `Document` object.
    *   Output: A standardized `Document` object (or list of `Document` objects) containing the extracted text and relevant metadata, regardless of the source type.

2.  **Chunk Text:**
    *   Input: `Document` object(s) from the abstract loader.
    *   Action: Use a text splitter (e.g., from `langchain.text_splitter`) to divide potentially large text content into smaller, manageable chunks suitable for the LLM's context window. Overlapping chunks can improve coherence.
    *   Output: List of `Document` objects representing the text chunks.

3.  **Create Vector Store:**
    *   Input: List of `Document` chunks from the text splitter.
    *   Action: 
        1. Create embeddings for each document chunk using HuggingFace embeddings.
        2. Store the embeddings and document content in a ChromaDB vector database.
    *   Output: Initialized vector store containing all document chunks.

4.  **Search Relevant Chunks:**
    *   Input: Vector store and the user's question.
    *   Action: 
        1. Create an embedding for the question.
        2. Search the vector database for the most similar document chunks (e.g., top 4).
    *   Output: List of relevant `Document` chunks.

5.  **Answer Question:**
    *   Input: List of relevant `Document` chunks and the user's question.
    *   Action:
        1.  Initialize an LLM wrapper using LangChain's OpenAI-compatible interface.
        2.  Configure the LLM wrapper using environment variables:
            *   `OPENROUTER_API_KEY`: The API key for OpenRouter.
            *   `OPENROUTER_BASE_URL`: The base URL for the OpenRouter API (e.g., `https://openrouter.ai/api/v1`).
            *   `LLM_MODEL`: The specific model identifier to use (e.g., `meta-llama/llama-3.1-8b-instruct:free`).
        3.  Combine all relevant chunks into a single context.
        4.  Call the configured LLM with a prompt instructing it to answer the provided question based on the provided context.
    *   Output: Final answer (string).

6.  **Output Answer:**
    *   Input: Final answer string.
    *   Action: Return the answer to the user or pass it to the next component in a larger system.

## Tools & Libraries

*   **LangGraph:** Core framework for building the state machine/graph.
*   **Document Loaders:** `WebLoader`, `PyPDFLoader`, `TextLoader` from `langchain_community.document_loaders`.
*   **Text Splitting:** Utilities from `langchain.text_splitter`.
*   **Vector Database:** ChromaDB for efficient similarity search.
*   **Embeddings:** HuggingFace embeddings for document encoding.
*   **LLM Integration:** LangChain's `ChatOpenAI` or equivalent OpenAI-compatible LLM wrapper.
*   **Environment Variables:** For `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `LLM_MODEL`.
*   **LangChain:** For LLM integration, chains, and utilities.
*   **Python:** Core language for implementation.
*   **uv:** Package manager for Python dependencies.

## Model Configuration

The system uses OpenRouter to access various LLMs. By default, it uses the `meta-llama/llama-3.1-8b-instruct:free` model, but this can be changed by setting the `LLM_MODEL` environment variable in your `.env` file.

Some free models available through OpenRouter include:
* `meta-llama/llama-3.1-8b-instruct:free` (default)
* `google/gemma-2-9b-it:free`
* `microsoft/phi-3-mini-128k-instruct:free`

If you encounter a "404 - No endpoints found" error, try changing the `LLM_MODEL` in your `.env` file to one of the available models above.

## Key Improvements from Original Plan

*   **Vector Database Integration:** Instead of processing all chunks, we use a vector database to find only the most relevant chunks for answering the question.
*   **Simplified Answering Process:** Instead of answering each chunk separately and then combining answers, we combine relevant chunks into a single context and answer the question directly.
*   **Efficiency:** This approach is more efficient for large documents as it only processes relevant content.

## Testing with Different Input Types

The system has been tested with all supported input types:

1.  **Direct Text:**
    ```bash
    python src/main.py --text "The LangGraph Question Answering System is a powerful tool that uses AI to answer questions based on provided documents. It supports multiple input formats including web URLs, PDF files, text files, and direct text input." --question "What input formats does the LangGraph Question Answering System support?"
    ```

2.  **PDF Files:**
    ```bash
    python src/main.py --pdf "samples/drylab.pdf" --question "What is this document about?"
    python src/main.py --pdf "samples/business_ai.pdf" --question "How is AI used in business according to this document?"
    ```

3.  **Text Files:**
    ```bash
    python src/main.py --textfile "samples/healthcare_ai.txt" --question "What are the benefits of AI in healthcare mentioned in this document?"
    ```

## Considerations

*   **Error Handling:** Handle errors from different loaders (network, file I/O), parsing errors, LLM call failures (including API quota issues), and missing environment variables gracefully.
*   **Async/Await:** Consider using asynchronous operations for LLM calls and potentially loaders if supported, to improve performance.
*   **State Management:** Use LangGraph's state management to pass `Document` objects, question, and answers between nodes.
*   **Prompt Engineering:** Carefully design prompts for the LLM to ensure good quality answers.
*   **Configurability:** Allow configuration for chunk size, overlap, and answer length via parameters or environment variables, while keeping the core LLM configuration in env vars.