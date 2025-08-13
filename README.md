# LangGraph Question Answering System

This project implements a flexible question answering pipeline using LangGraph. It can answer questions based on content from various sources including web URLs, PDF files, text files, and direct text input. It leverages Large Language Models (LLMs) accessed through the OpenRouter API and uses HuggingFace embeddings with ChromaDB for efficient document retrieval.

**Key Feature**: The system provides answers strictly based on the content of the provided document. If the document doesn't contain information to answer a question, it will explicitly state that it doesn't have enough information, rather than providing external knowledge.

This project is a demonstration of "Vibe Coding" and showcases how AI tools can be effectively integrated into a developer's day-to-day workflow for planning, designing, and documenting software.

## Features

*   **Multi-source Input:** Answer questions based on content from web pages, PDFs, text files, or direct text strings.
*   **Document-based Answers:** Provides answers strictly based on the content of the provided document. If the document doesn't contain information to answer a question, it will explicitly state that it doesn't have enough information.
*   **LangGraph Pipeline:** Robust and configurable workflow management using LangGraph for state management and execution.
*   **Vector Database Integration:** Uses ChromaDB for efficient similarity search to find relevant document chunks.
*   **LLM Integration:** Uses LLMs via the OpenRouter API.
*   **Configurable LLM:** Easily switch LLMs by changing environment variables.
*   **Intelligent Chunking:** Automatically splits large documents into manageable chunks for processing.
*   **Efficient Processing:** Only processes relevant document chunks instead of all chunks.
*   **Consistent Output:** Uses low temperature settings (0.0) for factual, consistent answers.

## Prerequisites

*   Python 3.9 or higher
*   An OpenRouter API key (sign up at [https://openrouter.ai/](https://openrouter.ai/))
*   `uv` package manager (install from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv))

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd langgraph-question-answering
    ```

2.  **Create a virtual environment with `uv` (recommended):**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies using `uv`:**

    ```bash
    uv pip install -r requirements.txt
    ```

## Configuration

The application requires the following environment variables to be set. Create a `.env` file in the project root directory and add your configuration:

```env
# OpenRouter API Configuration
# Get your API key from https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional: Configure chunking behavior
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Replace `your_openrouter_api_key_here` with your actual OpenRouter API key.

The `EMBEDDING_MODEL` variable specifies the HuggingFace model to use for creating document embeddings. The default `all-MiniLM-L6-v2` is a lightweight, efficient model that works well for semantic similarity tasks.

The `LLM_MODEL` variable specifies which LLM to use through OpenRouter. By default, it uses `meta-llama/llama-3.1-8b-instruct:free`, but you can change this to any model available through OpenRouter. Some free alternatives include:
* `google/gemma-2-9b-it:free`
* `microsoft/phi-3-mini-128k-instruct:free`

If you encounter a "404 - No endpoints found" error, try changing the `LLM_MODEL` to one of the available models above.

## Usage

Once configured, you can run the question answering system with various input types:

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Ask a question about content from a web page
python src/main.py --url "https://example.com/article" --question "What is this article about?"

# Ask a question about content from a PDF file
python src/main.py --pdf "samples/drylab.pdf" --question "What is this document about?"

# Ask a question about content from another PDF file
python src/main.py --pdf "samples/business_ai.pdf" --question "How is AI used in business according to this document?"

# Ask a question about content from a text file
python src/main.py --textfile "samples/healthcare_ai.txt" --question "What are the benefits of AI in healthcare mentioned in this document?"

# Ask a question about direct text content
python src/main.py --text "Your text content here..." --question "What is this about?"

# Override chunking parameters
python src/main.py --text "Your text here..." --question "What is this about?" --chunk-size 500 --chunk-overlap 50

# Limit final answer length to 3 sentences
python src/main.py --url "https://example.com/article" --question "What are the key points?" --max-answer_length 3
```

## Sample Files

The repository includes sample files for testing:
- `samples/healthcare_ai.txt` - A text file about AI in healthcare
- `samples/business_ai.pdf` - A PDF file about AI in business
- `samples/drylab.pdf` - A PDF file about Drylab company announcements

## Project Structure

```
src/
├── main.py              # Entry point for the application
├── graph.py             # LangGraph workflow implementation
├── nodes/               # Processing nodes
│   ├── __init__.py      # Package init file
│   ├── answer_node.py    # Question answering node
│   ├── loaders.py        # Content loader implementation
│   ├── text_splitter.py  # Text splitting utility
│   └── vector_store.py   # Vector database utilities
├── utils/               # Utility functions
│   ├── __init__.py       # Package init file
│   └── llm_utils.py      # Shared LLM utilities
```

## Testing

A simple test script is included to verify the pipeline works correctly:

```bash
# Run the test (requires OPENROUTER_API_KEY to be set)
python test_pipeline.py
```

## How It Works

This system uses LangGraph to create a workflow graph with the following components:

1. **Input Handler**: Determines the type of input source and routes to the appropriate loader
2. **Abstract Loader**: Loads content from various sources (URL, PDF, text file, direct text)
3. **Text Splitter**: Segments large documents into smaller chunks
4. **Vector Store Creator**: Creates embeddings for document chunks and stores them in ChromaDB
5. **Relevant Chunk Searcher**: Searches the vector database for chunks relevant to the question
6. **Answer Question Node**: Processes only the relevant document chunks to answer the question directly
7. **Output Node**: Returns the final answer to the user

The LangGraph framework manages state and transitions between these components, providing a robust and configurable workflow. By using vector similarity search, the system only processes relevant chunks rather than all chunks, making it more efficient for large documents. The system combines the relevant chunks into a single context for the LLM, eliminating the need for separate chunk processing and answer combination steps.

Note: The `combine_node.py` is now legacy and kept only for reference. The current implementation directly answers questions based on relevant chunks without needing to combine multiple answers.

## Contributing

Contributions are welcome! Here's how you can contribute:

1.  **Fork the repository** on GitHub.
2.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    # or
    git checkout -b bugfix/your-bug-fix
    ```
3.  **Make your changes** and ensure they adhere to the project's style and conventions.
4.  **Add or update tests** if applicable.
5.  **Commit your changes:**
    ```bash
    git commit -m "Add a brief description of your changes"
    ```
6.  **Push to your fork:**
    ```bash
    git push origin feature/your-feature-name
    ```
7.  **Open a Pull Request** on the original repository.

Please ensure your code is well-documented and tested before submitting a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.