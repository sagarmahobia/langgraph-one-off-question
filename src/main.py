#!/usr/bin/env python3
"""
Main entry point for the LangGraph Question Answering System.
Handles command-line arguments and executes the question answering pipeline.
"""

import argparse
import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv(override=True)

from src.graph import answer_question_with_graph


def main():
    """Main function to parse arguments and run the question answering pipeline."""
    parser = argparse.ArgumentParser(
        description="Answer questions based on content from various sources using LLMs via OpenRouter API."
    )

    # Input source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--url', type=str, help='URL of the web page to analyze')
    source_group.add_argument('--pdf', type=str, help='Path to the PDF file to analyze')
    source_group.add_argument('--textfile', type=str, help='Path to the text file to analyze')
    source_group.add_argument('--text', type=str, help='Direct text content to analyze')

    # Question argument
    parser.add_argument('--question', type=str, required=True, help='Question to answer based on the content')

    # Optional chunking parameters
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Chunk size for text splitting (default: from CHUNK_SIZE env var or 500)')
    parser.add_argument('--chunk-overlap', type=int, default=None,
                        help='Chunk overlap for text splitting (default: from CHUNK_OVERLAP env var or 50)')

    # Optional answer length limit
    parser.add_argument('--max-answer_length', type=int, default=None,
                        help='Maximum number of sentences in the final answer')

    args = parser.parse_args()

    # Validate input source
    input_type = None
    content = None

    if args.url:
        input_type = "url"
        content = args.url
    elif args.pdf:
        input_type = "pdf"
        content = args.pdf
        # Validate PDF file exists
        if not os.path.exists(content):
            print(f"Error: PDF file '{content}' not found")
            sys.exit(1)
    elif args.textfile:
        input_type = "textfile"
        content = args.textfile
        # Validate text file exists
        if not os.path.exists(content):
            print(f"Error: Text file '{content}' not found")
            sys.exit(1)
    elif args.text:
        input_type = "text"
        content = args.text

    # Get chunking parameters from environment variables if not provided
    chunk_size = args.chunk_size or int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = args.chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 50))

    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Please set it in your .env file")
        sys.exit(1)

    try:
        # Run the question answering pipeline with LangGraph
        answer = answer_question_with_graph(
            input_type=input_type,
            content=content,
            question=args.question,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_answer_length=args.max_answer_length
        )

        # Output the answer
        print("\nQuestion:", args.question)
        print("\nAnswer:")
        print(answer)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
