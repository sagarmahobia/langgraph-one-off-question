"""
Shared utilities for the LangGraph Question Answering System.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables with override to ensure .env file takes precedence
load_dotenv(override=True)

def initialize_llm():
    """
    Initialize the LLM with OpenRouter configuration.
    
    Returns:
        ChatOpenAI: Initialized LLM instance
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.0,  # Low temperature for factual answers
    )