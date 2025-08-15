#!/usr/bin/env python3
"""
Streamlit UI for the LangGraph Question Answering System.
"""

import streamlit as st
import os
import sys
import asyncio
import tempfile
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

# Validate environment variables
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
model = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Configure Streamlit page
st.set_page_config(
    page_title="LangGraph Question Answering System",
    page_icon="❓",
    layout="wide"
)

# Health check endpoint
if st.query_params.get("health") == "true":
    st.write("OK")
    st.stop()

# Display app title and description
st.title("❓ LangGraph Question Answering System")
st.markdown(
    "Answer questions based on content from various sources using LangGraph and LLMs via the OpenRouter API. "
    "Select a content type, provide the content and your question, and click \"Answer Question\" to get an answer. "
    "Note: File uploads are limited to 1MB."
)

# Check for API key
if not api_key:
    st.error("❌ OPENROUTER_API_KEY environment variable is required. Please set it in your .env file or environment.")
    st.stop()

# Create sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Chunk size configuration
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=int(os.getenv("CHUNK_SIZE", "500")),
        help="Size of text chunks for processing"
    )
    
    # Chunk overlap configuration
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=int(os.getenv("CHUNK_OVERLAP", "50")),
        help="Overlap between chunks"
    )
    
    # Max answer length
    max_answer_length = st.number_input(
        "Max Answer Length (sentences)",
        min_value=1,
        max_value=20,
        value=10,
        help="Maximum number of sentences in the final answer (optional)"
    )
    
    # Display current model info
    st.subheader("Model Info")
    st.info(f"Model: {model}")
    st.info(f"Base URL: {base_url}")

# Main content area
content_type = st.selectbox(
    "Select Content Type",
    options=["url", "pdf", "textfile", "text"],
    format_func=lambda x: {
        "url": "🌐 Web URL",
        "pdf": "📄 PDF File",
        "textfile": "📝 Text File",
        "text": "💬 Direct Text"
    }[x]
)

content = None
temp_file_path = None

# Display appropriate input field based on content type
if content_type == "url":
    content = st.text_input("Enter URL", placeholder="https://example.com/article")
elif content_type == "pdf":
    uploaded_file = st.file_uploader("Upload PDF file (Max 1MB)", type="pdf")
    if uploaded_file:
        # Check file size
        if uploaded_file.size > 1024 * 1024:  # 1MB in bytes
            st.error("❌ File size exceeds 1MB limit. Please upload a smaller file.")
            st.stop()
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        content = temp_file_path
elif content_type == "textfile":
    uploaded_file = st.file_uploader("Upload Text file (Max 1MB)", type=["txt", "md"])
    if uploaded_file:
        # Check file size
        if uploaded_file.size > 1024 * 1024:  # 1MB in bytes
            st.error("❌ File size exceeds 1MB limit. Please upload a smaller file.")
            st.stop()
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        content = temp_file_path
elif content_type == "text":
    content = st.text_area("Enter text", height=200, placeholder="Paste your text here...")

# Question input
question = st.text_input("Enter your question", placeholder="What is this document about?")

# Answer button
if st.button("❓ Answer Question", type="primary", use_container_width=True):
    # Validate input
    if not content:
        st.warning("Please provide content to analyze")
    elif not question:
        st.warning("Please enter a question")
    else:
        # Show processing status
        with st.spinner("Processing your content and answering your question..."):
            try:
                # Import the question answering function
                from src.graph import answer_question_with_graph
                
                # Run the async function (if needed)
                answer = answer_question_with_graph(
                    input_type=content_type,
                    content=content,
                    question=question,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_answer_length=max_answer_length if max_answer_length > 0 else None
                )
                
                # Display the result
                st.success("✅ Answer generated successfully!")
                st.subheader("Answer")
                st.markdown(answer)
                
            except Exception as e:
                st.error(f"❌ Error during question answering: {str(e)}")
                
                # Show additional debugging info for common issues
                if "401" in str(e):
                    st.info(" This error might be due to an invalid API key. Please check your OPENROUTER_API_KEY.")
                elif "404" in str(e):
                    st.info(" This error might be due to an invalid model. Please check your LLM_MODEL setting.")
            finally:
                # Cleanup temporary files
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        temp_file_path = None
                    except:
                        pass

# Display sample usage
st.divider()
st.subheader("Example Usage")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Try these sample files:**")
    st.markdown("- `samples/healthcare_ai.txt`")
    st.markdown("- `samples/drylab.pdf`")
    st.markdown("- `samples/business_ai.pdf`")

with col2:
    st.markdown("**Try with a URL:**")
    st.markdown("- `https://en.wikipedia.org/wiki/Artificial_intelligence`")
    
st.markdown("**Sample questions:**")
st.markdown("- \"What is this document about?\"")
st.markdown("- \"What are the key points discussed?\"")
st.markdown("- \"How is AI used according to this document?\"")