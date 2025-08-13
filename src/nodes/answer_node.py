"""
Answer node implementation for processing document chunks.

This implementation processes only the relevant chunks identified by
the vector database similarity search. Since we retrieve a limited number
of relevant chunks, we can answer the question directly without chunking.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.utils.llm_utils import initialize_llm


# Prompt template for answering questions based on relevant chunks
ANSWER_QUESTION_PROMPT = PromptTemplate.from_template("""
Use ONLY the following relevant context to answer the question. 
If the context does not contain enough information to answer the question, say "I don't have enough information in the provided document to answer this question."
Do not use any external knowledge or make assumptions beyond what is stated in the context.
Keep the answer concise and factual.

Relevant Context: {context}

Question: {question}

{answer_length_instruction}
""")


def answer_question_node(
    relevant_chunks: List[Document], 
    question: str,
    max_answer_length: int = None
) -> str:
    """
    Answer a question based on relevant document chunks from vector database.
    
    Args:
        relevant_chunks (List[Document]): List of relevant document chunks from vector search
        question (str): Question to answer based on the chunks
        max_answer_length (int, optional): Maximum number of sentences in the answer
        
    Returns:
        str: Answer to the question
    """
    # Initialize LLM
    llm = initialize_llm()
    
    # Combine all relevant chunks into a single context
    combined_context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Prepare the answer length instruction
    answer_length_instruction = ""
    if max_answer_length:
        answer_length_instruction = f"Answer in no more than {max_answer_length} sentences."
    
    # Format the prompt
    prompt = ANSWER_QUESTION_PROMPT.format(
        context=combined_context,
        question=question,
        answer_length_instruction=answer_length_instruction
    )
    
    # Get the answer from the LLM
    response = llm.invoke(prompt)
    return response.content