import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found. Please set it in your .env file.")
        st.stop()
    return Groq(api_key=api_key)

def generate_answer(question: str, context: list) -> str:
    """
    Generate answer using Groq API with context-optimized prompt
    """
    client = get_groq_client()
    
    # Prepare context
    context_str = "\n\n".join(context)
    
    # Optimized prompt that handles different document types
    prompt = f"""Based on the following context, please answer the question.
Only use information from the context to formulate your answer.
If the context doesn't contain the answer, say "I don't have enough information to answer this question."

Context:
{context_str}

Question: {question}

Answer:"""
    
    # Generate response using Groq
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Using Llama 3 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. For speeches, focus on main themes, key messages, and important points. For research papers, focus on findings, methods, and conclusions. Adapt your response style to match the document type."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300,
            top_p=0.9,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"