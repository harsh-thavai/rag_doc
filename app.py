import streamlit as st
import os
from dotenv import load_dotenv
from utils.document_loader import load_document
from utils.text_splitter import split_document
from utils.embedding_handler import generate_embeddings
from utils.vector_store import create_vector_store, search_similar
from utils.llm_handler import generate_answer

# Load environment variables
load_dotenv()

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Configure page
st.set_page_config(page_title="Document QA Chatbot", layout="wide")

# Main header
st.title("📄 Document QA Chatbot")
st.caption("Upload a document and chat about its content")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key status
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("✅ Groq API key loaded")
    else:
        st.error("❌ Groq API key not found in .env file")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT file",
        type=["pdf", "txt"],
        disabled=not api_key
    )
    
    if uploaded_file and api_key:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    document_text = load_document(uploaded_file)
                    chunks = split_document(document_text)
                    embeddings = generate_embeddings(chunks)
                    vector_store = create_vector_store(embeddings)
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.document_chunks = chunks
                    st.session_state.processing_complete = True
                    st.session_state.messages = []
                    
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main content
if st.session_state.processing_complete:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Only show context for assistant messages that have context
            if message["role"] == "assistant" and message.get("context"):
                with st.expander("View Context"):
                    for i, chunk in enumerate(message["context"], 1):
                        st.write(f"**Chunk {i}:** {chunk}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # For summary questions, retrieve more chunks
                    if any(word in prompt.lower() for word in ["summarize", "summary", "overview"]):
                        k = 8  # More chunks for summary
                    else:
                        k = 5  # Default chunks for specific questions
                    
                    # Retrieve relevant context
                    context_indices = search_similar(
                        st.session_state.vector_store,
                        prompt,
                        k=k
                    )
                    context = [st.session_state.document_chunks[i] for i in context_indices]
                    
                    # Generate answer
                    answer = generate_answer(prompt, context)
                    
                    # Display answer immediately
                    st.write(answer)
                    
                    # Display context immediately
                    with st.expander("View Context"):
                        for i, chunk in enumerate(context, 1):
                            st.write(f"**Chunk {i}:** {chunk}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                        "context": []
                    })
else:
    st.info("Please upload and process a document to start chatting")
