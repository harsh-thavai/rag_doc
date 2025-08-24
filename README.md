# 📚 Document QA Chatbot with RAG

A **Retrieval-Augmented Generation (RAG)** based chatbot that allows users to upload documents (PDF or TXT) and ask questions about their content. The system uses **semantic search** to retrieve relevant passages and generates accurate answers **based solely on the provided context**.

---

## 🚀 Features

- 📂 **Document Upload**: Supports PDF and TXT files  
- ✂️ **Intelligent Chunking**: Optimized for academic papers & speeches  
- 🔍 **Semantic Search**: FAISS-based vector similarity search  
- 🧠 **Context-Aware Answers**: LLM responses grounded in document context  
- 💬 **Interactive Chat Interface**: Built with Streamlit  
- 🔎 **Context Transparency**: Shows exact passages used for answers  

---

## 📋 Prerequisites

- Python **3.9+**  
- A **Groq API key** → [Get it here](https://console.groq.com/)  

---

## 🛠️ Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/harsh-thavai/rag_doc.git
    cd rag_doc
    ```

2. **Create a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**
   - Create a `.env` file in the root directory.
   - Add your Groq API key:

        ```env
        GROQ_API_KEY=your_api_key_here
        ```

---

## 🏃 Running the Application

1. **Start the Streamlit app**

    ```bash
    streamlit run app.py
    ```

2. **Open your browser**
   - The app runs at: `http://localhost:8501`

3. **Using the app**
   - Upload a PDF or TXT document in the sidebar.  
   - Click **"Process Document"** to analyze.  
   - Ask questions in the chat interface.  
   - See the context passages behind every answer.  

---

---

## 🧠 How It Works

1. **Document Ingestion**  
   - PDF → extracted via *PyPDF2*.  
   - TXT → direct reading.

2. **Text Processing**  
   - Recursive character splitter with overlap.  
   - Custom separators optimized for research & speeches.

3. **Vectorization**  
   - Embeddings via `all-mpnet-base-v2` model.  
   - Captures semantic meaning.

4. **Vector Storage**  
   - FAISS index for similarity search.  
   - Enables fast nearest-neighbor lookups.

5. **Query Processing**  
   - User query → embedding.  
   - Retrieve most relevant chunks.

6. **Answer Generation**  
   - Context + user query → crafted prompt.  
   - **Llama 3** generates grounded answers through **Groq** API.  
   - If no answer is in the context → the system acknowledges the limitation.

---

## 🛠️ Technologies Used

- **Streamlit** → UI framework  
- **LangChain** → Text splitting utilities  
- **Sentence Transformers** → Embeddings (`all-mpnet-base-v2`)  
- **FAISS** → Fast vector similarity search  
- **PyPDF2** → PDF text extraction  
- **Groq (Llama 3)** → LLM for answer generation  
- **Python** → Core language  

---

## 🔧 Customization

- **Chunk Size & Overlap** → `utils/text_splitter.py`  
- **Retrieval Parameters** → `app4.py` (tune the number of chunks to retrieve)  
- **LLM Settings** → `utils/llm_handler.py` (model, temperature, max tokens)  


