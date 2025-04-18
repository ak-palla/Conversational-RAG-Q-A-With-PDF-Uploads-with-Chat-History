
Conversational PDF Q&A with Chat History
========================================

An intelligent, conversational question-answering app that allows users to upload PDF documents and interact with their contents using natural language queries. Built with modern AI tools and a clean Streamlit interface, this project demonstrates the power of Retrieval-Augmented Generation (RAG) combined with chat memory for contextually rich responses.

Features
--------

- PDF Uploads – Ingest and process one or more PDF documents
- Conversational Q&A – Ask questions in natural language
- Chat History Memory – Maintains chat context using LangChain’s message history
- Intelligent Retrieval – Uses HuggingFace embeddings with ChromaDB for precise document search
- Powered by ChatGroq – Leverages Groq’s LLM for fast and intelligent responses

Tech Stack
----------

- Streamlit – Frontend for interactive UI
- LangChain – Framework for building LLM-powered applications
- Chroma – Vector database for document retrieval
- HuggingFace Embeddings – Semantic understanding of document content
- ChatGroq – Language model backend
- dotenv – Environment variable management

Installation
------------

1. Clone the repository:
    git clone https://github.com/ak-palla/Conversational-RAG-Q-A-With-PDF-Uploads-with-Chat-History.git
    cd your-repo-name

2. Create and activate a virtual environment:
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate

3. Install dependencies:
    pip install -r requirements.txt

4. Set up environment variables:
    Create a .env file in the root directory and add:
        HF_TOKEN=your_huggingface_token
        GROQ_API_KEY=your_groq_api_key

Running the App
---------------

    streamlit run test.py

Then open http://localhost:8501 in your browser.

Why This Project?
-----------------

This app showcases practical knowledge in LLMs, embeddings, and RAG pipelines. It's a demonstration of:
- Building user-friendly interfaces for AI systems
- Integrating multiple advanced ML components
- Developing production-ready tools for document-based AI chat


