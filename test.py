## Conversational RAG Q&A With PDF Uploads and Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit UI
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

# Proceed if Groq API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Session ID input
    session_id = st.text_input("Session ID", value="default_session")

    # Chat history state
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Upload PDFs
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    vectorstore_path = "./chroma_store"
    documents = []

    if uploaded_files:
        # Load and split documents
        for i, uploaded_file in enumerate(uploaded_files):
            temp_path = f"./temp_{i}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Initialize or load vectorstore
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=vectorstore_path
        )

        # Optional: Check if store is empty to prevent duplicates
        if len(vectorstore.get()["ids"]) == 0:
            vectorstore.add_documents(splits)
            vectorstore.persist()
    else:
        # Load existing vectorstore if no new uploads
        try:
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path
            )
        except Exception as e:
            st.error("Failed to load vectorstore. Upload a PDF to initialize.")
            st.stop()

    retriever = vectorstore.as_retriever()

    # History-aware prompt for question reformulation
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Manage session-specific history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User question input
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        st.markdown(f"**Assistant:** {response['answer']}")

        st.markdown("### Chat History:")
        for msg in session_history.messages:
            role = "User" if msg.type == "human" else "Assistant"
            st.markdown(f"**{role}:** {msg.content}")

else:
    st.warning("Please enter the Groq API Key")
