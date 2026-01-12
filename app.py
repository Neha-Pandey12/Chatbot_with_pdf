# app.py
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
import tempfile

# -----------------------
# Setup
# -----------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Missing GOOGLE_API_KEY in .env")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=1024,
)

#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -----------------------
# Sidebar: Upload PDF
# -----------------------
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Reading PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(pages)

            st.session_state.vectorstore = Chroma.from_documents(splits, embeddings)
            st.success("✅ Ready! Ask questions about your PDF.")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------
# Main Chat
# -----------------------
st.title("💬 PDF Q&A Chatbot")
st.caption("Upload a PDF and ask questions!")

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.write(msg.content)

# User input
if prompt := st.chat_input("Ask something about your PDF..."):
    # Save user message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Get response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            if st.session_state.vectorstore:
                # Retrieve relevant text
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(prompt)
                context = "\n\n".join(doc.page_content for doc in docs)

                # Simple prompt with context
                prompt_text = f"""
                You are a helpful assistant. Answer the question using only the context below.
                If you don't know, say "I don't know based on the document."

                Context:
                {context}

                Question: {prompt}
                Answer:
                """
                response = llm.invoke(prompt_text).content
            else:
                # No PDF: basic chat
                response = llm.invoke(prompt).content

        st.write(response)
        st.session_state.messages.append(AIMessage(content=response))