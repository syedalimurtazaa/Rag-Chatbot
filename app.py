"""
Streamlit Web UI for the RAG Chatbot (with PDF upload)
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load GROQ_API_KEY from .env
load_dotenv()

# ---------- Page setup ----------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS for a cleaner look ----------
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .answer-box {
        background-color: #1e2530;
        border-left: 4px solid #4CAF50;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    .source-chip {
        display: inline-block;
        background-color: #2d3748;
        color: #a0aec0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the embeddings model so it only loads once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Load the existing vector store (not cached, since we will keep adding to it)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Set up the LLM (cached so it only loads once)
@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

llm = load_llm()

# Prompt template that tells the LLM how to answer using retrieved context
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {input}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------- Header ----------
st.markdown("<h1 style='text-align: center;'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Ask questions about your documents, powered by Groq + LangChain</p>", unsafe_allow_html=True)
st.write("")

# ---------- Tabs for cleaner organization ----------
tab1, tab2 = st.tabs(["💬 Ask a Question", "📄 Add a Document"])

# ---------- TAB 1: Question Answering ----------
with tab1:
    st.write("")
    query = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if query and ask_clicked:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        qa_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(query)
            sources = retriever.invoke(query)

        st.write("")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        st.write("")
        st.caption("Source")
        if sources:
            top_source = sources[0].metadata['source']
            st.markdown(f'<span class="source-chip">📄 {top_source}</span>', unsafe_allow_html=True)


# ---------- TAB 2: PDF Upload ----------
with tab2:
    st.write("")
    st.write("Upload a PDF to expand the chatbot's knowledge base.")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("➕ Add this PDF", type="primary"):
            with st.spinner("Reading and indexing PDF..."):
                # Save the uploaded file temporarily (PyPDFLoader needs a file path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Load the PDF
                loader = PyPDFLoader(tmp_path)
                pdf_docs = loader.load()

                # Split into chunks (same settings as the other documents)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(pdf_docs)

                # Fix the source metadata to show the original filename, not the temp path
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name

                # Clean up the temp file
                os.remove(tmp_path)

                # Check if any text was actually extracted before adding to the database
                if len(chunks) == 0:
                    st.error(f"No readable text found in '{uploaded_file.name}'. This usually happens with scanned or image-based PDFs. Try a text-based PDF instead.")
                else:
                    vectorstore.add_documents(chunks)
                    st.success(f"✅ '{uploaded_file.name}' added! ({len(chunks)} chunks indexed)")
