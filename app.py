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

# Load GROQ_API_KEY from .env (locally) or from Streamlit secrets (on cloud)
load_dotenv()

# Cache the embeddings model so it only loads once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Load the existing vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Set up the LLM (cached)
@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

llm = load_llm()

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {input}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Page setup
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")

# --- PDF Upload Section ---
st.subheader("Add a new document")
st.caption("Note: On the live cloud demo, uploads are temporary and may not persist. For permanent uploads, run this app locally.")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Add this PDF to the chatbot's knowledge"):
        with st.spinner("Reading and indexing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                pdf_docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(pdf_docs)

                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name

                vectorstore.add_documents(chunks)
                os.remove(tmp_path)

                st.success(f"'{uploaded_file.name}' has been added! You can now ask questions about it.")
            except Exception as e:
                st.error("Sorry, adding documents isn't supported on this live demo (cloud storage is temporary). This feature works when running the app locally.")

st.divider()

# --- Question Answering Section ---
st.subheader("Ask a question")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.invoke(query)
        sources = retriever.invoke(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    seen = set()
    for doc in sources:
        source = doc.metadata['source']
        if source not in seen:
            st.write(f"- {source}")
            seen.add(source)
