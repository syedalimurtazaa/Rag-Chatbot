"""
STEP 4: Complete RAG Chatbot
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load GROQ_API_KEY from .env
load_dotenv()

# 2. Load the same embedding model used while creating the vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Load the existing Chroma vector database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Retriever - fetches the top 3 most relevant chunks for a query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Set up the Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
)

# 5. Prompt template - tells the LLM how to use retrieved context
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {input}
""")

# Helper function - joins the retrieved chunks into one text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Build the RAG chain using LCEL (pipe operator)
qa_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Interactive loop
print("=" * 50)
print("RAG Chatbot Ready! (ask a question, type 'exit' to quit)")
print("=" * 50)

while True:
    query = input("\nYour question: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Closing the chatbot. Goodbye!")
        break

    # Get the answer from the chain
    answer = qa_chain.invoke(query)

    print("\nAnswer:")
    print(answer)

    # Separately fetch source docs for display (retriever called again)
    sources = retriever.invoke(query)
    print("\nSources:")
    for doc in sources:
        print(f"  - {doc.metadata['source']}")