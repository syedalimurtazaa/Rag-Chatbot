"""
STEP 3: Embeddings + Vector Database

Here we:

1. Repeat the document loading and chunking process from Step 2.
2. Convert each chunk into an embedding (a numerical vector).
3. Store the embeddings in a Chroma vector database saved on disk.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load documents (same as Step 2)

loader = DirectoryLoader(
    "documents",
    glob="*.txt",
    loader_cls=TextLoader,
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks to embed: {len(chunks)}")

# 2. Load the embedding model

# This is a free local model, so no API key or cost is required.
# The first run downloads the model (~90 MB); it will be cached afterwards.

print("\nLoading embedding model (the first run may take a little time)...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Create the Chroma vector database and store the chunks

print("Creating embeddings and storing chunks in the Chroma database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",  # Saved to disk in this folder
)

print("\nDone! The vector database has been created and saved in 'chroma_db'.")

# 4. Quick test: run a sample search

print("\n--- Test Search ---")

query = "What is chunking in RAG?"
results = vectorstore.similarity_search(query, k=2)

print(f"Query: {query}\n")

for i, result in enumerate(results, 1):
    print(f"Result {i} (from {result.metadata['source']}):")
    print(result.page_content)
    print()