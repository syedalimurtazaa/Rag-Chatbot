"""
STEP 2: Document Loading + Chunking

Here we:

1. Load all `.txt` files from the `documents/` folder.
2. Split each document into smaller chunks to make searching easier.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load documents

loader = DirectoryLoader(
    "documents",
    glob="*.txt",
    loader_cls=TextLoader,
)

documents = loader.load()

print(f"Total documents loaded: {len(documents)}")

for doc in documents:
    print(f"  - {doc.metadata['source']} ({len(doc.page_content)} characters)")

# 2. Chunking: split each document into smaller pieces

# chunk_size = approximate number of characters in each chunk
# chunk_overlap = shared characters between consecutive chunks
#                 so that context is not lost

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

print(f"\nTotal chunks after splitting: {len(chunks)}")
print("\n--- Sample chunk ---")
print(chunks[0].page_content)
print("\nSource:", chunks[0].metadata["source"])