<h1 align="center">🤖 RAG Chatbot</h1>

<p align="center">
  Ask intelligent questions about your own documents using Retrieval-Augmented Generation.
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=900&color=0E75B6&center=true&vCenter=true&width=850&lines=Retrieval-Augmented+Generation+Chatbot;Chat+With+Your+Own+PDFs+and+Text+Files;LangChain+%7C+Chroma+%7C+Groq+%7C+Streamlit;Document-Grounded+AI+Answers" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-RAG%20Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Chroma-Vector%20Database-22C55E?style=for-the-badge" alt="Chroma" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20Interface-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

<br/>

## ✨ About the Project

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** that answers questions based on your own documents.

Upload PDF files or add text documents to the knowledge base, then ask questions through a clean Streamlit interface. The chatbot retrieves relevant document sections before generating an answer, helping keep responses grounded in your uploaded content.

<br/>

## 🚀 Features

- 📄 Ask questions about your own text files and PDFs
- 📤 Upload PDF documents directly through the web interface
- 🔎 Retrieves the most relevant document chunks before answering
- 🧠 Uses HuggingFace embeddings for semantic search
- 🗂️ Stores document embeddings locally in Chroma
- 💬 Generates natural answers with Groq LLM inference
- 📚 Shows source document(s) used for each answer
- 🎨 Clean and interactive Streamlit interface
- 💸 Built primarily with free and open-source tools

<br/>

## 🛠️ Tech Stack

<p align="left">
  <img src="https://skillicons.dev/icons?i=python" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" height="48" alt="LangChain" />
  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=flat-square&logo=huggingface&logoColor=black" height="48" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Chroma-Vector%20Database-22C55E?style=flat-square" height="48" alt="Chroma" />
  <img src="https://img.shields.io/badge/Groq-LLM%20Inference-F55036?style=flat-square" height="48" alt="Groq" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" height="48" alt="Streamlit" />
</p>

<br/>

## 📁 Project Structure

```text
rag_project/
│
├── documents/                    # Text documents for the knowledge base
├── app.py                         # Main Streamlit web application
├── step2_load_and_chunk.py        # Loads and splits documents into chunks
├── step3_create_vectorstore.py    # Creates embeddings and Chroma vector store
├── step4_rag_chatbot.py           # Terminal-based chatbot version
├── requirements.txt               # Python dependencies
└── README.md
```

<br/>

## ⚙️ How It Works

```text
PDF or Text Documents
        ↓
Load and Split Into Small Chunks
        ↓
HuggingFace Embeddings
        ↓
Chroma Vector Database
        ↓
User Question
        ↓
Retrieve Relevant Document Chunks
        ↓
Groq LLM Receives Question + Context
        ↓
Grounded Answer + Source Documents
```

### RAG Pipeline

1. Documents are loaded from the `documents/` folder or uploaded through the Streamlit interface.
2. Each document is split into small, manageable text chunks.
3. HuggingFace's `all-MiniLM-L6-v2` model converts every chunk into an embedding.
4. The embeddings are stored in a local Chroma vector database.
5. When the user asks a question, Chroma finds the most relevant chunks.
6. The retrieved context is sent to Groq along with the question.
7. The LLM generates an answer based on the provided document context.
8. The chatbot displays the source document(s) used.

<br/>

## ▶️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rag_project
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Groq API Key

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

<br/>

## 🏃 Run the Project

### Build the Vector Database

First, place your `.txt` files in the `documents/` folder. Then run:

```bash
python step2_load_and_chunk.py
python step3_create_vectorstore.py
```

### Run the Streamlit Chatbot

```bash
streamlit run app.py
```

The app will open in your browser at:

```text
http://localhost:8501
```

### Run the Terminal Version

```bash
python step4_rag_chatbot.py
```

<br/>

## 💬 Example Questions

```text
What is the main topic of this document?

Summarize the key points from the uploaded PDF.

What does the document say about machine learning?

Which source document was used for this answer?
```

<br/>

## 🔐 Privacy and Security Notes

- Your document embeddings are stored locally in the Chroma vector database.
- Never upload your `.env` file or API key to GitHub.
- Add `.env` to your `.gitignore` file:

```text
.env
venv/
__pycache__/
chroma_db/
```

<br/>

## 📚 Learning Concepts

- Retrieval-Augmented Generation (RAG)
- Large Language Models
- Semantic search
- Vector embeddings
- Vector databases
- Document chunking
- Prompt context injection
- LangChain orchestration
- Streamlit interface development
- API key management

<br/>
