# NoteBookBot
NoteBookBot | LLM-Based PDF Question-Answering Assistant

NoteBookBot is an AI-powered document question-answering application that allows users to upload multi-page PDF documents such as textbooks, manuals, and study materials and interact with them using natural-language questions. The application combines NLP, semantic search, vector embeddings, and Large Language Models (LLMs) to retrieve relevant information from uploaded documents and generate context-aware answers.

How It Works
PDF Upload & Text Extraction – Users upload PDF documents, which are processed using PyPDF2 to extract their textual content.
Document Chunking – Extracted text is divided into smaller, manageable chunks using LangChain's RecursiveCharacterTextSplitter, making the content suitable for LLM processing.
Vector Embeddings & Storage – Each text chunk is converted into high-dimensional OpenAI embeddings and stored in a FAISS vector database for efficient semantic similarity search.
Question Answering – When a user submits a question, relevant document chunks are retrieved from FAISS and passed as context to GPT-3.5 Turbo through a LangChain QA chain to generate a relevant and context-aware response.
Key Features
📄 Upload and process multi-page PDF documents
🔍 Semantic search using FAISS vector similarity
🧠 Context-aware question answering using GPT-3.5 Turbo
✂️ Efficient document chunking with LangChain
🔗 Retrieval-Augmented Generation (RAG) workflow
🖥️ Interactive web interface built with Streamlit

Tech Stack
Python · Streamlit · LangChain · OpenAI GPT-3.5 Turbo · OpenAI Embeddings · FAISS · PyPDF2 · NLP · Vector Search
