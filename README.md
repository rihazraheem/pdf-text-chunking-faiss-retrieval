# pdf-text-chunking-faiss-retrieval
Efficiently chunk, embed, and retrieve text from PDF documents using FAISS for fast semantic search. This project extracts text from PDFs, breaks it into manageable chunks, embeds those chunks with SentenceTransformer, and indexes them in FAISS for quick similarity-based retrieval.

## Overview

This Python project processes PDF documents to extract, chunk, and embed the text for efficient retrieval. Using the `SentenceTransformer` model for embeddings and `FAISS` for fast similarity search, you can ask questions about the document and retrieve relevant sections of text based on semantic similarity.

The project works in three main stages:
1. **Extract Text from PDF:** Extracts text from each page of the PDF document.
2. **Chunking:** Breaks the extracted text into smaller, manageable chunks with some overlap for better retrieval accuracy.
3. **Embedding and Indexing:** Uses the `SentenceTransformer` model to generate embeddings for each chunk and stores them in a FAISS index for fast similarity search.
4. **Query Retrieval:** Allows querying with a natural language question to retrieve the most relevant text chunks based on semantic similarity.

## Requirements

- `pypdf` (for PDF text extraction)
- `tiktoken` (for tokenization and chunking)
- `sentence-transformers` (for embeddings)
- `faiss` (for similarity search)
- `numpy` (for handling array operations)
