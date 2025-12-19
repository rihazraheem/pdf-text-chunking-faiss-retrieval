from pypdf import PdfReader

def read_pdf_with_pages(file_path):
    reader = PdfReader(file_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({
                "page": page_num,
                "text": text
            })
    return pages

pages = read_pdf_with_pages("notes.pdf")

"""Token Page Chunking each page"""

import tiktoken

def token_chunk_pages(pages, chunk_size=300, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    all_chunks = []
    chunk_id = 0

    for page in pages:
        tokens=enc.encode(page["text"])
        i = 0

        while i < len(tokens):
            chunk_text = enc.decode(tokens[i:i+chunk_size])
            if not chunk_text.strip():
                break
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": page["page"],
                "source": "notes.pdf"
            })
            chunk_id += 1
            i += chunk_size - overlap

    return all_chunks

chunks = token_chunk_pages(pages)
print(chunks[0])

"""Embed Only the Chunk Text"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts).astype('float32')

"""Store Embeddings in FAISS"""

import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

"""Ask a question"""

query="what is A/D Converter"

"""Retrieve with Metadata"""

query_embedding = model.encode([query])

D, I = index.search(query_embedding, k=3)

for rank, idx in enumerate(I[0], 1):
    chunk = chunks[idx]
    print(f"\n--- Result {rank} ---")
    print(f"Source : {chunk['source']}")
    print(f"Page   : {chunk['page']}")
    print(f"Chunk  : {chunk['chunk_id']}")
    print(f"Text   : {chunk['text'][:300]}")
