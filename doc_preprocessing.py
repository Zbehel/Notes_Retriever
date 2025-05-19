from pypdf import PdfReader
from docx import Document
from transformers import pipeline
import streamlit as st

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {file.name}: {e}")
            return ""
    elif file.name.endswith(".docx"):
        try:
            document = Document(file)
            for paragraph in document.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            st.error(f"Error reading DOCX {file.name}: {e}")
            return ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embeddings(texts):
    try:
        embedding_model = pipeline(
            "sentence-transformers/all-MiniLM-L6-v2"
        )  # Example model
        embeddings = embedding_model(texts)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

def process_files(files):
    all_chunks = []
    all_embeddings = []
    chunks_metadata = []

    for file in files:
        text = extract_text(file)
        if not text:  # Skip files that failed to process
            continue
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        if not embeddings: # Skip files that failed to embed
            continue

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        for i, chunk in enumerate(chunks):
            chunks_metadata.append({"file_name": file.name, "chunk_index": i})
    return all_chunks, all_embeddings, chunks_metadata