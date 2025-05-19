import faiss
import numpy as np
import streamlit as st

class VectorDatabase:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.chunks_metadata = []

    def add_data(self, embeddings, chunks, chunks_metadata):
        if not embeddings:
            st.error("No embeddings to add to the database.")
            return
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        self.chunks = chunks
        self.chunks_metadata = chunks_metadata

    def query(self, query_embedding, k=3):
        if self.index is None:
            st.error("Vector database is empty. Please upload files and process them first.")
            return []
        _, indices = self.index.search(np.array([query_embedding]), k=k)
        results = []
        for i in indices[0]:
            chunk_text = self.chunks[i]
            metadata = self.chunks_metadata[i]
            answer = get_answer(query, chunk_text)  # Corrected call
            results.append({
                "answer": answer,
                "chunk_text": chunk_text,
                "file_name": metadata["file_name"],
                "chunk_index": metadata["chunk_index"],
            })
        return results

    def is_empty(self):
        return self.index is None
from llm_interaction import get_answer