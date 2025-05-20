import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard
import re
import numpy as np
from doc_preprocessing import process_files, get_embeddings
from vector_DB import VectorDatabase  # Import the class
from llm_interaction import get_answer

# Initialize vector database (FAISS) - corrected instantiation
vector_database = VectorDatabase() #Instantiate the VectorDatabase Class
chunks_metadata = []

def main():
    st.title("Document Query App")

    uploaded_files = st.file_uploader(
        "Upload PDF or Word files", accept_multiple_files=True, type=["pdf", "docx"]
    )

    query = st.text_input("Enter your query:")

    if uploaded_files:
        global chunks_metadata
        all_chunks, all_embeddings, chunks_metadata = process_files(uploaded_files)
        vector_database.add_data(all_embeddings, all_chunks, chunks_metadata) # use the method

        st.session_state.files_processed = True
    
    if query:
        results = process_query(query)
        display_results(results)

def process_query(query):
    if vector_database.is_empty(): #Use the method
        return "Please upload files first."

    # query_embedding = get_embeddings([query])[0]
    # results = vector_database.query(query_embedding, k=3) # use the method
    query_embedding = get_embeddings([query])[0]  # Get the embedding for the query
    results = vector_database.query(query_embedding, k=10)  # Get the top 2 results

    return results

def normalize_line_breaks(text):
    text = text.replace("\\n", "  \n ")

    return text

def display_results(results):
    cpt = 1
    for result in (results):
        if result['score'] < 0.5:
            st.subheader(f"Réponse {cpt+1} :")
            st.write(f"Source File: {result['file_name']}, Chunk: {result['chunk_index']}, Score: {round((1-result['score'])*100,2)}%")
            st.subheader("Citations depuis le document :")
            st.write(normalize_line_breaks(result["chunk_text"]))
            st_copy_to_clipboard(normalize_line_breaks(result["chunk_text"]))
            cpt += 1


if __name__ == "__main__":
    main()