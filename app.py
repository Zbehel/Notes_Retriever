import streamlit as st
from doc_preprocessing import process_files
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

    query_embedding = get_embeddings([query])[0]
    results = vector_database.query(query_embedding, k=3) # use the method
    return results

def display_results(results):
    for result in results:
        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Quote from Document")
        st.write(result["chunk_text"])
        st.subheader("Source")
        st.write(f"File: {result['file_name']}, Chunk: {result['chunk_index']}")

if __name__ == "__main__":
    main()