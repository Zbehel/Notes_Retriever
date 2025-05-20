import faiss
import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Union

from doc_preprocessing import process_files 

class VectorDatabase:
    """
    A class to manage a vector database using FAISS for efficient similarity search.
    """
    def __init__(self, dimension: int = 0):
        """
        Initializes the VectorDatabase.

        Args:
            dimension (int, optional): The dimension of the embeddings. If None, the
                index is not initialized until data is added. Defaults to None.
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.chunks_metadata: List[Dict] = []

    def add_data(self, embeddings: List[np.ndarray], chunks: List[str], chunks_metadata: List[Dict]):
        """
        Adds embeddings, text chunks, and metadata to the vector database.

        Args:
            embeddings (List[List[float]]): A list of embeddings (each a list or numpy array).
            chunks (List[str]): A list of corresponding text chunks.
            chunks_metadata (List[Dict]): A list of metadata dictionaries, one for each chunk.
        """
        if not embeddings:
            st.error("No embeddings to add to the database.")
            return

        # Ensure embeddings are numpy arrays
        embeddings = [np.array(emb) for emb in embeddings]

        if self.dimension == 0:
            self.dimension = embeddings[0].shape[0]
            self.index = faiss.IndexFlatL2(self.dimension)  # Use L2 distance
        elif self.dimension != embeddings[0].shape[0]:
            st.error(f"Embedding dimension ({embeddings[0].shape[0]}) does not match database dimension ({self.dimension}).")
            return

        # Convert embeddings to a float32 numpy array for FAISS
        embeddings_np = np.array(embeddings, dtype=np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        self.chunks = chunks
        self.chunks_metadata = chunks_metadata

    def query(self, query_embedding: Union[List[float], np.ndarray], k: int = 3) -> List[Dict]:
        """
        Queries the vector database for the most similar chunks to a query embedding.

        Args:
            query_embedding (List[float] or np.ndarray): The embedding of the query.
            k (int, optional): The number of nearest neighbors to retrieve. Defaults to 3.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary contains:
                - "chunk_text" (str): The text of the retrieved chunk.
                - "file_name" (str): The name of the file the chunk came from.
                - "chunk_index" (int): The index of the chunk in the file.
        """
        if self.index is None:
            st.error("Vector database is empty. Please upload files and process them first.")
            return []

        # Ensure query_embedding is a numpy array
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)  # Reshape for FAISS

        dist, indices = self.index.search(query_embedding, k=k)
        results = []
        for (i, j) in zip(indices[0], dist[0]):
            chunk_text = self.chunks[i]
            metadata = self.chunks_metadata[i]
            results.append({
                "chunk_text": chunk_text,
                "file_name": metadata["file_name"],
                "chunk_index": metadata["chunk_index"],
                "score": j
            })
        return results

    def is_empty(self) -> bool:
        """
        Checks if the vector database is empty.

        Returns:
            bool: True if the database is empty, False otherwise.
        """
        return self.index is None

if __name__ == "__main__":
    # This part is for testing the VectorDatabase class.
    #  It will only run if you execute this file directly: python vector_database.py

    # # Create some dummy data
    # embeddings = [
    #     np.array([1.0, 2.0, 3.0]),
    #     np.array([4.0, 5.0, 6.0]),
    #     np.array([7.0, 8.0, 9.0]),
    #     np.array([10.0, 11.0, 12.0]),
    # ]
    # chunks = [
    #     "This is chunk 1 from file A.",
    #     "This is chunk 2 from file A.",
    #     "This is chunk 1 from file B.",
    #     "This is chunk 2 from file B.",
    # ]
    # chunks_metadata = [
    #     {"file_name": "file_a.pdf", "chunk_index": 0},
    #     {"file_name": "file_a.pdf", "chunk_index": 1},
    #     {"file_name": "file_b.docx", "chunk_index": 0},
    #     {"file_name": "file_b.docx", "chunk_index": 1},
    # ]
    dummy_files = ['/Users/zac/Downloads/Janna/verbatimprocs/FZ- revenante - sept24.docx']
    chunks, embeddings, chunks_metadata = process_files(dummy_files)

    # 1. Initialize the VectorDatabase
    vector_db = VectorDatabase(dimension=embeddings[0].shape[0]) # Initialize with dimension

    # 2. Add data to the VectorDatabase
    vector_db.add_data(embeddings, chunks, chunks_metadata)
    print("Data added to VectorDatabase.")

    # 3. Perform a query
    query_embedding = np.random.rand(embeddings[0].shape[0]).astype(np.float32)  # Random query embedding
    results = vector_db.query(query_embedding, k=2)  # Get the top 2 results

    print("\nQuery results:")
    for result in results:
        print(f"Chunk: {result['chunk_text']}")
        print(f"  File: {result['file_name']}")
        print(f"  Index: {result['chunk_index']}")

    # 4. Check if the database is empty
    print(f"\nIs the database empty? {vector_db.is_empty()}") # Check is_empty method

    # 5.  Initialize without dimension and then add data
    vector_db2 = VectorDatabase()
    vector_db2.add_data(embeddings, chunks, chunks_metadata)
    print("\nData added to VectorDatabase2 (without initial dimension).")

    query_embedding_2 = np.random.rand(embeddings[0].shape[0]).astype(np.float32)  # Random query embedding
    results_2 = vector_db2.query(query_embedding_2, k=1)
    print("\nQuery results from VectorDatabase2:")
    for result in results_2:
        print(f"Chunk: {result['chunk_text']}")
        print(f"  File: {result['file_name']}")
        print(f"  Index: {result['chunk_index']}")




"""

Key improvements and explanations:

    Class Structure: The VectorDatabase class encapsulates the FAISS index, chunks, and metadata, providing a clean and organized way to manage the vector database.
    Initialization: The __init__ method now takes an optional dimension argument. If not provided during initialization, the dimension is inferred when the first data is added. This provides more flexibility.
    Data Handling: The add_data method takes lists of embeddings, chunks, and metadata, and stores them in the object. It also converts the embeddings to a float32 numpy array, which is the format FAISS expects, and checks for dimension consistency.
    Querying: The query method performs a similarity search using FAISS and returns a list of dictionaries containing the relevant information. It also handles the case where the database is empty.
    Error Handling: The add_data and query methods include error handling for invalid input or an empty database.
    Clarity: The code is well-commented and easy to understand.
    Testing: The if __name__ == "__main__": block provides a comprehensive test of the VectorDatabase class, demonstrating how to add data, perform queries, and check if the database is empty. I've added a test for initializing the database without a dimension.



"""