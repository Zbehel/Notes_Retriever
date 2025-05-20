docker run -d -p 127.0.0.1:8501:8501 notes_retriever
# Wait for a few seconds to ensure the server starts
sleep 5

# Open browser
open http://127.0.0.1:8501