#!/bin/bash
docker stop $(docker ps -q --filter ancestor=notes_retriever)
docker build --no-cache -t notes_retriever .
docker run -d -p 127.0.0.1:8501:8501 notes_retriever
# Wait for a few seconds to ensure the server starts
sleep 5

# Open browser
xdg-open http://127.0.0.1:8501