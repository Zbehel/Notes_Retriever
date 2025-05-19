#!/bin/bash
docker stop $(docker ps -q --filter ancestor=notes_retriever)
docker build --no-cache -t notes_retriever .
docker run -d -p 127.0.0.1:8501:8501 notes_retriever