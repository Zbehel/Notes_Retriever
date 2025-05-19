## This project aims to help long notes writers to locate previous scripts written and drown in massive texts

docker build --no-cache -t notes_retriever .
docker run -d -p 127.0.0.1:8501:8501 notes_retriever