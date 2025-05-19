from transformers import pipeline
import streamlit as st

def get_answer(query, context):
    try:
        qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        result = qa_model(question=query, context=context)
        return result["answer"]
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I could not process your query."