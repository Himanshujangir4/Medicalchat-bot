import streamlit as st
st.cache_data.clear()
import os
from helper import build_rag_chain

# Set page config
st.set_page_config(page_title="🧑‍⚕️ Medical Chatbot", layout="centered")

st.title("🧑‍⚕️ Medical Chatbot - GALE Book Based")
st.write("Ask any medical question based on the uploaded book...")

# Question input
question = st.text_input("Enter your question:")

if question:
    try:
        with st.spinner("Searching and generating response..."):
            rag_chain = build_rag_chain(os.path.abspath("../research/Data"))
            response = rag_chain.invoke({"query": question})
            st.success("Here's the answer:")
            st.write(response["result"])
    except Exception as e:
        st.error(f"Error: {e}")
