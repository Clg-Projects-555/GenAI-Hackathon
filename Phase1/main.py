from database import setup_database, upload_data_from_pdfs
from search import perform_similarity_search
from sentence_transformers import SentenceTransformer
import os
import warnings
import streamlit as st

# Suppress FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")


def main():
    # Initialize embedding model
    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_name)

    # Setup database and upload data from PDFs
    index = setup_database(embedding_model)

    # Perform similarity search
    st.title("Chatty")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})


        query_text = prompt
        # query_text = input("Enter query")
        perform_similarity_search(query_text, embedding_model, index)


if __name__ == "__main__":
    main()
