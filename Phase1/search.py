import streamlit as st
import random
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Load BART model and tokenizer for summarization
bart_summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def answer_question(text):
    # Tokenize input text for BART
    bart_inputs = bart_tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    # Generate summary
    summary_ids = bart_summarizer.generate(bart_inputs['input_ids'], max_length=200, num_beams=5, early_stopping=True)
    # Decode the generated summary
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def perform_similarity_search(query_text, embedding_model, index):
    query_embedding = embedding_model.encode([query_text])[0]
    results = index.query(
        top_k=5,
        include_values=True,
        include_metadata=True,
        vector=query_embedding.tolist(),
    )
    # Define probing questions
    probing_questions = [
        "Could you clarify what you mean by that?",
        "Can you provide more details?",
        "What specifically are you referring to?",
    ]
    
    context_text = "\n".join(match["metadata"]["text"] for match in results["matches"])

    # Answer the query using BART
    answer = answer_question(context_text)
    if not answer:
        answer = probing_questions[random.randint(0, len(probing_questions) - 1)]
    response = f"Chatty: {answer}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

