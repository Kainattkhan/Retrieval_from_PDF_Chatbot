import streamlit as st
from backend import load_and_process_documents, initialize_faiss, initialize_model, get_chatbot_response

# Load and process documents
st.title("Chatbot for Attention is all you need - Paper ")
st.sidebar.header("Configuration will be added later!")

folder_path = "./attention.pdf"

# Load and process documents
documents = load_and_process_documents(folder_path)

# Initialize FAISS and embedding model
if "faiss_index" not in st.session_state:
    faiss_index, embedding_model = initialize_faiss(documents)
    st.session_state["faiss_index"] = faiss_index
    st.session_state["embedding_model"] = embedding_model

# Access FAISS index and embedding model
faiss_index = st.session_state["faiss_index"]
embedding_model = st.session_state["embedding_model"]

if "model" not in st.session_state:
    model, tokenizer = initialize_model()
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer

# Chatbot UI
if "faiss_index" in st.session_state and "model" in st.session_state:
    user_input = st.text_input("Ask a question:", "")
    print(f"User Input: {user_input}")
    if st.button("Submit") and user_input:
        response = get_chatbot_response(
            user_input,
            st.session_state["faiss_index"],
            documents,
            st.session_state["embedding_model"],
            st.session_state["model"],
            st.session_state["tokenizer"],
        )
        print("response: ",response)
        st.write("Chatbot Response:", response)






