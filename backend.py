from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

# Load documents and create chunks
def load_and_process_documents(folder_path):
    loader = PyPDFLoader(folder_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

# # Initialize FAISS index
def initialize_faiss(documents):
    embedding_model = HuggingFaceEmbeddings()
    faiss_index = FAISS.from_documents(documents, embedding_model)
    return faiss_index, embedding_model

def get_chatbot_response(query, faiss_index, documents, embedding_model, model, tokenizer):
    # Casual responses for common greetings
    casual_responses = {
        "hi": "Hello! How can I help you?",
        "hello": "Hi there! What would you like to know?",
        "hey": "Hey! how can I assist you?",
        "bye": "Goodbye! It was nice talking to you!",
        "good bye": "Goodbye! Take care!",
        "how are you": "I'm an AI-powered assistant, always here to help you!",
    }
    
    # Handle casual queries
    if query.strip().lower() in casual_responses:
        return casual_responses[query.strip().lower()]
    
    # Generate query vector
    query_vector = embedding_model.embed_query(query)
    query_vector = np.array(query_vector).reshape(1, -1)

    # Retrieve top-2 relevant chunks
    _, indices = faiss_index.index.search(query_vector, k=2)
    if not indices.size or indices[0][0] >= len(documents):
        return "Sorry, I couldn't find any relevant information in the documents."

    # Construct the context from the retrieved documents
    context = " ".join(
        [documents[idx].page_content for idx in indices[0] if idx < len(documents) and idx >= 0]
    )
    if not context.strip():
        return "Sorry, I couldn't find any relevant information in the documents."

    # Define the prompt for the language model
    prompt = (
        f"You are an expert AI assistant trained to answer questions and explain concepts in detail.\n\n"
        f"User's Question: {query}\n\n"
        f"Relevant Context: {context}\n\n"
        f"Based on the context, provide a detailed, clear, and concise explanation to the user's question."
    )

    # Tokenize and generate a response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1000,
        num_beams=5,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

# Initialize the model and tokenizer
def initialize_model():
    model_name = "google/flan-t5-base"  # Lightweight conversational model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer
