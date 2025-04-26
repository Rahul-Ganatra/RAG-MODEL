import os
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini embedding and chat models
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
    task_type="semantic_similarity"
)
chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Helper: Extract text from different file types
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

# Helper: Chunk text
def chunk_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Build FAISS index from document chunks
def build_faiss_index(chunks):
    embeddings = np.array([embedding_model.embed_documents([chunk])[0] for chunk in chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Retrieve top-k relevant chunks
def retrieve_chunks(query, chunks, index, k=2):
    query_vec = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# Ask Gemini with context
def ask_gemini(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = chat_model.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# Streamlit UI
st.title("Chat with your Document (Gemini + FAISS)")

uploaded_file = st.file_uploader("Upload a document (txt, pdf, docx)", type=["txt", "pdf", "docx"])
if uploaded_file:
    text = extract_text(uploaded_file)
    if not text.strip():
        st.error("Could not extract text from the uploaded file.")
    else:
        st.success("Document loaded and processed!")
        chunks = chunk_text(text)
        st.write(f"Document split into {len(chunks)} chunks.")
        with st.spinner("Embedding and indexing..."):
            index, _ = build_faiss_index(chunks)
        st.session_state['chunks'] = chunks
        st.session_state['index'] = index

if 'chunks' in st.session_state and 'index' in st.session_state:
    # Add New Chat and Clear Chat buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat"):
            # Remove all session state related to the chat and document
            for key in ['chat_history', 'chunks', 'index']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state['chat_history'] = []
            st.rerun()

    # Display chat history with icons
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    for q, a in st.session_state['chat_history']:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(a)

    # Use a unique key for the input so it clears after each question
    question = st.text_input("Ask a question about your document:", key=f"question_input_{len(st.session_state['chat_history'])}")

    if question:
        with st.spinner("Retrieving answer..."):
            top_chunks = retrieve_chunks(question, st.session_state['chunks'], st.session_state['index'], k=1)
            answer = ask_gemini(question, top_chunks)
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
        st.session_state['chat_history'].append((question, answer))
        st.rerun()

# If no document is loaded, prompt the user to upload one
if 'chunks' not in st.session_state or 'index' not in st.session_state:
    st.info("Please upload a document to start chatting.")