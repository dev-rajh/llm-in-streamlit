
import streamlit as st
import ollama
import pypdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from datetime import datetime
import hashlib
import tempfile
import torch
from pathlib import Path
import uuid
import requests


# Configuration

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class Config:
    MODEL = "orca-mini:3b"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    OLLAMA_API_BASE_URL = "http://localhost:11434"
    HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE = "cpu"

# Function to save chats to a JSON file
def save_chats():
    with open("chats.json", "w") as f:
        json.dump(st.session_state.chats, f)

# Function to load chats from a JSON file
def load_chats():
    if os.path.exists("chats.json"):
        with open("chats.json", "r") as f:
            return json.load(f)
    return {}


def split_text_into_chunks(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk))
        start += chunk_size - chunk_overlap
    return chunks

def process_pdf(file, chunk_size, chunk_overlap):
    filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getbuffer())
            filename = temp_file.name

        reader = pypdf.PdfReader(filename)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

        return chunks, file.name
    finally:
        if filename is not None and os.path.exists(filename):
            os.remove(filename)


class SimpleVectorStore:
    def __init__(self, index, chunks, embedding_model):
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model

    def as_retriever(self, k=4):
        class Retriever:
            def __init__(self, store, k):
                self.store = store
                self.k = k

            def get_relevant_documents(self, query):
                query_embedding = self.store.embedding_model.encode([query])
                distances, indices = self.store.index.search(np.array(query_embedding).astype('float32'), self.k)

                results = []
                for i in indices[0]:
                    if i != -1 and i < len(self.store.chunks):
                        results.append(self.store.chunks[i])
                return results

        return Retriever(self, k)

    def save_local(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "chunks.json"), "w") as f:
            json.dump([chunk.page_content for chunk in self.chunks], f)

def create_context(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

@st.cache_resource
def load_embedding_model(model_name, normalize_embedding=True):
    print("Loading embedding model...")
    return SentenceTransformer(model_name, device=Config.HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE)

def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    print("Creating embeddings...")
    if not chunks:
        print("Warning: No chunks to process. The PDF might be empty or unreadable.")
        return None

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    vectorstore = SimpleVectorStore(index, chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

def get_response(query, retriever, model, base_url, template):
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = template.format(context=context, question=query)

    client = ollama.Client(host=base_url)

    response = ""
    for chunk in client.chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True):
        if 'message' in chunk and 'content' in chunk['message']:
            response += chunk['message']['content']

    return response.strip()

def chat_without_pdf(prompt, selected_model):
    client = ollama.Client(host=Config.OLLAMA_API_BASE_URL)
    response = ""
    for chunk in client.chat(model=selected_model, messages=[{'role': 'user', 'content': prompt}], stream=True):
        if 'message' in chunk and 'content' in chunk['message']:
            response += chunk['message']['content']
    return response
    
class PDFHelper:
    def __init__(self, ollama_api_base_url, model_name=Config.MODEL, embedding_model_name=Config.EMBEDDING_MODEL_NAME):
        self._ollama_api_base_url = ollama_api_base_url
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name

    def ask(self, uploaded_file, question):
        vector_store_directory = os.path.join(str(Path.home()), 'pdf-store', 'vectorstore',
                                              'pdf-doc-helper-store', str(uuid.uuid4()))
        os.makedirs(vector_store_directory, exist_ok=True)

        embed = load_embedding_model(model_name=self._embedding_model_name)
        
        # Create a temporary file to save the uploaded file content
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                # 🛡️ Sentinel: Assign path BEFORE write so finally block cleans up even if write fails (e.g., disk full)
                temp_file_path = temp_file.name
                temp_file.write(uploaded_file.getvalue())

            # Use pypdf to load text
            reader = pypdf.PdfReader(temp_file_path)
            text = ""
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        finally:
            # 🛡️ Sentinel: Clean up the temporary file in a finally block to prevent disk leaks if reading fails
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        if not text.strip():
            return "The uploaded PDF appears to be empty or unreadable. Please check the file and try again."

        documents = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        
        if not documents:
            return "Unable to extract meaningful content from the PDF. The file might be empty, corrupted, or contain only images."

        vectorstore = create_embeddings(chunks=documents, embedding_model=embed, storing_path=vector_store_directory)
        
        if vectorstore is None:
            return "Unable to process the PDF content. The file might be empty or contain no extractable text."

        retriever = vectorstore.as_retriever()

        template = """
        ### System:
        You are an honest assistant.
        You will accept PDF files and you will answer the question asked by the user appropriately.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
    
        ### Context:
        {context}
    
        ### User:
        {question}
    
        ### Response:
        """

        return get_response(question, retriever, self._model_name, self._ollama_api_base_url, template)

def pull_model(model_name):
    print(f"Pulling model '{model_name}'...")
    url = f"{Config.OLLAMA_API_BASE_URL}/api/pull"
    data = json.dumps({"name": model_name})
    headers = {'Content-Type': 'application/json'}

    with requests.post(url, data=data, headers=headers, stream=True, timeout=30) as response:
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    print(chunk.decode('utf-8'), end='')
        else:
            print(f"Error: {response.status_code} - {response.text}")

def main():
    st.set_page_config(page_title="Ollama PDF Chat Bot")
    st.title("Ollama PDF Chat Bot")

    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = ""
    if "current_file" not in st.session_state:
        st.session_state.current_file = None

    with st.sidebar:
        st.write('This chatbot can chat normally or answer questions about a PDF file.')
        available_models = ollama.list()['models']
        selected_model = st.selectbox("Select a model", [model['model'] for model in available_models])

        uploaded_file = st.file_uploader("Upload a PDF file (optional)", type="pdf")

        if uploaded_file is not None:
            st.write("PDF mode: Ask questions about the uploaded document.")
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=50)
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=100, value=50, step=10)

            if st.session_state.current_file != uploaded_file.name:
                chunks, filename = process_pdf(uploaded_file, chunk_size, chunk_overlap)
                context = create_context(chunks)
                st.session_state.context = context
                st.session_state.current_file = filename
                
                chat_title = f"Chat about {filename}"
                st.session_state.current_chat = chat_title
                st.session_state.chats[chat_title] = {
                    'messages': [],
                    'context': context,
                    'file': filename
                }
                st.session_state.messages = []
                
                system_msg = f"New file uploaded: {filename}. You can now ask questions about this document."
                st.session_state.messages.append({"role": "system", "content": system_msg, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                
                save_chats()
                st.rerun()
        else:
            st.write("Regular chat mode: Ask any questions.")

    # Chat selection
    chat_names = ["New Chat"] + list(st.session_state.chats.keys())
    current_chat = st.selectbox("Select a chat", chat_names, index=chat_names.index(st.session_state.current_chat))


    if current_chat != st.session_state.current_chat:
        if current_chat == "New Chat":
            st.session_state.current_chat = "New Chat"
            st.session_state.messages = []
            st.session_state.context = ""
            st.session_state.current_file = None
        else:
            st.session_state.current_chat = current_chat
            st.session_state.messages = st.session_state.chats[current_chat]['messages']
            st.session_state.context = st.session_state.chats[current_chat]['context']
            st.session_state.current_file = st.session_state.chats[current_chat]['file']
        
        st.rerun()
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['timestamp']}**")
            st.markdown(message["content"])

    # Chat input and response handling
    if prompt := st.chat_input("What is your question?"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        
        with st.chat_message("user"):
            st.markdown(f"**{timestamp}**")
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if st.session_state.current_file is None:
                full_response = chat_without_pdf(prompt, selected_model)
            else:
                pdf_helper = PDFHelper(
                    ollama_api_base_url=Config.OLLAMA_API_BASE_URL,
                    model_name=selected_model
                )
                full_response = pdf_helper.ask(
                    uploaded_file=uploaded_file,
                    question=prompt
                )
            
            message_placeholder.markdown(full_response)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": timestamp})

        
        # Update the chat in st.session_state.chats
        if st.session_state.current_chat != "New Chat":
            st.session_state.chats[st.session_state.current_chat]['messages'] = st.session_state.messages

        save_chats()
        st.rerun()

    if st.session_state.current_file:
        st.sidebar.write(f"Current file: {st.session_state.current_file}")
    if st.session_state.context:
        with st.expander("Current Context"):
            st.write(st.session_state.context)

    if st.sidebar.button('New Chat'):
        st.session_state.current_chat = "New Chat"
        st.session_state.messages = []
        st.session_state.context = ""
        st.session_state.current_file = None
        st.rerun()

if __name__ == "__main__":
    main()
