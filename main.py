
import streamlit as st
import ollama
import pypdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from datetime import datetime
import string
import hashlib
import tempfile
import torch
from pathlib import Path
import uuid
import requests
import concurrent.futures
import copy


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

@st.cache_resource
def get_io_executor():
    """🛡️ Sentinel: Background thread pool for offloading file I/O to maintain UI responsiveness."""
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _save_chats_task(chats_data):
    try:
        with open("chats.json", "w") as f:
            json.dump(chats_data, f)
    except Exception as e:
        print(f"Error saving chats: {e}")

# Function to save chats to a JSON file
def save_chats():
    # 🛡️ Sentinel: Deepcopy session state to prevent thread-safety issues and offload to background thread to prevent UI blocking
    chats_copy = copy.deepcopy(st.session_state.chats)
    get_io_executor().submit(_save_chats_task, chats_copy)

# Function to load chats from a JSON file
def load_chats():
    if os.path.exists("chats.json"):
        try:
            with open("chats.json", "r") as f:
                return json.load(f)
        except Exception as e:
            # 🛡️ Sentinel: Handle exception gracefully to prevent Information Disclosure and persistent DoS if chats.json is corrupted
            print(f"Error loading chats: {e}")
            return {}
    return {}


def split_text_into_chunks(text, chunk_size, chunk_overlap):
    # 🛡️ Sentinel: Prevent infinite loop / DoS if chunk_size <= chunk_overlap
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be strictly greater than chunk_overlap to prevent infinite loops.")

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
            # 🛡️ Sentinel: Assign filename before writing to ensure cleanup if disk is full
            filename = temp_file.name
            temp_file.write(file.getbuffer())

        reader = pypdf.PdfReader(filename)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

        return chunks, file.name
    except Exception as e:
        print(f"Error processing PDF: {e}")
        st.error("An error occurred while processing the PDF file. Please ensure it is a valid PDF and try again.")
        st.stop()
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
    try:
        # 🛡️ Sentinel: Wrap model loading in try-except to handle network/API issues and avoid stack trace leakage (Information Disclosure/DoS)
        return SentenceTransformer(model_name, device=Config.HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        st.error("An error occurred while loading the embedding model. The model service might be unavailable.")
        st.stop()
        return None

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
    if storing_path:
        vectorstore.save_local(storing_path)
    return vectorstore

def get_response(query, retriever, model, base_url, template):
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 🛡️ Sentinel: Use string.Template for safe substitution to prevent Context Poisoning/Prompt Injection vulnerabilities present in sequential string replacements, and to avoid KeyError/ValueError from user-controlled input containing unescaped curly braces in template.format().
    prompt = string.Template(template).safe_substitute(context=context, question=query)

    # 🛡️ Sentinel: Explicitly configure timeout to prevent indefinite blocking (DoS risk). Set upper bound to allow for model cold starts.
    client = ollama.Client(host=base_url, timeout=120)

    response = ""
    try:
        # 🛡️ Sentinel: Handle external API errors to prevent stack trace leakage
        for chunk in client.chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True):
            if 'message' in chunk and 'content' in chunk['message']:
                response += chunk['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama API: {e}")
        return "An error occurred while communicating with the AI service. Please try again later."

    return response.strip()

def chat_without_pdf(prompt, selected_model):
    # 🛡️ Sentinel: Explicitly configure timeout to prevent indefinite blocking (DoS risk). Set upper bound to allow for model cold starts.
    client = ollama.Client(host=Config.OLLAMA_API_BASE_URL, timeout=120)
    response = ""
    try:
        # 🛡️ Sentinel: Handle external API errors to prevent stack trace leakage
        for chunk in client.chat(model=selected_model, messages=[{'role': 'user', 'content': prompt}], stream=True):
            if 'message' in chunk and 'content' in chunk['message']:
                response += chunk['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama API: {e}")
        return "An error occurred while communicating with the AI service. Please try again later."
    return response
    
class PDFHelper:
    def __init__(self, ollama_api_base_url, model_name=Config.MODEL, embedding_model_name=Config.EMBEDDING_MODEL_NAME):
        self._ollama_api_base_url = ollama_api_base_url
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name

    def ask(self, uploaded_file, question):
        embed = load_embedding_model(model_name=self._embedding_model_name)
        
        temp_file_path = None
        text = ""
        try:
            # Create a temporary file to save the uploaded file content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                # 🛡️ Sentinel: Assign filename before writing to ensure cleanup in finally block
                temp_file_path = temp_file.name
                temp_file.write(uploaded_file.getvalue())

            # Use pypdf to load text
            reader = pypdf.PdfReader(temp_file_path)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF in PDFHelper.ask: {e}")
            return "An error occurred while reading the PDF file. It might be malformed or corrupted."
        finally:
            # 🛡️ Sentinel: Ensure the temporary file is cleaned up even if reading/writing fails
            if temp_file_path is not None and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        if not text.strip():
            return "The uploaded PDF appears to be empty or unreadable. Please check the file and try again."

        documents = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        
        if not documents:
            return "Unable to extract meaningful content from the PDF. The file might be empty, corrupted, or contain only images."

        # 🛡️ Sentinel: Pass storing_path=None to avoid unbounded disk usage (DoS risk) from temporary vector stores
        vectorstore = create_embeddings(chunks=documents, embedding_model=embed, storing_path=None)
        
        if vectorstore is None:
            return "Unable to process the PDF content. The file might be empty or contain no extractable text."

        retriever = vectorstore.as_retriever()

        template = """
        ### System:
        You are an honest assistant.
        You will accept PDF files and you will answer the question asked by the user appropriately.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
    
        ### Context:
        $context
    
        ### User:
        $question
    
        ### Response:
        """

        return get_response(question, retriever, self._model_name, self._ollama_api_base_url, template)

def pull_model(model_name):
    print(f"Pulling model '{model_name}'...")
    url = f"{Config.OLLAMA_API_BASE_URL}/api/pull"
    data = json.dumps({"name": model_name})
    headers = {'Content-Type': 'application/json'}

    try:
        with requests.post(url, data=data, headers=headers, stream=True, timeout=30) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        print(chunk.decode('utf-8'), end='')
            else:
                print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Network error while pulling model: {e}")

@st.cache_data(ttl=60)
def get_available_models():
    """🛡️ Sentinel: Wrap external API call to prevent unhandled exceptions and stack trace leaks."""
    try:
        # 🛡️ Sentinel: Avoid module-level ollama.list() to configure network timeout and prevent indefinite blocking
        client = ollama.Client(host=Config.OLLAMA_API_BASE_URL, timeout=30)
        return client.list().get('models', [])
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

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
        available_models = get_available_models()

        if not available_models:
            st.error("Could not connect to the Ollama service. Please ensure it is running.")
            st.stop()

        selected_model = st.selectbox("Select a model", [model.get('model', 'Unknown') for model in available_models])

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
