
import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
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

def process_pdf(file, chunk_size, chunk_overlap):
    file_hash = hashlib.md5(file.getvalue()).hexdigest()
    filename = f"temp_{file_hash}.pdf"
    
    with open(filename, "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(pages)

    os.remove(filename)

    return chunks, file.name

def create_context(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

def generate_chat_title(context, question):
    response = ollama.chat(
        model="orca-mini:3b",
        messages=[
            {"role": "system", "content": "Generate a short, descriptive title (max 6 words) for a chat based on the given context and question."},
            {"role": "user", "content": f"Context: {context[:500]}...\n\nQuestion: {question}\n\nTitle:"}
        ]
    )
    return response['message']['content'].strip()

def load_embedding_model(model_name, normalize_embedding=True):
    print("Loading embedding model...")
    hugging_face_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': Config.HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE},
        encode_kwargs={
            'normalize_embeddings': normalize_embedding
        }
    )
    return hugging_face_embeddings

def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    print("Creating embeddings...")
    if not chunks:
        print("Warning: No chunks to process. The PDF might be empty or unreadable.")
        return None
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

def load_qa_chain(retriever, llm, prompt):
    print("Loading QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def get_response(query, chain):
    response = chain({'query': query})
    return response['result'].strip()

def chat_without_pdf(prompt, selected_model):
    llm = ChatOllama(
        temperature=0,
        base_url=Config.OLLAMA_API_BASE_URL,
        model=selected_model,
        streaming=True,
        top_k=10,
        top_p=0.3,
        num_ctx=3072,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return llm.predict(prompt)
    
class PDFHelper:
    def __init__(self, ollama_api_base_url, model_name=Config.MODEL, embedding_model_name=Config.EMBEDDING_MODEL_NAME):
        self._ollama_api_base_url = ollama_api_base_url
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name

    def ask(self, uploaded_file, question):
        vector_store_directory = os.path.join(str(Path.home()), 'langchain-store', 'vectorstore',
                                              'pdf-doc-helper-store', str(uuid.uuid4()))
        os.makedirs(vector_store_directory, exist_ok=True)

        llm = ChatOllama(
            temperature=0,
            base_url=self._ollama_api_base_url,
            model=self._model_name,
            streaming=True,
            top_k=10,
            top_p=0.3,
            num_ctx=3072,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        embed = load_embedding_model(model_name=self._embedding_model_name)
        
        # Create a temporary file to save the uploaded file content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Use the temporary file path for PyPDFLoader
        docs = PyPDFLoader(file_path=temp_file_path).load()
        
        # Clean up the temporary file
        os.unlink(temp_file_path) 
        
        if not docs:
            return "The uploaded PDF appears to be empty or unreadable. Please check the file and try again."

        documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        
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

        prompt = PromptTemplate.from_template(template)
        chain = load_qa_chain(retriever, llm, prompt)
        return get_response(question, chain)

def pull_model(model_name):
    print(f"Pulling model '{model_name}'...")
    url = f"{Config.OLLAMA_API_BASE_URL}/api/pull"
    data = json.dumps({"name": model_name})
    headers = {'Content-Type': 'application/json'}

    with requests.post(url, data=data, headers=headers, stream=True) as response:
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
        selected_model = st.selectbox("Select a model", [model['name'] for model in available_models])

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
