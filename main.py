
import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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

# Project Management
def get_project_dir(project_name):
    if not project_name:
        return None
    project_dir = os.path.join("data", "projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "files"), exist_ok=True)
    return project_dir

def get_project_chats_file(project_name):
    project_dir = get_project_dir(project_name)
    if project_dir:
        return os.path.join(project_dir, "chats.json")
    return "chats.json"

# Function to save chats to a JSON file
def save_chats(project_name=None):
    chats_file = get_project_chats_file(project_name)
    with open(chats_file, "w") as f:
        json.dump(st.session_state.chats, f)

# Function to load chats from a JSON file
def load_chats(project_name=None):
    chats_file = get_project_chats_file(project_name)
    if os.path.exists(chats_file):
        with open(chats_file, "r") as f:
            return json.load(f)
    return {}

def process_pdf(file, chunk_size, chunk_overlap, project_name="Default"):
    project_dir = get_project_dir(project_name)
    files_dir = os.path.join(project_dir, "files")

    file_hash = hashlib.md5(file.getvalue()).hexdigest()
    filename = f"temp_{file_hash}.pdf"
    
    with open(filename, "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    # Save as MD file
    md_filename = os.path.splitext(file.name)[0] + ".md"
    md_filepath = os.path.join(files_dir, md_filename)

    with open(md_filepath, "w", encoding="utf-8") as md_file:
        for page in pages:
            md_file.write(page.page_content + "\n\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(pages)

    os.remove(filename)

    return chunks, file.name, md_filepath

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

def summarize_chat(messages, selected_model):
    if len(messages) <= 10:
        return ""

    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-2]])

    summary_prompt = f"Summarize the following chat history concisely, capturing the key points and context:\n\n{chat_text}"

    response = ollama.chat(
        model=selected_model,
        messages=[{"role": "user", "content": summary_prompt}]
    )

    summary = response['message']['content'].strip()
    return summary

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
        
    if os.path.exists(os.path.join(storing_path, "index.faiss")):
        vectorstore = FAISS.load_local(storing_path, embedding_model, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        
    vectorstore.save_local(storing_path)
    return vectorstore

def chat_with_ollama(messages, selected_model, temperature, max_tokens, system_prompt, custom_instruction, tone, context=None, summary=None):
    # Construct the final system prompt
    final_system_prompt = system_prompt
    if custom_instruction:
        final_system_prompt += f"\n\nCustom Instructions: {custom_instruction}"
    if tone:
        final_system_prompt += f"\n\nTone: Please respond in a {tone.lower()} tone."
    if context:
        final_system_prompt += f"\n\nUse the following context from uploaded documents to answer the user's questions:\n{context}"
    if summary:
        final_system_prompt += f"\n\nPrevious chat summary to keep in context:\n{summary}"
        
    formatted_messages = [{"role": "system", "content": final_system_prompt}]
    
    # If there's a summary, we only include the last 2 messages (the ones not summarized) to save tokens
    messages_to_include = messages if not summary else messages[-2:]
    
    for msg in messages_to_include:
        if msg["role"] != "system": # We've already set the overall system prompt
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

    response_stream = ollama.chat(
        model=selected_model,
        messages=formatted_messages,
        stream=True,
        options={
            "temperature": temperature,
            "num_predict": max_tokens
        }
    )
    return response_stream


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

    os.makedirs(os.path.join("data", "projects"), exist_ok=True)

    if "current_project" not in st.session_state:
        st.session_state.current_project = "Default"

    if "chats" not in st.session_state:
        st.session_state.chats = load_chats(st.session_state.current_project)
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = ""
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "chat_summary" not in st.session_state:
        st.session_state.chat_summary = ""

    with st.sidebar:
        st.header("Project Management")
        project_names = [d for d in os.listdir(os.path.join("data", "projects")) if os.path.isdir(os.path.join("data", "projects", d))]
        if "Default" not in project_names:
            project_names = ["Default"] + project_names

        selected_project = st.selectbox("Select a project", project_names, index=project_names.index(st.session_state.current_project) if st.session_state.current_project in project_names else 0)

        new_project_name = st.text_input("Create new project")
        if st.button("Create Project") and new_project_name:
            if new_project_name not in project_names:
                get_project_dir(new_project_name)
                st.session_state.current_project = new_project_name
                st.session_state.chats = load_chats(st.session_state.current_project)
                st.session_state.current_chat = "New Chat"
                st.session_state.messages = []
                st.session_state.context = ""
                st.session_state.current_file = None
                st.session_state.chat_summary = ""
                st.rerun()

        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            st.session_state.chats = load_chats(st.session_state.current_project)
            st.session_state.current_chat = "New Chat"
            st.session_state.messages = []
            st.session_state.context = ""
            st.session_state.current_file = None
            st.session_state.chat_summary = ""
            st.rerun()

        st.divider()

        st.write('This chatbot can chat normally or answer questions about a PDF file.')

        st.header("Model Selection & Parameters")
        available_models = ollama.list().get('models', [])
        if available_models:
            st.session_state.selected_model = st.selectbox("Select a model", [model['model'] for model in available_models], key="model_select")
        else:
            st.session_state.selected_model = None
            st.warning("No Ollama models found. Please make sure Ollama is running.")

        st.session_state.temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        st.session_state.max_tokens = st.number_input("Max Tokens (num_predict)", min_value=100, max_value=8192, value=3072)

        st.header("Personalisation")
        st.session_state.tone = st.selectbox("Tone", ["Neutral", "Professional", "Casual", "Friendly", "Concise", "Detailed"])
        st.session_state.system_prompt = st.text_area("System Prompt", value="You are a helpful, honest assistant.")
        st.session_state.custom_instruction = st.text_area("Custom Instructions", value="")

        st.divider()

        uploaded_file = st.file_uploader("Upload a PDF file (optional)", type="pdf")

        if uploaded_file is not None:
            st.write("PDF mode: Ask questions about the uploaded document.")
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=50)
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=100, value=50, step=10)

            if st.session_state.current_file != uploaded_file.name:
                with st.spinner("Processing PDF..."):
                    chunks, filename, md_filepath = process_pdf(uploaded_file, chunk_size, chunk_overlap, st.session_state.current_project)

                    # Update vectorstore
                    project_dir = get_project_dir(st.session_state.current_project)
                    vectorstore_dir = os.path.join(project_dir, "vectorstore")
                    os.makedirs(vectorstore_dir, exist_ok=True)

                    embed = load_embedding_model(model_name=Config.EMBEDDING_MODEL_NAME)
                    create_embeddings(chunks, embed, storing_path=vectorstore_dir)

                    context = create_context(chunks)
                    st.session_state.context = context
                    st.session_state.current_file = filename
                    st.session_state.current_md_file = md_filepath

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

                    save_chats(st.session_state.current_project)
                st.rerun()

            if "current_md_file" in st.session_state and st.session_state.current_md_file:
                with open(st.session_state.current_md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
                st.download_button(
                    label="Download as MD",
                    data=md_content,
                    file_name=os.path.basename(st.session_state.current_md_file),
                    mime="text/markdown"
                )
        else:
            st.write("Regular chat mode: Ask any questions.")

    # Chat selection and Search
    st.sidebar.header("Chat Search")
    search_query = st.sidebar.text_input("Search chats")

    chat_names = ["New Chat"] + list(st.session_state.chats.keys())

    if search_query:
        chat_names = ["New Chat"] + [chat for chat in list(st.session_state.chats.keys()) if search_query.lower() in chat.lower() or any(search_query.lower() in msg['content'].lower() for msg in st.session_state.chats[chat]['messages'])]

    current_chat = st.selectbox("Select a chat", chat_names, index=chat_names.index(st.session_state.current_chat) if st.session_state.current_chat in chat_names else 0)


    if current_chat != st.session_state.current_chat:
        if current_chat == "New Chat":
            st.session_state.current_chat = "New Chat"
            st.session_state.messages = []
            st.session_state.context = ""
            st.session_state.current_file = None
            st.session_state.chat_summary = ""
        else:
            st.session_state.current_chat = current_chat
            st.session_state.messages = st.session_state.chats[current_chat]['messages']
            st.session_state.context = st.session_state.chats[current_chat]['context']
            st.session_state.current_file = st.session_state.chats[current_chat]['file']
            st.session_state.chat_summary = st.session_state.chats[current_chat].get('summary', "")
        
        st.rerun()
        
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['timestamp']}**")
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "stats" in message:
                    stats = message["stats"]
                    st.caption(f"Tokens: {stats.get('prompt_eval_count', 0)} prompt, {stats.get('eval_count', 0)} completion | Duration: {stats.get('total_duration', 0)/1e9:.2f}s")

                # Option to save response
                save_key = f"save_{i}"
                if st.button("Save response as file", key=save_key):
                    project_dir = get_project_dir(st.session_state.current_project)
                    files_dir = os.path.join(project_dir, "files")
                    response_filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    response_filepath = os.path.join(files_dir, response_filename)
                    with open(response_filepath, "w", encoding="utf-8") as f:
                        f.write(message["content"])
                    st.success(f"Response saved to {response_filepath}")

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
            final_stats = {}
            
            # Extract relevant context if using a PDF
            context_to_use = None
            if st.session_state.current_file:
                project_dir = get_project_dir(st.session_state.current_project)
                vectorstore_dir = os.path.join(project_dir, "vectorstore")
                if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
                    embed = load_embedding_model(model_name=Config.EMBEDDING_MODEL_NAME)
                    vectorstore = FAISS.load_local(vectorstore_dir, embed, allow_dangerous_deserialization=True)
                    docs = vectorstore.similarity_search(prompt, k=4)
                    context_to_use = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate response stream
            if st.session_state.selected_model:
                stream = chat_with_ollama(
                    messages=st.session_state.messages,
                    selected_model=st.session_state.selected_model,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                    system_prompt=st.session_state.system_prompt,
                    custom_instruction=st.session_state.custom_instruction,
                    tone=st.session_state.tone,
                    context=context_to_use,
                    summary=st.session_state.chat_summary
                )

                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response += chunk['message']['content']
                        message_placeholder.markdown(full_response + "▌")
                    if chunk.get('done'):
                        final_stats = chunk

                message_placeholder.markdown(full_response)

                # Update messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stats": {
                        "total_duration": final_stats.get("total_duration"),
                        "prompt_eval_count": final_stats.get("prompt_eval_count"),
                        "eval_count": final_stats.get("eval_count")
                    }
                })

                # Check for summarization
                if len(st.session_state.messages) > 10:
                    st.session_state.chat_summary = summarize_chat(st.session_state.messages, st.session_state.selected_model)
            else:
                st.error("Please ensure Ollama is running and a model is selected.")

        # Update the chat in st.session_state.chats
        if st.session_state.current_chat != "New Chat":
            st.session_state.chats[st.session_state.current_chat]['messages'] = st.session_state.messages
            st.session_state.chats[st.session_state.current_chat]['summary'] = st.session_state.chat_summary

        save_chats(st.session_state.current_project)
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
        st.session_state.chat_summary = ""
        st.rerun()

if __name__ == "__main__":
    main()
