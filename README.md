# Ollama PDF Chat Bot
## Overview
The Ollama PDF Chat Bot is a Streamlit-based application designed to facilitate conversations about the content of PDF files. It leverages the Ollama API for natural language processing and the LangChain library for handling document embeddings and retrieval-based question answering.
Features

* **PDF Upload**: Users can upload PDF files to ask questions about their content.

* **Chat Interface**: A user-friendly chat interface for asking questions and receiving answers.

* **Context Management**: The application maintains context from the uploaded PDF to provide relevant answers.

* **Model Selection**: Users can select different models for generating responses.

* **Chat History**: Saves and loads chat history from a JSON file.

## Installation

To run the Ollama PDF Chat Bot, you need to have the following dependencies installed:
```batch
pip install streamlit ollama langchain torch requests
```
## Configuration

The application uses a configuration class to set up various parameters:
```python
class Config:
  MODEL = "orca-mini:3b"
  EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
  OLLAMA_API_BASE_URL = "http://localhost:11434"
  HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE = "cpu"
```
You can modify these parameters according to your requirements.
## Usage

### 1. Run the Application:
```
streamlit run main.py
```
### 2. Upload a PDF:
* Use the file uploader in the sidebar to upload a PDF file.
* Adjust the chunk size and chunk overlap using the sliders provided.

### 3. Select a Model:
* Choose a model from the dropdown list in the sidebar.

### 4. Start Chatting:
* Use the chat input box to ask questions.
* The application will provide answers based on the content of the uploaded PDF or general knowledge if no PDF is uploaded.

## Functions and Classes
### Configuration

* **Config**: A class containing configuration parameters for the application.

### Chat Management

* **save_chats()**: Saves the current chat history to a JSON file.

* **load_chats()**: Loads the chat history from a JSON file.

### PDF Processing

* **process_pdf(file, chunk_size, chunk_overlap)**: Processes the uploaded PDF file into chunks.

* **create_context(chunks)**: Creates a context string from the PDF chunks.

* **generate_chat_title(context, question)**: Generates a title for the chat based on the context and question.

### Embeddings and QA Chain

* **load_embedding_model(model_name, normalize_embedding=True)**: Loads the embedding model.

* **create_embeddings(chunks, embedding_model, storing_path="vectorstore")**: Creates embeddings for the PDF chunks.

* **load_qa_chain(retriever, llm, prompt)**: Loads the QA chain for retrieval-based question answering.

* **get_response(query, chain)**: Gets a response from the QA chain.

### Chat Functions

* **chat_without_pdf(prompt, selected_model)**: Generates a response without using a PDF.

* **PDFHelper**: A class to handle PDF-based question answering.

### Model Management

* **pull_model(model_name)**: Pulls a model from the Ollama API.

### Main Function

* **main()**: The main function that sets up the Streamlit interface and handles user interactions.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.
