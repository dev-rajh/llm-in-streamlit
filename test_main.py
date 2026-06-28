import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock required dependencies
mock_st = MagicMock()

# 🛡️ Sentinel: Mock Streamlit caching decorators to allow function execution in tests
def mock_cache_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def wrapper(func):
        return func
    return wrapper

mock_st.cache_resource = mock_cache_decorator
mock_st.cache_data = mock_cache_decorator
sys.modules['streamlit'] = mock_st

sys.modules['ollama'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from main import get_response, create_embeddings, Document, split_text_into_chunks, load_chats, load_embedding_model
import main

def test_load_embedding_model_crash():
    # 🛡️ Sentinel: Verify exception handling to prevent stack trace leaks
    with patch('main.SentenceTransformer') as mock_st_class:
        mock_st_class.side_effect = Exception("Hugging Face API Down")

        with patch('main.st.error') as mock_st_error, patch('main.st.stop') as mock_st_stop:
            result = load_embedding_model("some-model")

            mock_st_error.assert_called_once()
            mock_st_stop.assert_called_once()
            assert result is None

def test_split_text_into_chunks_dos():
    # 🛡️ Sentinel: Test that an invalid chunk overlap raises ValueError
    # to prevent infinite loop / resource exhaustion DoS
    with pytest.raises(ValueError, match="chunk_size must be strictly greater than chunk_overlap to prevent infinite loops."):
        split_text_into_chunks("hello world "*100, 100, 100)

    with pytest.raises(ValueError):
        split_text_into_chunks("hello world "*100, 50, 100)

def test_split_text_into_chunks_valid():
    chunks = split_text_into_chunks("hello world", 10, 5)
    assert len(chunks) > 0

def test_get_response_basic():
    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [Document(page_content="doc content")]

    # Mock ollama chat via patching inside get_response is tricky if it imports inside,
    # but we already mocked sys.modules['ollama']. Let's adjust our mock.
    mock_client_instance = MagicMock()
    sys.modules['ollama'].Client.return_value = mock_client_instance

    mock_client_instance.chat.return_value = [
        {'message': {'content': '  test '}},
        {'message': {'content': 'result  '}}
    ]

    template = "Context: $context\\nQuestion: $question"
    result = get_response("test query", mock_retriever, "test-model", "http://test", template)

    assert result == "test result"
    mock_retriever.get_relevant_documents.assert_called_once_with("test query")
    sys.modules['ollama'].Client.assert_called_once_with(host="http://test", timeout=120.0)
    mock_client_instance.chat.assert_called_once()

    args, kwargs = mock_client_instance.chat.call_args
    assert kwargs['model'] == "test-model"
    assert kwargs['messages'][0]['content'] == "Context: doc content\\nQuestion: test query"
    assert kwargs['stream'] is True

def test_get_response_empty_result():
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = []

    mock_client_instance = MagicMock()
    sys.modules['ollama'].Client.return_value = mock_client_instance
    mock_client_instance.chat.return_value = [{'message': {'content': '   '}}]

    template = "$context $question"
    result = get_response("test query", mock_retriever, "test-model", "http://test", template)
    assert result == ""

def test_create_embeddings_empty_chunks():
    embedding_model = MagicMock()
    result = create_embeddings([], embedding_model)
    assert result is None

def test_create_embeddings_none_chunks():
    embedding_model = MagicMock()
    result = create_embeddings(None, embedding_model)
    assert result is None

def test_create_embeddings_crash():
    # 🛡️ Sentinel: Test error handling for embedding model failure to prevent stack trace leaks
    chunks = [Document('chunk1')]
    embedding_model = MagicMock()
    embedding_model.encode.side_effect = Exception("Embedding Model Down")

    result = create_embeddings(chunks, embedding_model, storing_path="custom_path")
    assert result is None

def test_get_relevant_documents_crash():
    # 🛡️ Sentinel: Test error handling in retriever to prevent stack trace leaks
    chunks = [Document('chunk1')]
    embedding_model = MagicMock()
    embedding_model.encode.side_effect = Exception("Embedding Model Down")

    mock_index = MagicMock()

    from main import SimpleVectorStore
    store = SimpleVectorStore(mock_index, chunks, embedding_model)
    retriever = store.as_retriever()

    result = retriever.get_relevant_documents("query")
    assert result == []

def test_create_embeddings_with_chunks():
    chunks = [Document('chunk1'), Document('chunk2')]
    embedding_model = MagicMock()

    # Mock encode to return dummy embeddings
    dummy_embeddings = MagicMock()
    dummy_embeddings.shape = (2, 768)
    sys.modules['numpy'].array.return_value = dummy_embeddings
    sys.modules['numpy'].array.return_value.astype.return_value = dummy_embeddings
    embedding_model.encode.return_value = dummy_embeddings

    # Mock faiss.IndexFlatL2
    mock_index = MagicMock()
    sys.modules['faiss'].IndexFlatL2.return_value = mock_index

    # We also mock os.makedirs and faiss.write_index which are called in save_local
    import os
    from unittest.mock import patch

    with patch('os.makedirs'), patch('builtins.open'):
        result = create_embeddings(chunks, embedding_model, storing_path="custom_path")

    sys.modules['faiss'].IndexFlatL2.assert_called_once_with(768)

    # We mocked numpy so we need to be careful, but we just verify it runs without error
    assert result is not None
    assert result.index == mock_index
    assert result.chunks == chunks
    assert result.embedding_model == embedding_model

def test_load_chats_corrupted_file():
    # 🛡️ Sentinel: Test error handling for corrupted file to prevent stack trace leaks
    with patch('os.path.exists') as mock_exists, patch('builtins.open') as mock_open:
        mock_exists.return_value = True
        mock_open.side_effect = Exception("Corrupted file error")

        result = load_chats()

        assert result == {}
        mock_open.assert_called_once_with("chats.json", "r")
