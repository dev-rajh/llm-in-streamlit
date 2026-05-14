import pytest
import sys
from unittest.mock import MagicMock

# Mock required dependencies
sys.modules['streamlit'] = MagicMock()
sys.modules['ollama'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['langchain.text_splitter'] = MagicMock()
sys.modules['langchain.document_loaders'] = MagicMock()
sys.modules['langchain.embeddings'] = MagicMock()
sys.modules['langchain.vectorstores'] = MagicMock()
sys.modules['langchain.chains'] = MagicMock()
sys.modules['langchain.chat_models'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()

from main import get_response, create_embeddings

def test_get_response_basic():
    # Mock chain
    def mock_chain(input_dict):
        assert input_dict['query'] == "test query"
        return {'result': "  test result  "}

    result = get_response("test query", mock_chain)
    assert result == "test result"

def test_get_response_empty_result():
    def mock_chain(input_dict):
        return {'result': "   "}

    result = get_response("test query", mock_chain)
    assert result == ""

def test_get_response_missing_key():
    def mock_chain(input_dict):
        return {'wrong_key': "value"}

    with pytest.raises(KeyError):
        get_response("test query", mock_chain)


def test_create_embeddings_empty_chunks():
    embedding_model = MagicMock()
    result = create_embeddings([], embedding_model)
    assert result is None

def test_create_embeddings_none_chunks():
    embedding_model = MagicMock()
    result = create_embeddings(None, embedding_model)
    assert result is None

def test_create_embeddings_with_chunks():
    chunks = ['chunk1', 'chunk2']
    embedding_model = MagicMock()

    # We need to mock FAISS.from_documents which is accessed through the mocked langchain.vectorstores module
    mock_vectorstore = MagicMock()
    sys.modules['langchain.vectorstores'].FAISS.from_documents.return_value = mock_vectorstore

    result = create_embeddings(chunks, embedding_model, storing_path="custom_path")

    sys.modules['langchain.vectorstores'].FAISS.from_documents.assert_called_once_with(chunks, embedding_model)
    mock_vectorstore.save_local.assert_called_once_with("custom_path")
    assert result == mock_vectorstore
