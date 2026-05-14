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

from main import get_response

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
