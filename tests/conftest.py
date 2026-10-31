import sys
import os

# Make src/ importable when pytest is run from RAG_GPT/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import MagicMock

os.environ.setdefault("OPENROUTER_API_KEY", "test_key_placeholder")

@pytest.fixture
def mock_llm_response() -> MagicMock:
  response = MagicMock()
  response.choices[0].message.content = "Mock LLM response"
  return response

@pytest.fixture
def mock_openai_client(mock_llm_response: MagicMock) -> MagicMock:
  client = MagicMock()
  client.chat.completions.create.return_value = mock_llm_response
  return client

@pytest.fixture
def mock_vectordb() -> MagicMock:
  doc = MagicMock()
  doc.page_content = "Sample retrieved content"
  doc.metadata = {"source": "test.pdf", "page": 1}
  db = MagicMock()
  db.similarity_search.return_value = [doc]
  return db