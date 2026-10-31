import pytest
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

@pytest.fixture
def cfg() -> None:
  from utils.load_config import LoadConfig
  return LoadConfig()

def test_llm_engine_is_gpt_oss_120b(cfg) -> None:
  assert cfg.llm_engine == "openai/gpt-oss-120b:free"

def test_openrouter_client_configured(cfg) -> None:
  assert isinstance(cfg.openai_client, OpenAI)
  assert str(cfg.openai_client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"

def test_embedding_model_configured(cfg) -> None:
  assert isinstance(cfg.openai_embeddings, OpenAIEmbeddings)
  assert cfg.openai_embeddings.model == cfg.embedding_model_engine

def test_embedding_openrouter_endpoint(cfg) -> None:
  assert str(cfg.openai_embeddings.openai_api_base).rstrip("/") == "https://openrouter.ai/api/v1"

def test_retrieval_k_loaded(cfg) -> None:
  assert isinstance(cfg.k, int) and cfg.k > 0

def test_chunk_config_loaded(cfg) -> None:
  assert isinstance(cfg.chunk_size, int) and cfg.chunk_size > 0
  assert isinstance(cfg.chunk_overlap, int) and cfg.chunk_overlap > 0

def test_memory_pairs_loaded(cfg) -> None:
  assert isinstance(cfg.number_of_q_a_pairs, int) and cfg.number_of_q_a_pairs > 0