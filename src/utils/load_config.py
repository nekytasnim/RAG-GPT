import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pyprojroot import here

load_dotenv()

class LoadConfig:
  def __init__(self) -> None:
    with open(here("configs/app_config.yml")) as cfg:
      app_config = yaml.load(cfg, yaml.FullLoader())

    # LLM configs
    self.llm_engine: str = app_config['llm_config']['engine']
    self.llm_system_role: str = app_config['llm_config']['llm_system_role']
    # str() required: Chroma DB backend concatenates thsi path internally
    self.persist_directory: str = app_config['directory']['persist_directory']

    # Cloudflare R2 configs
    self.cloudflare_bucket_name = app_config['cloudflare']['bucket_name']
    self.cloudflare_processed_prefix = app_config['cloudflare']['processed_prefix']
    self.cloudflare_uploads_prefix = app_config['cloudflare']['uploads_prefix']
    self.cloudflare_account_id = os.getenv('CLOUDFLARE_R2_ACCOUNT_ID')
    self.cloudflare_access_key = os.getenv('CLOUDFLARE_R2_ACCESS_KEY_ID')
    self.cloudflare_secret_access_key = os.getenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')

    # Retrieval configs
    self.k = app_config['retrieval_config']['k']
    self.embedding_model_engine = app_config['embedding_model_config']['engine']
    self.chunk_size = app_config['splitter_config']['chunk_size']
    self.chunk_overlap = app_config['splitter_config']['chunk_overlap']

    self.openai_client = OpenAI(
      api_key = os.getenv('OPENROUTER_API_KEY'),
      base_url = "https://openrouter.ai/api/v1"
    )

    self.openai_embeddings = OpenAIEmbeddings(
      model=self.embedding_model_engine,
      openai_api_key=os.getenv('OPENROUTER_API_KEY'),
      openai_api_base='https://openrouter.ai/api/v1',
      check_embedding_ctx_length=False
    )

    # Summarizer config
    self.max_final_token = app_config['summarizer_config']['max_final_token']
    self.character_overlap = app_config['summarizer_config']['character_overlap']
    self.token_threshold = app_config['summarizer_config']['token_threshold']
    self.summarizer_llm_system_role = app_config['summarizer_config']['summarizer_llm_system_role']
    self.final_summarizer_llm_system_role = app_config['summarizer_config']['final_summarizer_llm_system_role']
    self.temperature = app_config['llm_config']['temperature']

    # Memory
    self.number_of_q_a_pairs = app_config['memory']['number_of_q_a_pairs']