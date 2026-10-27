import os
import shutil
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