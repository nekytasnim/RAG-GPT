import tempfile
import os
from serve import R2Storage
from utils.load_config import LoadConfig
from utils.prepare_vectordb import PrepareVectorDB

APPCFG = LoadConfig()

def build_processed_vectordb() -> None:
  storage = R2Storage(bucket_name=APPCFG.cloudflare_bucket_name)

  with tempfile.TemporaryDirectory() as temp_dir:
    document_paths = storage.download_preprocessed_files(temp_dir)

    prepare_vectordb_instance = PrepareVectorDB(
      document_paths=document_paths,
      persist_directory=APPCFG.persist_directory,
      embedding_model=APPCFG.openai_embeddings,
      chunk_size=APPCFG.chunk_size,
      chunk_overlap= APPCFG.chunk_overlap,
      persist_vectordb=True
    )

    db_file = os.path.join(APPCFG.persist_directory, "chroma.sqlite3")

    if not os.path.exists(db_file):
      prepare_vectordb_instance.prepare_and_save_vectordb()
    else:
      print(f"VectorDB already exists in {APPCFG.persist_directory}")

    return None

if __name__ == "__main__":
  build_processed_vectordb()

  