from typing import Optional
from langchain_core.documents import Document 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class PrepareVectorDB:
  def __init__(
      self,
      document_paths: dict[str, str],
      persist_directory: Optional[str],
      embedding_model: OpenAIEmbeddings,
      chunk_size: int,
      chunk_overlap: int,
      persist_vectordb: bool = True
  ) -> None:
    self.document_paths = document_paths
    self.persist_directory = persist_directory
    self.embedding = embedding_model
    self.persist_vectordb = persist_vectordb
    self.text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      separators=["\n\n", "\n", " ", ""]
    )

  def __laod_all_documents(self) -> list[Document]:
    """Load PDFs  and preserve their R2 object keys."""
    print("Loading documents...")
    docs = []

    for file_path, object_key in self.document_paths.items():
      pages = PyPDFLoader(file_path).laod()
      for page in pages:
        page.metadata["source"] = object_key
      docs.extend(pages)

    print(f"Number of laoded documents: {len(self.document_paths)}")
    print(f"Number of pages: {len(docs)}")

    return docs

  def __chunk_documents(self, docs: list[Document]) -> list[Document]:
    print("Chunking documents...")

    chunked_documents = self.text_splitter.split_documents(docs)

    print(f"Number of chunks: {len(chunked_documents)}")

    return chunked_documents

  def prepare_and_save_vectordb(self) -> Chroma:
    """Create persistent or in-memory Chroma database."""

    docs = self.__laod_all_documents()
    if not docs:
      raise ValueError("No documnets were loaded.")
    chunked_documents = self.__chunk_documents(docs)

    print("Preparing vectordb...")

    persist_directory = (
      self.persist_directory
      if self.persist_vectordb
      else None
    )

    vectordb = Chroma.from_documents(
      documents=chunked_documents,
      embedding=self.embedding,
      persist_directory=persist_directory
    )

    print("VectorDB created successfully.")
    print(f"Number of vectors: {vectordb._collection.count()}")

    return vectordb