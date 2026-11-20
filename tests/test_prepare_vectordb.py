from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

def _make_doc(content: str = "doc content") -> Document:
  return Document(
    page_content=content,
    metadata={"source": "test.pdf", "page": 0}
  )

def test_prepare_and_save_vectordb_calls_from_documents(
    tmp_path,
    mock_vectordb: MagicMock
) -> None:
  """Chroma recieves chunked documents with their R2 metadata"""

  from utils.prepare_vectordb import PrepareVectorDB

  fake_docs = [_make_doc("page 1"), _make_doc("page 2")]
  mock_vectordb._collection.count.return_value = 2
  mock_embeddings = MagicMock()

  with patch("utils.prepare_vectordb.PyPDFLoader") as mock_loader, \
       patch("utils.prepare_vectordb.Chroma") as mock_chroma:
    mock_loader.return_value.load.return_value = fake_docs
    mock_chroma.from_documents.return_value = mock_vectordb
    instance = PrepareVectorDB(
      document_paths={"fake.pdf": "pre-processed/fake.pdf"},
      persist_directory=str(tmp_path / "vectordb"),
      embedding_model=mock_embeddings,
      chunk_size=1500,
      chunk_overlap=500
    )
    result = instance.prepare_and_save_vectordb()

  mock_loader.assert_called_once_with("fake.pdf")
  mock_chroma.from_documents.assert_called_once()

  call_kwargs = mock_chroma.from_documents.call_args.kwargs

  assert "documents" in call_kwargs
  assert call_kwargs["embedding"] is mock_embeddings
  assert call_kwargs["persist_directory"] == str(tmp_path / "vectordb")

  chunked_docs = call_kwargs["documents"]

  assert len(chunked_docs) > 0
  for doc in chunked_docs:
    assert doc.metadata["source"] == "pre-processed/fake.pdf"

  assert result is mock_vectordb

def test_chunk_sizes_applied_from_config() -> None:
  """Text splitter uses the supllied chunk configuration."""

  from utils.prepare_vectordb import PrepareVectorDB

  instance = PrepareVectorDB(
    document_paths={},
    persist_directory="/some/vetcordb",
    embedding_model=MagicMock(),
    chunk_size=800,
    chunk_overlap=200
  )

  assert instance.text_splitter._chunk_size == 800
  assert instance.text_splitter._chunk_overlap == 200

