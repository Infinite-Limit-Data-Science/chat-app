from typing import (
    Dict,
    Any,
)
import pytest
import uuid
import time
from pathlib import Path
from ..mixed_content_text_splitter import MixedContentTextSplitter
from ...gwblue_huggingface.huggingface_transformer_tokenizers import (
    get_tokenizer_by_prefix,
    BaseLocalTokenizer,
)
from ...gwblue_document_loaders.loaders.extended_pypdf_loader import ExtendedPyPDFLoader
from ...gwblue_document_loaders.parsers.base64_blob_parser import Base64BlobParser

assets_dir = Path(__file__).parent / "assets"

@pytest.fixture
def teams_to_consider_word_path() -> Path:
    return assets_dir / "Teams to Consider.docx"

@pytest.fixture
def genesys_contract_pdf_path() -> Path:
    return assets_dir / "64654-genesys.pdf"

@pytest.fixture
def arag_ignite_pdf_path() -> Path:
    return assets_dir / "ARAG Ignite 2025 Flier 1.pdf"

@pytest.fixture
def calculus_book1_path() -> Path:
    return assets_dir / "CalculusBook1.pdf"

@pytest.fixture
def jpeg_pdf_path() -> Path:
    return assets_dir / "jpeg.pdf"

@pytest.fixture
def message_metadata() -> Dict[str, Any]:
    return {
        "uuid": uuid.uuid4(),
        "conversation_id": uuid.uuid4(),
    }

@pytest.fixture
def vlm_tokenizer() -> BaseLocalTokenizer:
    return get_tokenizer_by_prefix("TIGER-Lab/VLM2Vec-Full")

def test_calculus_book_split_by_500(
    message_metadata: Dict[str, Any],
    calculus_book1_path: Path,
    vlm_tokenizer,
):
    metadata = {
        **message_metadata,
        "source": "calculus_book1_path",
    }

    loader = ExtendedPyPDFLoader(
        calculus_book1_path,
        extract_images=True,
        images_parser=Base64BlobParser(),
        images_inner_format="raw",
        mode="page",
    )

    docs = loader.lazy_load()

    text_splitter = MixedContentTextSplitter(
        vlm_tokenizer.tokenizer,
        chunk_size=500,
        metadata=metadata,
    )

    chunks = text_splitter.split_documents(docs)
    
    total = 0
    start_time = time.time()
    for chunk in chunks:
        total += 1
        print(chunk)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Number of chunks: {total}")
    print(f"Time elapsed: {elapsed:.2f} seconds")

    assert total > 0

def test_calculus_book_split_by_250(
    message_metadata: Dict[str, Any],
    calculus_book1_path: Path,
    vlm_tokenizer,
):
    metadata = {
        **message_metadata,
        "source": "calculus_book1_path",
    }

    loader = ExtendedPyPDFLoader(
        calculus_book1_path,
        extract_images=True,
        images_parser=Base64BlobParser(),
        images_inner_format="raw",
        mode="page",
    )

    docs = loader.lazy_load()

    text_splitter = MixedContentTextSplitter(
        vlm_tokenizer.tokenizer,
        chunk_size=250,
        metadata=metadata,
    )

    chunks = text_splitter.split_documents(docs)
    
    total = 0
    start_time = time.time()
    for chunk in chunks:
        total += 1
        print(chunk)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Number of chunks: {total}")
    print(f"Time elapsed: {elapsed:.2f} seconds")

    assert total > 0

def test_calculus_book_split_by_2000(
    message_metadata: Dict[str, Any],
    calculus_book1_path: Path,
    vlm_tokenizer,
):
    metadata = {
        **message_metadata,
        "source": "calculus_book1_path",
    }

    loader = ExtendedPyPDFLoader(
        calculus_book1_path,
        extract_images=True,
        images_parser=Base64BlobParser(),
        images_inner_format="raw",
        mode="page",
    )

    docs = loader.lazy_load()

    text_splitter = MixedContentTextSplitter(
        vlm_tokenizer.tokenizer,
        chunk_size=2000,
        metadata=metadata,
    )

    chunks = text_splitter.split_documents(docs)
    
    total = 0
    start_time = time.time()
    for chunk in chunks:
        total += 1
        print(chunk)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Number of chunks: {total}")
    print(f"Time elapsed: {elapsed:.2f} seconds")

    assert total > 0