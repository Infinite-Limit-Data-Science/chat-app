import pytest
from pathlib import Path
from ..loaders.extended_pypdf_loader import ExtendedPyPDFLoader
from ..parsers.base64_blob_parser import Base64BlobParser

assets_dir = Path(__file__).parent / "assets"

@pytest.fixture
def genesys_contract_pdf_path() -> Path:
    return assets_dir / "64654-genesys.pdf"

@pytest.mark.asyncio
async def test_genesys_contract_doc_prompt(
    genesys_contract_pdf_path: Path,
):
    loader = ExtendedPyPDFLoader(
        genesys_contract_pdf_path,
        extract_images=True,
        images_parser=Base64BlobParser(),
        images_inner_format="raw",
        mode="page",
    )

    docs_stream = loader.lazy_load()
    docs = list(docs_stream)
    assert len(docs) > 0