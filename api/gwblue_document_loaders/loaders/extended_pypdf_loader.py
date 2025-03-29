from pathlib import PurePath
from typing import Any, Iterator, Literal, Optional, Union
from langchain_core.documents.base import Blob
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import BasePDFLoader
from langchain_community.document_loaders.base import BaseBlobParser
from ..parsers.extended_pdf_parser import ExtendedPyPDFParser

class ExtendedPyPDFLoader(BasePDFLoader):
    """
    If mode is page, the loader will stream page by page
    else stream content in a single block
    """
    def __init__(
        self,
        file_path: Union[str, PurePath],
        password: Optional[Union[str, bytes]] = None,
        headers: Optional[dict] = None,
        extract_images: bool = False,
        *,
        mode: Literal["single", "page"] = "page",
        images_parser: Optional[BaseBlobParser] = None,
        images_inner_format: Literal["text", "markdown-img", "html-img"] = "text",
        pages_delimiter: str = "\n\f",
        extraction_mode: Literal["plain", "layout"] = "plain",
        extraction_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(file_path, headers=headers)
        self.parser = ExtendedPyPDFParser(
            password=password,
            mode=mode,
            extract_images=extract_images,
            images_parser=images_parser,
            images_inner_format=images_inner_format,
            pages_delimiter=pages_delimiter,
            extraction_mode=extraction_mode,
            extraction_kwargs=extraction_kwargs,
        )

    def lazy_load(self) -> Iterator[Document]:
        blob = Blob.from_path(self.file_path)
        yield from self.parser.lazy_parse(blob)