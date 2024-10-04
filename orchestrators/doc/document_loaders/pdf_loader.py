import logging
import re
import numpy as np
import pypdf
from enum import Enum, auto
from typing import Iterator, Union, Dict, Optional
from pypdf import PdfReader, PageObject
from pypdf.generic import ContentStream
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import BasePDFLoader
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob

_BYTE_FOR_BYTE_MAPPING = 'latin1'

_PDF_FILTER_WITH_LOSS = ['DCTDecode', 'DCT', 'JPXDecode']

_PDF_FILTER_WITHOUT_LOSS = [
    'LZWDecode',
    'LZW',
    'FlateDecode',
    'Fl',
    'ASCII85Decode',
    'A85',
    'ASCIIHexDecode',
    'AHx',
    'RunLengthDecode',
    'RL',
    'CCITTFaxDecode',
    'CCF',
    'JBIG2Decode',
]

class PdfTrailor(Enum):
    """metadata of PDF Trailor Dictionary"""
    CROSS_REFERENCE_TABLE = auto()
    INFO = auto()
    ID = auto()
    ENCRYPTION = auto()

class PyPDFImageParser(BaseBlobParser):
    def __init__(
            self, 
            file_path: str, 
            extract_images: bool = True, 
            password: Union[None, str, bytes] = None, 
            extraction_kwargs: Optional[Dict] = None):
        self._file_path = file_path
        self._reader = PdfReader(self._file_path)
        self._extract_images = extract_images
        self._password = password
        self._extraction_kwargs = extraction_kwargs

    @property
    def _table_extract(self) -> str:        
        for page_num in range(len(self._reader.pages)):
            page = self._reader.pages[page_num]
            content_stream = page.get_contents()
            
            operations = self._operators_from_stream(content_stream)
            
            if self._tabular(operations):
                return 'layout'
        
        return 'plain'

    def _operators_from_stream(self, content_stream: ContentStream) -> list[tuple[str, tuple[float, float]]]:
        """Extract operators and their operands from the content stream"""
        stream_data = content_stream.get_data()
        stream_text = stream_data.decode(_BYTE_FOR_BYTE_MAPPING)
        
        tm_pattern = re.compile(r"(?P<a>-?\d+\.\d+)\s+(?P<b>-?\d+\.\d+)\s+(?P<c>-?\d+\.\d+)\s+(?P<d>-?\d+\.\d+)\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+Tm")
        td_pattern = re.compile(r"(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+Td")

        tm_matches = tm_pattern.findall(stream_text)
        td_matches = td_pattern.findall(stream_text)
        
        operations = [('Tm', (float(m[4]), float(m[5]))) for m in tm_matches] + \
                    [('Td', (float(m[0]), float(m[1]))) for m in td_matches]
        
        return operations

    def _tabular(self, operations: list[tuple[str, tuple[float, float]]]):
        """Analyze the operations"""
        h_pos = set()
        v_gaps = []
        
        for _, (x_position, y_position) in operations:
            h_pos.add(x_position)
            
            if v_gaps and abs(y_position - v_gaps[-1]) < 15:
                return True
            
            v_gaps.append(y_position)
            
            if len(h_pos) > 1:
                return True
        
        return False

    def _extract_images_from_page(self, page: pypdf._page.PageObject) -> str:
        """Extract images from page and get the text with internal LLM image-to-text multimodal"""
        if not self._extract_images or '/XObject' not in page['/Resources'].keys():
            return ''

        xObject = page['/Resources']['/XObject'].get_object()
        images = []
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                if xObject[obj]["/Filter"][1:] in _PDF_FILTER_WITHOUT_LOSS:
                    height, width = xObject[obj]['/Height'], xObject[obj]['/Width']

                    images.append(
                        np.frombuffer(xObject[obj].get_data(), dtype=np.uint8).reshape(
                            height, width, -1
                        )
                    )
                elif xObject[obj]['/Filter'][1:] in _PDF_FILTER_WITH_LOSS:
                    images.append(xObject[obj].get_data())
                else:
                    logging.warning('Unknown PDF Filter!')
        logging.warning(f'THE IMAGES {str(images)}')
        return ''

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Implementation of abstract method `lazy_parse`"""
        def _extract_text_from_page(page: "PageObject") -> str:
            return page.extract_text(
                extraction_mode=self._table_extract
            )

        pages = self._reader.pages
        count = len(pages)
        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = PdfReader(pdf_file_obj, password=self._password)

            yield from [
                Document(
                    page_content=_extract_text_from_page(page=page)
                    + self._extract_images_from_page(page),
                    metadata={'source': blob.source, 'page': page_number, 'total_pages': count},
                )
                for page_number, page in enumerate(pdf_reader.pages, start=1)
            ]

class PyPDFImageLoader(BasePDFLoader):
    """Use internal LLM only for image extract"""
    def __init__(self, file_path: str, password: Union[None, str, bytes] = None, extraction_kwargs: Optional[Dict] = None) -> None:
        super().__init__(file_path, headers=None)
        self.parser = PyPDFImageParser(
            file_path=file_path,
            extract_images=True,
            password=password,
            extraction_kwargs=extraction_kwargs,
        )

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)