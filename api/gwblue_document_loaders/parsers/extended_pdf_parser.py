import io
import logging
from pathlib import PurePath
from typing import Any, Iterator, Literal, Optional, Union, cast
import numpy as np
from PIL import Image
from langchain_core.documents.base import Blob
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_core.document_loaders.base import BaseBlobParser
from pypdf import PdfReader
from pypdf.generic import DictionaryObject

logger = logging.getLogger(__name__)


_FORMAT_IMAGE_STR = "{image_text}"
_JOIN_IMAGES = "\n"

_PDF_FILTER_WITH_LOSS = {"DCTDecode", "JPXDecode"}
_PDF_FILTER_WITHOUT_LOSS = {"FlateDecode", "LZWDecode", "CCITTFaxDecode"}


def _format_inner_image(blob: Blob, body: str, inner_format: str) -> str:
    if inner_format == "text":
        return body
    elif inner_format == "markdown-img":
        return f"![{body}](#)"
    elif inner_format == "html-img":
        return f'<img alt="{body}" src="#"/>'
    return body


def _merge_text_and_extras(extras: list[str], page_text: str) -> str:
    return page_text + "\n" + "\n".join(extras)

class SinglePixelImageError(TypeError):
    """Indicates a single-pixel (1,1,1) image was encountered and is unsupported."""
    pass

class ExtendedPyPDFParser(BaseBlobParser):
    """A forked version of the original PyPDFParser, 
    with enhanced Pillow support.
    """

    def __init__(
        self,
        password: Optional[Union[str, bytes]] = None,
        extract_images: bool = False,
        *,
        mode: Literal["single", "page"] = "page",
        pages_delimiter: str = "\n\f",
        images_parser: Optional[BaseBlobParser] = None,
        images_inner_format: Literal["text", "markdown-img", "html-img"] = "text",
        extraction_mode: Literal["plain", "layout"] = "plain",
        extraction_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.password = password
        self.extract_images = extract_images
        self.images_parser = images_parser
        self.images_inner_format = images_inner_format
        self.mode = mode
        self.pages_delimiter = pages_delimiter
        self.extraction_mode = extraction_mode
        self.extraction_kwargs = extraction_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Mimics the original lazy_parse, but uses pypdf internally."""
        import pypdf

        def _extract_text_from_page(page: pypdf.PageObject) -> str:
            if pypdf.__version__.startswith("3"):
                return page.extract_text()
            else:
                return page.extract_text(
                    extraction_mode=self.extraction_mode,
                    **self.extraction_kwargs,
                )

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
            single_texts = []
            total_pages = len(pdf_reader.pages)

            for page_number, page in enumerate(pdf_reader.pages):
                text_from_page = _extract_text_from_page(page)
                images_from_page = self.extract_images_from_page(page)
                merged = _merge_text_and_extras([images_from_page], text_from_page)
                if self.mode == "page":
                    yield Document(
                        page_content=merged.strip(),
                        metadata={
                            "source": blob.source,
                            "total_pages": total_pages,
                            "page_number": page_number,
                        },
                    )
                else:
                    single_texts.append(merged.strip())

            if self.mode == "single":
                yield Document(
                    page_content=self.pages_delimiter.join(single_texts),
                    metadata={"source": blob.source, "total_pages": total_pages},
                )

    def extract_images_from_page(self, page: Any) -> str:
        if not self.extract_images or not self.images_parser:
            return ""

        resources_ref = page.get("/Resources")
        if not resources_ref:
            return ""

        resources = resources_ref.get_object()
        if not isinstance(resources, DictionaryObject):
            return ""
        
        if "/XObject" not in resources:
            return ""
        
        xObject_ref = resources["/XObject"]
        xObject = xObject_ref.get_object()
        if not isinstance(xObject, DictionaryObject):
            return ""

        images = []
        for obj_key in xObject:
            xobj_dict = xObject[obj_key].get_object()
            subtype = xobj_dict.get("/Subtype", None)
            if subtype == "/Image":
                pdf_filter = xobj_dict.get("/Filter")
                if not pdf_filter:
                    continue
                if isinstance(pdf_filter, list):
                    pdf_filter = pdf_filter[-1]

                filter_name = pdf_filter[1:]

                if filter_name == "JBIG2Decode":
                    logger.warning("Skipping JBIG2 image (unsupported by pypdf).")
                    continue

                try:
                    raw_data = xobj_dict.get_data()
                except NotImplementedError as e:
                    logger.warning(f"Skipping image due to unsupported filter: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Generic decode error for image: {e}")
                    continue

                np_image = None
                if filter_name in _PDF_FILTER_WITHOUT_LOSS:
                    height = xobj_dict["/Height"]
                    width = xobj_dict["/Width"]
                    try:
                        np_image = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                            height, width, -1
                        )
                    except ValueError:
                        logger.warning("Shape mismatch, skipping image.")
                        continue
                elif filter_name in _PDF_FILTER_WITH_LOSS:
                    try:
                        np_image = np.array(Image.open(io.BytesIO(raw_data)))
                    except Exception as e:
                        logger.warning(f"Skipping image decode error: {e}")
                        continue
                else:
                    logger.warning(f"Unknown PDF filter: {filter_name}")
                    continue

                if np_image is not None:
                    image_bytes = io.BytesIO()
                    try:
                        Image.fromarray(np_image).save(image_bytes, format="PNG")
                    except TypeError as e:
                        if "Cannot handle this data type: (1, 1, 1)" in str(e):
                            # raise SinglePixelImageError(
                            #     "Found a single-pixel image of shape (1,1,1). Cannot handle this image."
                            # ) from e
                            continue

                        raise

                    blob = Blob.from_data(image_bytes.getvalue(), mime_type="image/png")
                    image_text = next(self.images_parser.lazy_parse(blob)).page_content
                    images.append(_format_inner_image(blob, image_text, self.images_inner_format))

        return _FORMAT_IMAGE_STR.format(
            image_text=_JOIN_IMAGES.join(filter(None, images))
        )