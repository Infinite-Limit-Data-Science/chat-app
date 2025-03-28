import re
from typing import List, Callable, Dict, Any, Iterator
from langchain_core.documents import Document

def _token_based_split_text_stream(
    text: str,
    *,
    encode_fn: Callable[[str], List[int]],
    decode_fn: Callable[[List[int]], str],
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[str]:
    tokens = encode_fn(text)
    stride = chunk_size - chunk_overlap if chunk_size > chunk_overlap else chunk_size
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        yield decode_fn(chunk_tokens)

        start += stride

class MixedContentTextSplitter:
    def __init__(
        self,
        *,
        encode_fn: Callable[[str], List[int]],
        decode_fn: Callable[[List[int]], str],
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        img_pattern: str = r'(<img[^>]*src="[^"]+"[^>]*>)',
        metadata: Dict[str, Any] = None,
    ):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.metadata = metadata or {}
        self.img_regex = re.compile(img_pattern, flags=re.IGNORECASE)

    def split_documents_stream(
        self,
        docs: Iterator[Document],
    ) -> Iterator[Document]:
        for doc in docs:
            merged_meta = {**doc.metadata, **self.metadata}
            page_number = merged_meta.get("page_number")

            parts = self.img_regex.split(doc.page_content)

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if self.img_regex.match(part):
                    src_match = re.search(r'src="([^"]+)"', part, flags=re.IGNORECASE)
                    img_src = src_match.group(1) if src_match else part

                    yield Document(
                        page_content=img_src,
                        metadata={**merged_meta, "chunk_type": "image"},
                    )
                else:
                    for text_chunk in _token_based_split_text_stream(
                        part,
                        encode_fn=self.encode_fn,
                        decode_fn=self.decode_fn,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    ):
                        if page_number:
                            text_chunk = f"Page {page_number}. {text_chunk}"
                        yield Document(
                            page_content=text_chunk,
                            metadata={**merged_meta, "chunk_type": "text"},
                        )