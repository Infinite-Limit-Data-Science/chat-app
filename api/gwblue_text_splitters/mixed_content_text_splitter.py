import re
from typing import List, Callable, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document


def _merge_contextless_chunks(
    text: str, token_len_func: Callable[[str], int], min_paragraph_tokens: int = 50
) -> str:
    paragraphs = text.split("\n\n")
    merged_paragraphs = []

    current = []
    current_tokens = 0

    for p in paragraphs:
        p_tokens = token_len_func(p)
        if current_tokens + p_tokens < min_paragraph_tokens:
            current.append(p)
            current_tokens += p_tokens
        else:
            if current:
                merged_paragraphs.append("\n\n".join(current))
            current = [p]
            current_tokens = p_tokens

    if current:
        merged_paragraphs.append("\n\n".join(current))

    return "\n\n".join(merged_paragraphs)


class MixedContentTextSplitter(TextSplitter):
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        img_pattern: str = r'(<img[^>]*src="[^"]+"[^>]*>)',
        metadata: Dict[str, Any] = {},
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

        self.img_regex = re.compile(img_pattern, flags=re.IGNORECASE)
        self.metadata = metadata

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        output_docs = []
        for doc in docs:
            new_meta = {
                k: v
                for k, v in doc.metadata.items()
                if k not in ("producer", "creator", "creationdate")
            }
            new_meta = {**new_meta, **self.metadata}

            if "source" in new_meta:
                from pathlib import Path

                new_meta["source"] = Path(new_meta["source"]).name

            chunks = self.split_text(doc.page_content)

            for chunk in chunks:
                if self.img_regex.match(chunk):
                    match = re.search(r'src="([^"]+)"', chunk, flags=re.IGNORECASE)
                    if match:
                        chunk = match.group(1)
                    output_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={**new_meta, "chunk_type": "image"},
                        )
                    )
                else:
                    if page_number := new_meta.get("page", None):
                        page_info = f"Page {page_number}. "
                        chunk = page_info + chunk
                    output_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={**new_meta, "chunk_type": "text"},
                        )
                    )
        return output_docs

    def split_text(self, text: str) -> List[str]:
        parts = self.img_regex.split(text)
        final_chunks = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if self.img_regex.match(part):
                final_chunks.append(part)
            else:
                text_subchunks = self.text_splitter.split_text(part)
                final_chunks.extend(text_subchunks)
        return final_chunks
