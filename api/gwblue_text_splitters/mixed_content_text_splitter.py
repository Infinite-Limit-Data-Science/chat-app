import re
from typing import List, Callable, Dict, Any, Iterator
from langchain_core.documents import Document
from pathlib import Path
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from .streaming_text_splitter import StreamingTextSplitter
"""
CharacterTextSplitter drawbacks:
- Simply delimits by separater, with option of adding delimiter to beginning or ending of chunk
    - If single separater is "", then just returns list of characters
- Niave merge splits where it appends multiple smaller splits into one chunk up to a size limit
    - if separater never reached, exceeds chunk_size, potentially losing surplas tokens in embeddings
- Does not support streaming results

RecursiveCharacterTextSplitter drawbacks:
- recursively subdivides text until each resulting chunk is at or below the desired size
    - potential for very small chunks, if one chunk is within size limit but next chunk is too large
    - length_function using tokenizers for long documents is extremely slow
- Does not support streaming results

repeated length_function invocation is expensive. Conversely, tokenizing entire text and then splitting
by token ids is challenging as the representation of a newline or period character can be embodied by 
multiple token ids in the same vocabulary. 

".", "\n", "\n\n", " " may not be represented as individual tokens in the model vocabulary. So
`PreTrainedTokenizerBase.tokenize(".")` might yield token ID 12345 for ".". But in actual text, 
that period could appear as "Hello." within the same token, e.g. ["Hello."], which might be ID 99999.
So the search for ID 12345 (the standalone period) might never succeed if the period nearly always 
merges with preceding characters in real text.

In the fast (Rust-based) tokenizers from Hugging Face, each encoded input has detailed alignment 
between tokens and their original text substrings. Specifically, the BatchEncoding object includes:
- encoding.tokens(): a list of the actual tokens (subword pieces) as strings.
- encoding.offset_mapping: a list of (start_offset, end_offset) pairs, telling you which slice of 
the original input text corresponds to each token.

Hence, for each token i, you can retrieve its text span in the original input by looking at 
offset_mapping[i]. In a typical “fast” tokenizer, this data is attached to an internal Encoding 
object (one per sequence).

```python
output = tokenizer("Hello. How are you?", return_offsets_mapping=True)
print("Tokens:", output.tokens())  
Tokens: ['Hello.', 'How', 'are', 'you', '?']
print("Offsets:", output.offset_mapping)
Offsets: [(0, 6), (7, 10), (11, 14), (15, 18), (18, 19)]
```

The Adaptive Token Boundary algorithm will not only chunk data as close to the chunk size as possible,
ensuring a proper termination, such as a newline or period to indicate end of context, but if multiple 
chunks must be split because they exceed chunk_size, for example, a chunk size of 500 tokens but pdf
page has 551 tokens, it will backtrack to the last terminator, such as a newline at token 498, and then
the remainder 2 tokens will be appended to the 51 tokens that were divided, now leaving a remainder a 53 
tokens, representing the remaining part of the page, even in this case, the page number will still be 
appended to those 53 tokens to preserve the page context of the "last part" of the page. All this is 
happening in a streaming fashion so the document can be infinitely long in size.

Must include leftover check: avoid small leftover chunks
"""

def _adaptive_token_boundary_stream(
    text: str,
    tokenizer,
    chunk_size: int,
    backtrack_window: int = 30,
):
    if isinstance(tokenizer, PreTrainedTokenizer):
        print("Using PreTrainedTokenizer, consider using PreTrainedTokenizerFast for offsets.")

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    tokens = encoding.tokens()
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    n_tokens = len(input_ids)

    if n_tokens <= chunk_size:
        yield text
        return

    pending_chunk_ids = None

    start_idx = 0

    while True:
        remainder = n_tokens - start_idx

        if remainder <= chunk_size:
            final_chunk_ids = input_ids[start_idx : n_tokens]

            if pending_chunk_ids is not None:
                final_size = len(final_chunk_ids)
                if final_size < chunk_size:
                    needed = chunk_size - final_size
                    can_overlap = len(pending_chunk_ids)
                    overlap_amount = min(needed, can_overlap)

                    if overlap_amount > 0:
                        overlap_slice = pending_chunk_ids[-overlap_amount:]
                        final_chunk_ids = overlap_slice + final_chunk_ids

            if pending_chunk_ids is not None:
                yield tokenizer.decode(pending_chunk_ids)

            yield tokenizer.decode(final_chunk_ids)
            return

        proposed_end = min(start_idx + chunk_size, n_tokens)
        best_boundary = proposed_end

        for j in range(proposed_end, max(start_idx, proposed_end - backtrack_window), -1):
            if j <= start_idx:
                break
            start_char, end_char = offsets[j - 1]
            substring = text[start_char:end_char]
            if substring.endswith(".") or substring.endswith("\n"):
                best_boundary = j
                break

        current_chunk_ids = input_ids[start_idx:best_boundary]
        start_idx = best_boundary

        if pending_chunk_ids is not None:
            yield tokenizer.decode(pending_chunk_ids)

        pending_chunk_ids = current_chunk_ids

        if start_idx >= n_tokens:
            yield tokenizer.decode(pending_chunk_ids)
            return

class MixedContentTextSplitter(StreamingTextSplitter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        chunk_size: int = 2000,
        img_pattern: str = r'(<img[^>]*src="[^"]+"[^>]*>)',
        metadata: Dict[str, Any] = None,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.metadata = metadata or {}
        self.img_regex = re.compile(img_pattern, flags=re.IGNORECASE)

    def split_documents(
        self,
        docs: Iterator[Document],
    ) -> Iterator[Document]:
        for doc in docs:
            merged_meta = {**doc.metadata, **self.metadata}
            if "source" in merged_meta:
                merged_meta["source"] = Path(merged_meta["source"]).name

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
                    for text_chunk in _adaptive_token_boundary_stream(
                        part, 
                        tokenizer=self.tokenizer, 
                        chunk_size=self.chunk_size,
                    ):
                        if page_number:
                            pnum = int(page_number)
                            page_label = f"Page {pnum+1}"
                            if page_label not in text_chunk:
                                text_chunk = f"{page_label} {text_chunk}"

                        yield Document(
                            page_content=text_chunk,
                            metadata={**merged_meta, "chunk_type": "text"},
                        )