#### Chunking Strategy: Hugging Face Transformers 

Most modern LLMs (e.g., GPT, BERT, etc.) use subword tokenization techniques (e.g., Byte-Pair Encoding, WordPiece). Instead of splitting text strictly by white space or punctuation, subword tokenizers learn a vocabulary of frequently appearing word fragments. For example, the word “unbelievable” may be split into tokens like: \text{"un"}, \quad \text{"##bel"}, \quad \text{"##ievable"}. This helps the model handle unknown words and morphological variations more gracefully than a naive word-level or character-level approach.

When you first instantiate the Transformer (or any neural network), you allocate memory for all trainable parameters—including the token and positional embedding tables. These tables are usually randomly initialized or loaded from a pretrained checkpoint. If you’re training a model from scratch, those initial token and positional embedding values are typically randomized and don’t carry any semantic meaning.  Over the course of training, **backpropagation** will adjust these vectors so that tokens with similar meanings end up with similar embedding vectors. By the end, the once-random embeddings become rich, learned representations that help the network minimize its loss. 

1. **Tokenization**  
   We tokenize the input text with `return_offsets_mapping=True`, giving us:
   - `input_ids`: numeric IDs of each token  
   - `offsets`: (start_char, end_char) for each token  
   - `n_tokens`: total number of tokens

2. **Chunking**  
   - We iterate over the token IDs in **chunks** of size up to `chunk_size`, but we try to backtrack up to `backtrack_window` tokens to find a sentence boundary (`.` or `\n`) so we don’t break in the middle of a sentence.
   - We maintain a *pending_chunk_ids* so that after we finalize one chunk, we don’t immediately yield it if there may be leftover overlap needed.

3. **Overlap on Final Chunk**  
   - If the *final chunk* is smaller than `chunk_size`, we attempt to “pull in” tokens from the previous chunk so the final chunk isn’t too small.

4. **Yield**  
   - Each chunk is `decode`d via `tokenizer.decode(...)`.
   - Streams out chunk by chunk until no more tokens remain.

This ensures a more **natural boundary** for text chunks, helps avoid mid-sentence splits, and prevents a very tiny leftover chunk at the end. 