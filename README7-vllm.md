# https://docs.vllm.ai/en/latest/getting_started/examples/openai_embedding_client.html
# https://docs.vllm.ai/en/latest/getting_started/examples/openai_chat_embedding_client_for_multimodal.html !!!!!!!!!!!!!!!!!!!!!!!!!!!!

token=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
model=TIGER-Lab/VLM2Vec-Full
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$token" -p 8070:8000 --ipc=host vllm/vllm-openai:latest --model $model --trust-remote-code --task embed --uvicorn-log-level debug --tensor-parallel-size 1 --max-num-batched-tokens 24256 --max-num-seqs 65 --max-model-len 24256


docker run -d --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$token" -p 8070:8000 --ipc=host vllm/vllm-openai:latest --model $model --trust-remote-code --task embed --tensor-parallel-size 4
docker container run -d -p 6379:6379 --name redisearch redis/redis-stack-server:latest

 --tensor-parallel-size 4

### ulimit
sudo vi /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536

ulimit -a

 
 

Stochastic Gradient Descent (SGD) is a method of iterative fitting that works by repeatedly nudging the model’s parameters in the direction that reduces the training error (loss). Here’s a step-by-step outline of how it happens:
- Initialize Parameters
  - Start with random values for all the model’s parameters (e.g., weights in the token embedding table, position embedding table, the attention matrices, feed-forward layers, etc.).
  - If your Transformer uses learned (absolute) positional embeddings, then that table is also randomly initialized at the start of training, right alongside the token embeddings, attention matrices, feed-forward layers, etc. Each row in the [max_position,hidden_dim] positional embedding matrix starts with random values. As training proceeds (via SGD and backpropagation), those values are updated so the model learns how to encode positional order effectively.
- Select a Mini-Batch
  - Instead of using the entire dataset at once (which could be huge), you pick a small batch of training examples (e.g. 32 or 64 sentences).
- Compute Loss
  - For that mini-batch, you run the model forward:
    - Tokenize the input into token IDs.
    - Map IDs to embeddings.
    - Pass them through the Transformer layers (self-attention, feed-forward, etc.).
    - Compare the model’s predictions to the actual targets (like the next token in each sentence)
    - Calculate a loss measure (often cross-entropy).
    - This loss tells you how far off the model’s predictions are for that mini-batch.
- Compute Gradients (Backpropagation)
  - Use backprop to find, for each parameter, how changing that parameter slightly would affect the loss.
  - In other words, compute the partial derivative of the loss with respect to each weight.
- Update Parameters
  - “Descend” in the direction that lowers the loss the most.
- Repeat
  - Move on to the next mini-batch, compute the new loss, new gradients, update parameters again.
  - Over time (many epochs), the parameters converge toward values that minimize the overall loss on the training set.

In effect, we use Cross-Entropy Loss and Stochastic Gradient Descent (SGD) to learn the meaning of a word respect to its position in a sentence (so not just the meaning of the word itself but also its position in a sentence) and then we store that essence as a vector in a positional embedding table? Then we later add this stored vector in the positional embedding table with the vector stored in the token embedding table to find overall semantic meaning of word with respect to surrounding words? Essentially. We use cross-entropy loss + SGD to fine-tune those embedding vectors (and all other model parameters) across countless examples of different word orders. Over many training steps, the model learns general patterns of how word order impacts meaning. It doesn’t memorize each unique sentence permutation; it gradually adjusts the positional embeddings (and the rest of the network) to handle any order that shows up in the data.

The key here is that just like the token embedding table, the positional embedding table is storing a vector in an index of its table. This vector is comprised of hundreds or thousands of floats, depending on how many dimensions the vector has. It does not store words or integers representing a position. It stores vectors. So there’s no row for “the cat” or any other multi-word sequence. Each row is just a vector encoding “this is the i-th token in the sequence,” and it never stores actual word pairs.

They’re two different “lookup tables”—each one answers a different question:
- Token (Word) Embedding Table
  - Indexed by token ID (which word/subword in the vocabulary).
  - Says “What is the semantic meaning of token #X?”
  - Example shape: [vocab_size,hidden_dim].
  - Row 42 might correspond to the word “cat,” row 464 might be “the,” etc.
- Positional Embedding Table
  - Indexed by position (which spot in the sequence: 1st token, 2nd token, etc.).
  - Says “What does it mean to be in position #i in the sequence?”
  - Example shape: [max_position,hidden_dim].
  - Row 0 might correspond to “1st token,” row 1 to “2nd token,” and so on—no actual words here, just a “position vector.”

Why both are needed
- The token embedding tells the model which word each token is.
- The positional embedding tells the model where in the sequence that token occurs.

The Transformer then adds those two vectors together (token embedding + positional embedding) so each token knows both what it is and where it’s located in the sentence.

When you feed a sentence into the Transformer, each token is combined with a position:
- Token ID (e.g., “cat” → ID 1234) is looked up in the token embedding table → yields a vector like [0.51, -0.12, 1.33, ...].
- Position index (e.g., 2nd token in the sentence) is looked up in the positional embedding table → yields another vector like [0.09, 0.85, -1.02, ...].
The model then adds these two vectors—so the token’s final representation is something like: token_embedding("cat") + positional_embedding(for position #2). That merged vector says: “I’m the token ‘cat,’ and I’m the 2nd token in this sentence.”

Positional Embedding approaches:
There are several ways to encode positional information. The learned (absolute) positional embedding approach is just one of them. Some common alternatives include:
- Sinusoidal (Fixed) Positional Embeddings
  - First popularized in the original “Attention Is All You Need” Transformer paper (2017).
  - Instead of having a learned [max_position,hidden_dim] matrix, they use a fixed sin/cos function of the position index p.
  - No trainable parameters; the idea is that different positions get different sine-wave patterns in each dimension of the embedding.
- Relative Positional Encodings
  - Used in models like T5 and Transformer-XL.
  - The model encodes distance between pairs of tokens (e.g., “how far is token i from token j?”) rather than storing a separate embedding for each absolute position.  
- Rotary Positional Embeddings (RoPE)
  - Employed by models like GPT-NeoX, LLaMA, and others.
  - They multiply the hidden states by rotation matrices whose angles are functions of the token’s position.
  - No explicit table of learnable parameters for each position. Instead, you configure a “base frequency,” and the rotation handles positions of any length (though typically truncated at a max position in practice).


Summary:

Transformers are permutation-invariant by design. “Permutation-invariant” basically means if you shuffle the input tokens around, the model (without any extra position info) can’t tell you changed the order—it sees them as the same set of tokens. Think of it like a bag of words: if you take words out of a sentence and reinsert them in a different order, a permutation-invariant system would treat those words the same way regardless of sequence.

A Transformer’s self-attention mechanism, by default, just sees a collection of token embeddings and compares every token embedding to every other token embedding. Without positional embeddings, there’s no built-in notion of “the first token,” “the second token,” etc. That’s why we must provide positional information (like positional embeddings) so the model knows the order in which tokens appear.

Why the model needs positional information? Order matters: “I love cats” vs. “Cats love me” use the same tokens, but the meaning depends on which token is first/second/third. Transformer self-attention:
- Self-attention doesn’t inherently preserve input order; it treats the input embeddings as an unordered set of vectors.
- Positional embeddings inject sequence order into the representation so the model can learn to attend more (or less) to nearby tokens.

### model weights: token embedding table and positional embedding table and memory

Above, we learned how token embeddings and positional embeddings work in a typical Transformer. It captures how:
- Token embeddings map each vocabulary token to a unique continuous vector capturing semantic meaning.
- Positional embeddings map each sequence position (1st token, 2nd token, etc.) to another continuous vector capturing structural information (where a token appears in the sentence).
- Self-attention sees all token vectors at once, so without positional information, it has no built-in sense of order.
- Training via cross-entropy loss and stochastic gradient descent (SGD) iteratively shapes both the token embedding table and positional embedding table, so the model learns how word meaning (token embedding) interacts with its position (positional embedding) to form correct predictions.

Both tables are trainable parameters, so both the positional embedding table and the token embedding table are typically stored in GPU memory at inference (and during training). Each is simply a separate learned matrix within the Transformer’s overall parameter set:
- Token Embedding Table
  - Shape: [vocab_size,hidden_dim]
  - Stores a learned vector for each token in the vocabulary.
- Positional Embedding Table
  - Shape: [max_position,hidden_dim]
  - Stores a learned vector for each possible position in the sequence.

Since both must be accessed quickly by the model, they are loaded into GPU memory just like the attention weights, feed-forward layers, and other parameters.

### model weights: self-attention (only forward pass during inference), final output project, and softmax

I truly understand this: "Training via cross-entropy loss and stochastic gradient descent (SGD) iteratively shapes both the token embedding table and positional embedding table, so the model learns how word meaning (token embedding) interacts with its position (positional embedding) to form correct predictions."    And this is key to have meaningful vectors stored in two tables (token and positional) so that we can later reference them when consuming input sequences (such as English text or paragraphs). My question is how is this accomplished during inference? Text comes in, a tokenizer converts it to a token, can we just look up the token id in both the token embedding and positional embedding table, find semantically related tokens, and then send them back to the user? Or do we actually have to do any forward passes and backpropagation, Cross-Entropy Loss, and Stochastic Gradient Descent during inference? And if we have to do this, why? Why can't we just find semantically related vectors in those two tables and send them back? 

During inference, you do not retrain the model or do backpropagation. The token and positional embeddings are already learned (frozen) from training. You still have to do a forward pass, but no gradient updates. Back propagation and gradient updates are only done during training? **Yes. Backpropagation and gradient updates are done only during training (when the model is learning its parameters). Once the model is trained and you’re using it for inference, there’s no more backprop or parameter adjustment—you only perform a forward pass with the frozen (already-learned) weights.**

Is self-attention only done in backpropagation and gradient updates? Or is self-attention done in forward pass as well? Self-attention is used in both the forward pass and during backpropagation (training), but in different ways:
- Forward pass (inference and training):
  - The model computes self-attention to update each token’s representation based on all the other tokens.
  - This is the “normal” use of self-attention: combining (query, key, value) across tokens.
- Backward pass (training only):
  - After the forward pass, you compute a loss (like cross-entropy).
  - Then you do backpropagation, which calculates how changing each parameter (including those in self-attention) would affect that loss.
  - So you also “go through” the self-attention logic in reverse (mathematically) to get gradients for the attention parameters.

It's important to note that self-attention is done in a forward pass. It is not that self-attention has a forward pass. Forward pass encompasses both self-attention and feed-forward. In other words, both self-attention and feed forward are both part of the Transformer block's forward pass.

**Why do we need to do self-attention in the forward pass during inference? Because self-attention is the mechanism that actually incorporates context during inference. Simply looking up token embeddings (and positional embeddings) only gives you a basic representation of “which token, at which position.”** But to understand or generate text, the model must figure out:
- How each token relates to the others in the sequence (e.g. “Does ‘cat’ modify ‘sat,’ or does something else happen?”). In terms of sequence, we are talking about the input sequence of tokens that you feed into the model at inference time.
- Which tokens are important for predicting the next token (in a language model) or determining the answer (in a QA model).

During inference, you typically use the token and positional embedding tables once at the start of the forward pass for each token in the input (or each newly generated token in a language-model setting). Here’s the sequence:
- Embed the Tokens
  - For each input token ID, you look up its row in the token embedding table.
  - For each position i, you also look up its row in the positional embedding table.
  - You add those two vectors: TokenEmbedding(tokenID) + PositionalEmbedding(i).
  - This sum is the initial representation of that token (including content + position).
    - This initial representation is a vector - specifically, a high-dimensional float vector of size hidden_dim. For instance, if your model’s hidden_dim is 3072, then the initial representation for each token will be a 3072-dimensional vector.
- Process Self-Attention in Forward Pass
  - You feed all these token representations (one for each input token) into the subsequent Transformer layers (self-attention + feed-forward).
  - The model “mixes” the tokens’ embeddings based on context, refining each token’s representation.
    - In self-attention, mixing means each token’s updated representation becomes a weighted combination of the other tokens’ representations, guided by the attention scores. Think of it like this: 
      - You have multiple tokens – say “The,” “cat,” “sat.”
      - Each token has some current representation (a vector).
      - Self-attention asks: “For token X, which other tokens might provide useful information?”
      - Attention scores determine how much each other token contributes.
      - If scores for token X ↔ token Y are high, it means “X should pay close attention to Y’s content.”
      - Weighted combination (the “mixing”):
        - Token X’s new representation is built by summing the “value” vectors of all tokens, each multiplied by its attention score. "All tokens" refers to all tokens in the sequence, not all tokens in the entire vocabulary. During self-attention, each token only interacts with the other tokens in the current input.
        - For example, if your input sequence has 10 tokens, each token’s new representation is formed by combining (via weighted sums) the “value” vectors from those 10 tokens in this input sequence.
        - It does not look at every possible token in the token embedding table; only the tokens actually present in the current batch/sequence.
        - Due to weighted combination, **That means X’s new vector is partly from itself, partly from “cat,” partly from “sat,” and so on—weighted by how relevant each is.**
        - Because of this, each token’s updated representation is no longer just “the embedding for that token alone.” It’s enriched by bits of information from the other tokens it’s attending to. That’s what we mean by a “weighted combination” or “mixing.”
- Process Feed Forward in Forward Pass
  - Self-attention and feed-forward are two distinct sub-layers within a Transformer block. Both happen in the forward pass, but they serve different purposes:
    - Self-Attention
      - Lets each token interact with the other tokens in the sequence.
      - Produces context-aware representations by weighting relevant tokens more heavily.
    - Feed-Forward
      - Imagine you have already run self-attention, so each token now has an updated hidden state h. The feed-forward (FFN) sub-layer then operates on each token individually.
      -  You have a token’s hidden state. Let’s call it h. This is a vector of size hidden_dim (e.g., 3072 dimensions). 
      - After self-attention, h already includes context from other tokens, but it’s still a single vector that “belongs” to one specific token.
      - The same MLP is applied to every token’s h. MLP stands for Multi-Layer Perceptron. It’s basically a small fully-connected neural network—usually with one or more hidden layers, each followed by a non-linear activation function. Here’s a simple breakdown:
        - Fully-connected (or “dense”) means every input neuron connects to every neuron in the next layer, using a learned weight for each connection.
        - Multi-layer implies at least one hidden layer between input and output—so the data flows through multiple linear transformations plus non-linear activations.
        - Perceptron is just the old term for a basic unit that does a weighted sum and applies an activation function.
        - In the context of Transformers: Each feed-forward sub-layer is a small MLP that takes a single token’s embedding (a vector) as input, expands it to a larger dimension, applies a non-linear activation (like ReLU or GELU), and then projects it back down. This gives the model more complex (“non-linear”) transformation capacity than just a single linear layer.
        - So, an MLP is just a traditional neural network block: Input vector → (Linear → Activation → Linear) → Output vector.
        - MLP effectively enriches each token’s representation by letting the model learn a more complex, non-linear transformation of that token’s context-enriched vector.
- Final Output Layer
  - After self-attention and feed-forward, each token has a final hidden state.
  - The model does not go back to the embedding tables here; instead, it passes these hidden states into a final output projection (often a linear layer) that maps each hidden state to a vocabulary-sized vector (logits).
  - Softmaxing those logits yields a probability distribution over possible next tokens (for a language model) or a set of class probabilities (for a classifier), etc.

 I am a little confused here. If attention score is only applied to tokens in the input sequence, then how can the response generate tokens not in the input sequence but rather tokens from the token embeddings table? **Because self‐attention is not responsible for choosing the output token from the entire vocabulary—it's responsible for contextualizing the input tokens. The final output layer is what produces a probability distribution over all possible tokens in the vocabulary, allowing the model to predict tokens not in the input.** Here’s how:
- Self‐attention:
  - Operates only over the current input tokens (the sequence so far).
  - Learns how those tokens relate to each other, refining their representations with contextual information.
- Feed forward
- Final Output Projection (the “language modeling head”):
  - Takes the context‐enriched representation of the last token (or each token, depending on the setup) and applies a linear transformation that has one weight vector per vocabulary token.
  - In other words, for each position, the model now has a “hidden state” that encodes what’s happening in the sequence so far.
  - **The final layer maps that hidden state to a logits vector of size = vocab_size, giving a score for every token in the vocabulary—whether or not it appeared in the input.**
- Softmax over those logits:
  - **Converts the scores into a probability distribution over all tokens in the vocabulary.**
  - This distribution often has high probability on tokens that logically follow the sequence context, even if those tokens never appeared in the input.

So, attention only deals with the tokens in the current input (so it knows how they interrelate), but the final projection always covers the entire vocabulary. That’s how the model can generate words or tokens that weren’t in the input.

Do all three of these components have model weights:  self-attention (only forward pass during inference), final output project, and softmax
- Self-Attention: Yes. It has learnable parameters—namely the query, key, value, and output projection matrices (and possibly biases).
- Feed forward: does feed forward have separate model weights than self-attention during forward pass? Yes. Feed-forward and self-attention are separate components within a Transformer block, each with its own trainable parameters:
  - Self-Attention has parameters for:
    - Query, Key, Value projection matrices
    - Output projection matrix (after combining the values)
    - (Plus optional biases)
  - Feed-Forward has parameters for:
    - The first linear layer (W₁, b₁)
    - The non-linear activation (no trainable parameters, just a function)
    - The second linear layer (W₂, b₂)
- Final Output Projection: Yes. This linear layer (hidden_dim → vocab_size) has a weight matrix (and typically a bias) mapping the final hidden states to logits over the vocabulary.
- Softmax: No. Standard softmax is just a mathematical function (exponentiate and normalize). It has no trainable parameters. The model’s weights end at the final linear layer; the softmax is just applied to those logits to turn them into a probability distribution.

### Self-attention and Maximum Sequence Length

When passing data through a transformers model, you must consider:
- How many tokens are in a specific input sequence (the “actual sequence length”).
- How large that sequence can be before exceeding the model’s “maximum sequence length.”

Once you tokenize your input text, you end up with a sequence of token IDs. The length of that list of token IDs is your actual sequence length. For example, if the tokenizer produces 512 tokens from your text, then your input’s sequence length is 512. This count is determined by:
- Tokenization: The tokenizer splits your text into tokens (subwords, wordpieces, etc.).
- Number of tokens: However many tokens the text yields – for instance, 512 tokens, 1024 tokens, etc.

So the sequence length for that batch/input is just the count of tokens the tokenizer produces.

A Transformer also has a maximum sequence length it can handle. This limit can come from one (or more) of the following:
- Positional Embedding Shape
  - For learned, absolute positional embeddings, you typically have a table of shape [max_position,hidden_dim]. If max_position = 4096, the model can’t natively handle tokens at positions beyond 4096, because there are no embeddings for positions 4097, 4098, etc.
  - Example: If a model has max_position = 2048, it can’t directly handle sequences of length 4096 (there’s no row for position 3000, say).
- Rotary or Relative Positional Embeddings
  - Some models (e.g., LLaMA, GPT-NeoX, T5) use relative or rotary embeddings, which can be more flexible about maximum sequence length. However, even these often have a practical limit defined in the code or config (e.g., 4096 or 8192).
  - Going beyond that limit might degrade performance or require special tweaks, even if it’s theoretically possible to extend.
- Memory Constraints
  - Even if the positional encoding allows large sequences (say 64k tokens), you might run out of GPU memory. Self-attention scales roughly with O(n²) in the number of tokens n. So at some length, you simply can’t store all the key/value states in GPU memory. When we say “O(n²),” we’re using Big-O notation to describe how the compute and memory cost of self-attention grows with the sequence length 𝑛. Specifically:
    - Self-attention requires computing attention scores between every pair of tokens in the sequence. If you have n tokens, that’s n queries each compared to n keys, i.e. n×n=n² comparisons.
    - This also typically involves storing or processing an n×n attention matrix in memory (for attention scores).
    - So, as n (the sequence length) grows larger, the required compute and GPU memory grow quadratically, which quickly becomes prohibitive for very large sequences.
- Configuration Files
  - Many model configs (e.g., config.json) explicitly set a max_position_embeddings field. That number is the official cap on sequence length.
 
### Self-attention and KV Cache

The K/V cache (key/value cache) isn’t part of the model’s trainable weights; instead, it’s temporary (ephemeral) storage the model uses during inference (and possibly training) to speed up self-attention across tokens. Here’s how it fits in:
- Not Trainable Parameters
  - Unlike the token embedding table or the attention weight matrices (Q, K, V projections), the K/V cache isn’t “learned.”
  - It’s just a buffer that holds the key and value representations of each token in a sequence, so you don’t have to recompute them from scratch every time you attend to those tokens.
- Why It Exists (Especially in Autoregressive Generation)
  - In a language model generating text token by token, at each new step the model only needs to compute queries for the newest token—then look up the keys and values of all previously generated tokens (which have been cached).
  - Without a K/V cache, you’d have to re-run self-attention over the entire prefix each time you generate a new token, repeating a lot of computations.
- Memory Usage
  - The K/V cache can be large because it stores a hidden-state vector (the key and the value) for each token at each layer of the Transformer.
  - That’s why the memory usage for the K/V cache can grow linearly with the length of your sequence—and is often cited in the O(n^2) discussion (since bigger sequences also mean more attention score computations).
- Ephemeral vs. Persistent
  - The cache is ephemeral: it exists only for the current forward pass or the incremental generation session. Once you’re done generating text or finish a batch inference, you can discard it.
  - Model weights, on the other hand, are persistent trainable parameters (like the embedding tables, attention matrices, feed-forward layers). They remain fixed in GPU memory during inference.

During inference you typically need both:
- Model Weights (all the trainable parameters like token/positional embeddings, attention matrices, feed-forward layers), which are fixed (frozen) during inference but must be in GPU memory so the model can do its forward pass efficiently.
- K/V cache, which is temporary storage for the attention keys/values of each token in the current context.
  - This grows with your sequence length so that each newly generated token (in an autoregressive model) can quickly “look up” previously computed keys/values.
  - Not trainable, but kept in GPU memory for fast access during generation or inference steps.

### Beyond Model Weights and KV Cache

Beyond model weights and the K/V cache, you’ll also see some additional GPU memory usage for:
- Intermediate Activations
  - Even in a purely forward-pass scenario (inference), each Transformer layer produces intermediate tensors (the outputs of attention, feed-forward steps, etc.) while computing the final output.
  - These temporary activations only persist briefly but do occupy GPU memory as the model runs forward.
- Framework Overheads and Buffers
  - PyTorch, TensorFlow, or another deep-learning framework may allocate working buffers, scratch space, and CUDA contexts. This overhead can vary depending on batch size, sequence length, and the exact hardware/software stack.
- Compiled Kernels or Graphs (optional / implementation-specific)
  - If you’re using features like torch.compile, TensorRT, or other optimizations, the system might store compiled code or runtime graphs in GPU memory (or partially in CPU memory, depending on the setup).
- Other Temporary Data Structures
  - Depending on your pipeline (e.g., multi-head attention might have extra buffers for reshaping heads, or gather/scatter operations).
  - Some libraries also keep reference copies of certain parameters or partial computations to accelerate subsequent steps.

### Managing memory with max_model_len

Is there a way to instruct vLLM when running a model to not use all token embeddings, positional embeddings, self-attention forward pass capabillity, feed forward forward pass capability, final output projection capability in order to ensure the model weights doesn't reach near the gpu limits?

Not really. You can’t just say, “don’t load the self-attention or final projection,” because those components are fundamental to the model’s architecture – if you drop them, the model’s forward pass wouldn’t work at all. Most serving solutions (including vLLM) assume you load the entire model and run a proper forward pass.

That said, here are some practical ways to reduce GPU memory usage with vLLM (or other frameworks) without literally “skipping” major model components. max_model_len  controls the maximum number of tokens the model will handle in a single forward pass. In other words, it’s an upper bound on the context window (the sequence length) during inference. By lowering max_model_len, you reduce how many tokens the model can process at once—thus reducing memory usage (especially for attention and K/V cache). How it affects each component:
- Token embeddings:
  - You still have a full token embedding matrix (vocab_size × hidden_dim). That’s a fixed portion of the model’s weights. max_model_len doesn’t prune or shrink the embedding table for tokens.
  - Instead, max_model_len just limits how many tokens from the prompt/response are fed through the model at once.
- Positional embeddings:
  - If you set --max-model-len=2048, the model will only use positional embeddings up to position 2048. It effectively ignores embeddings beyond that index.
  - This can also help avoid out-of-bounds lookups if the original model supports more tokens but you want to cap it.
- Self-attention:
  - Self-attention scales with O(n^2) in the number of tokens n. By capping n (the sequence length) at max_model_len, you keep these computations and memory needs under control.
  - Self-attention and feed-forward happen inside each Transformer layer. Both require GPU memory and compute proportional to the sequence length (n) in slightly different ways:
    - Why O(n^2): Self-attention compares each token (query) with every token (key) to compute attention scores. That’s n×n comparisons (dot products), forming an n×n attention matrix.
    - Also, each token’s updated representation is a weighted sum of up to n “value” vectors.
  - How max_model_len helps:
    - If you limit n to a smaller maximum (e.g. 2048 tokens instead of 4096), then the number of operations and the size of intermediate tensors (like the attention matrix) shrink quadratically. For instance, going from 4096 tokens to 2048 tokens reduces memory/compute for attention by roughly a factor of 4 (4096^2=16,777,216 vs. 2048^2=4,194,304). 16,777,216 is the number of elements in a 4096×4096 matrix. 4,194,304 is the number of elements in a 2048×2048 matrix. They represent how many attention “slots” or comparisons you’d store or compute if you have n tokens in your sequence. 
- Feed-Forward:
  - The position-wise feed-forward network processes each token vector individually. That’s an O(n) operation if you do it for each token, but the exact compute is linear in the number of tokens.
  - High dimension expansions: Each token is mapped to an intermediate dimension (often 4× hidden_dim) and back. If hidden_dim is large, feed-forward can still be expensive.
  - Fewer tokens, fewer feed-forward calls: If you cap max_model_len at a lower number of tokens, you have fewer per-token feed-forward operations to do. That linearly reduces memory usage for feed-forward intermediate activations.
- K/V cache:
  - The K/V cache size is roughly proportional to [# of tokens] × [# of Transformer layers] × [hidden size].
    - How do I find out the number of transformer layers from hugging face model card config? n most Hugging Face transformer configs (the JSON file stored alongside the model on the Hub), you’ll see an integer field that indicates the number of layers. The exact name varies by architecture: BERT-like models often have a key like "num_hidden_layers".
    - In vLLM can I reduce the number of transformer layers? no. vLLM expects to load and run the entire model as it was trained. There’s no built-in feature to “drop layers” or only load a subset of them. If you tried to remove some layers on your own, the model’s weights and structure wouldn’t match up, and it would no longer produce meaningful results without a full retraining or fine-tuning process.
  - Restricting max_model_len puts a hard limit on how many tokens can appear in the prompt (and ongoing generation), so it caps the maximum K/V cache memory usage.
- Final output projection:
  - The dimension of the final output projection is [hidden_dim, vocab_size]. This doesn’t change based on max_model_len, because you still need to map any hidden state to a vocab-sized logits vector.
  - Lowering max_model_len won’t shrink this layer, but it does ensure fewer tokens go through that projection in a single pass.

### Managing memory with max_num_batched_tokens

When you run vLLM in batching mode (where multiple requests are served together in a single forward pass to improve throughput) - does this mean if we have 2000 tokens per input sequence and there are two concurrent requests, both requests will be batched together and pass 4000 tokens in a single forward pass (self-attention and feed forward)? Yes—that’s exactly what batching means. If vLLM can bundle both 2,000‑token requests into one forward pass, it effectively runs a single forward pass over 4,000 tokens total. Concretely:
- Two concurrent requests arrive, each with a 2,000‑token input sequence.
- vLLM checks its batching limits (e.g., --max-num-batched-tokens) to see if both requests can fit into a single pass.
- If they do fit, vLLM combines them into one batched forward pass. Internally, the model sees 2 sequences in the same “batch,” each sequence having 2,000 tokens (or the model might pad them so they fit neatly into a single tensor).
- Self-attention and feed-forward then process 4,000 tokens in total (2 sequences × 2,000 tokens each) in one go, leveraging the GPU’s parallelism.

This boosts throughput—you pay the overhead of one forward pass instead of two separate passes. However:
- You use more GPU memory in one shot (4,000 tokens → bigger attention matrices and feed-forward buffers).
- If --max-num-batched-tokens < 4,000, vLLM can’t combine them fully; it might split them into separate passes or partial batches.

So, in short: batched means multiple request sequences are concatenated (or processed simultaneously) in a single forward pass, leading to a total token count = sum of each sequence’s token count.

how does vLLM not mix tokens between the two separate input sequences when referencing token embedding, positional embedding, self-attention, feed forward, and final output projection, thus sacrificing the semantic context and output? Because batching simply processes multiple sequences in parallel rather than merging them into a single sequence. The model sees each sequence as a separate row in the batch dimension, and it uses attention masking (or sequence-length metadata) to ensure:
- Token embeddings:
  - Each token ID is looked up by (batch_index, token_index). The embedding table returns a vector, but it knows which batch row it’s for. There’s no cross-talk in the embedding lookup.
- Positional embeddings:
  - Each token in each sequence has its own position index (0…n-1). The model keeps track of “sequence A, position i” vs. “sequence B, position j.” They don’t share the same positional indexes.
- Self-attention:
  - For each batch row (i.e., each sequence), the model typically applies a mask so tokens in one sequence do not attend to tokens from another. This is done by zeroing out cross-sequence attention scores or setting them to −∞.
  - This way, “sequence A’s tokens” only attend to “sequence A’s tokens,” and similarly for B.
- Feed-forward:
  - This MLP runs per token. Each token vector is processed independently within the batch dimension. So “sequence A, token i” is fed through the same feed-forward sub-layer as “sequence B, token j,” but with separate vectors and no cross-mixing.
- Final output projection:
  - Once each token has its final hidden state, the model applies the same linear mapping + softmax to produce logit distributions for each token in each sequence. Sequences are just separate rows in the batch dimension, so each token’s output is separate.
Overall, batching in vLLM means you combine multiple requests into one forward pass. Internally, each sequence is still isolated by padding or masking. The model never confuses or merges tokens across different sequences. It’s just more efficient to run them in parallel on the GPU rather than one at a time.

### Managing memory with cpu_offload_gb

The --cpu-offload-gb CPU_OFFLOAD_GB option in vLLM allows you to offload part of the model’s memory usage to CPU if you don’t have enough VRAM. Example: vllm serve --model <model> --cpu-offload-gb 4. This tries to offload up to 4 GB of model data to CPU. It can slow down inference, but reduces GPU memory pressure.

### Managing memory with Quantization

If a model supports 8-bit or 4-bit quantization, you can reduce memory usage by storing weights in fewer bits. (vLLM’s built-in quantization support is still evolving, but many frameworks provide ways to quantize.)

### Managing memory with Sharded 

can shard the weights across more than one GPU, effectively giving you more VRAM capacity.

### Managing memory with reduced transformer layers 

vLLM expects to load and run the entire model as it was trained. There’s no built-in feature to “drop layers” or only load a subset of them. If you tried to remove some layers on your own, the model’s weights and structure wouldn’t match up, and it would no longer produce meaningful results without a full retraining or fine-tuning process.

Distillation: If you truly want a smaller model with fewer layers, you can do a model distillation process—training a new “student” model with fewer layers on the outputs of the “teacher” model. That’s a bigger effort, but it’s how many “mini” or “distilled” models get created.

### vLLM setup

If you are using NVIDIA GPUs, you can install vLLM using pip directly. It’s recommended to use uv, a very fast Python environment manager, to create and manage Python environments. Please follow the documentation to install uv. After installing uv, you can create a new Python environment and install vLLM using the following commands:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
uv venv myenv --python 3.12 --seed
source myenv/bin/activate
uv pip install vllm
```
  
### OpenAI-Compatible Server

So remember  I have a maximum sequence length per forward pass, e.g. 30k. In the logistic function, I believe L is the upper bounds. which i define as 120 representing 120k tokens. Hence, per retriever we are doing something like this:        

```python
for index, metadata in enumerate(state["metadata"]):
            filter_expression = self.create_filter_expression(metadata)
            search_kwargs = {
                "k": int(round(Sigmoid.logistic(
                    x=self.chat_model.tokenizer.sequence_length_forward_pass, 
                    L=120, 
                    a=2e-5, 
                    m=60000
                ))),
                "filter": filter_expression,
            }   
```

so k is the number of child documents we want to fetch with retriever. Each retriever will pull a chunk at maximum token length of 150. It is not guaranteed to be 150, but that is the max token length that each retriever can pull. So  so if the logistic function returns 47, that is 47 child documents returned, but remember we don't actually load the child documents into the LLM context length. Instead we load the parents. Since many documents share the same parents, that 47 can turn into 10. Each parent document is at most 2000 tokens. It is not guaranteed to be 2000, but that is the max for chunk. Consequently, it is possible to have 20,000 tokens if 10 documents retrieved each 2000 tokens (which of course they are not all going to be 2000 tokens). That 20k is within 30k, but that is only for one retriever. If the user uploaded 4 documents, there will be 4 retrievers, and each may potentially produce a k that leads to 20k returned of parent childs. 20 * 4 is 80 which is above the 30k max. In those situations, it must penalize all retrievers as fairly as possible but of course to get as close to 30k, some will have more tokens then others. Is this where you are going with the hamilton method?

\[ 
k_i = \mathrm{round}\Bigl(\mathrm{Sigmoid.logistic}(x_i, L, a, m)\Bigr)
\]
$k_i =$ original value
$k_i' =$ updated value

\[
k_{\text{sum}} = \sum_{i=1}^{N} k_i
\]

\[
\text{If } \sum k_i \,\le\, K_{\max}, 
\quad \text{then define } k_i' = k_i.
\]

When each retriever estimates a desired number of docs (e.g. 50, 40, 60) but the total (120) exceeds your global limit (say 100), you want to:
Stay within that overall capacity (120)
Preserve the relative proportions among the retrievers’ requests as much as possible.

So we divide K sub max by summation of k sub i (which represents the total of all the docs). That division gives you the exact ratio to “shrink” all requests so they sum to K sub max. For example: If each retriever’s request sums to 150, but you can only afford 100 total, the ratio α is 100/150 = 0.666.
\[
\text{If } k_{\text{sum}} > K_{\max}, 
\quad \text{then let } \alpha = \frac{K_{\max}}{\sum k_i}
\quad 
\]

Multiplying each original $k_i$ by the same $α$ ensures the relative proportions among them stay the same.
\[
k_i' = \alpha \times k_i
\]

For instance: 

Originally:
- Retriever A wants 50 (which is $1/3$ of the total 150),
- Retriever B wants 40 ($4/15$) 
- Retriever C wants 60 ($2/5$)

After scaling by $𝛼 = 0.666$ where only each retriever can have 66 percent of what it had:
A wants 33.3 (still $1/3$ of the new total 100),
B wants 26.6 (still $4/15$),
C wants 40 (still $2/5$)

Hence they remain in proportion but no longer exceed 100 total.

Then we use the hamilton method:
\[
\text{base\_sum} 
= \sum_{i=1}^{N} \lfloor k_i' \rfloor.
\]
Above we get the floors of each proportion. Suppose three retrievers have original $k_1$ = 50, $k_2$ = 40, $k_3$ = 60. Sum is 150, but $K_{\text{max}}$ = 100. Then $α = 100/150 = 0.666.$ So $k_1' = 33.3, k_2' = 26.6, k_3' = 40.0$. Floors:  
\[
\lfloor 33.3 \rfloor = 33, 
\quad
\lfloor 26.6 \rfloor = 26, 
\quad
\lfloor 40.0 \rfloor = 40.
\]Sum=99, leftover=1.

Remainders: $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.0$. Highest remainder is 0.6 (retriever #2), so it gets the leftover doc. Total is 100, staying under the limit, and each retriever’s share is proportionally fair.
\[
\widehat{k}_1 = 33, 
\quad
\widehat{k}_2 = 27, 
\quad
\widehat{k}_3 = 40.
\]


 
LLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API. By default, it starts the server at http://localhost:8000. You can specify the address with --host and --port arguments. The server currently hosts one model at a time and implements endpoints such as list models, create chat completion, and create completion endpoints. Run the following command to start the vLLM server with the vlm2vec-full model:

```shell
ulimit -n
65536

# you may need to start a new container frequently due to the load it receives
# otherwise it can lead to errors where the writes stop working and it tries
# using a read-replica to write to
docker container run -d -p 6379:6379 redis/redis-stack-server:latest

source myenv/bin/activate
# 140.77 seconds (with new improvement! batching really works with the embedding model!!!!)
vllm serve --port 8070 --host 0.0.0.0 --trust-remote-code --tensor-parallel-size 1 --max-model-len 2048 --max-num-batched-tokens 8192 --task embed TIGER-Lab/VLM2Vec-Full
# 145.53 seconds 
# second run: 140.75 seconds
# (with new improvement! batching really works with the embedding model!!!!)
vllm serve --port 8070 --host 0.0.0.0 --trust-remote-code --tensor-parallel-size 1 --max-model-len 8192 --max-num-batched-tokens 8192 --task embed TIGER-Lab/VLM2Vec-Full

# using parent document retriever with 20 concurrency: 
# Time elapsed: 141.60 seconds
# using parent document retriever with 4 concurrency:
# Time elapsed: 148.17 seconds

# ensuring last chunks of page aren't too small:
# Time elapsed: 153.60 seconds

INFO 03-23 05:00:55 [__init__.py:256] Automatically detected platform cuda.
INFO 03-23 05:00:57 [api_server.py:977] vLLM API server version 0.8.1
INFO 03-23 05:00:57 [api_server.py:978] args: Namespace(subparser='serve', model_tag='TIGER-Lab/VLM2Vec-Full', config='', host='0.0.0.0', port=8070, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, enable_ssl_refresh=False, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='TIGER-Lab/VLM2Vec-Full', task='auto', tokenizer=None, hf_config_path=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=True, allowed_local_media_path=None, download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='auto', kv_cache_dtype='auto', max_model_len=2048, guided_decoding_backend='xgrammar', logits_processor_pattern=None, model_impl='auto', distributed_executor_backend=None, pipeline_parallel_size=1, tensor_parallel_size=1, enable_expert_parallel=False, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=None, enable_prefix_caching=None, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=None, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=2048, max_num_partial_prefills=1, max_long_partial_prefills=1, long_prefill_token_threshold=0, max_num_seqs=None, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, disable_mm_preprocessor_cache=False, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, use_tqdm_on_load=True, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', scheduler_cls='vllm.core.scheduler.Scheduler', override_neuron_config=None, override_pooler_config=None, compilation_config=None, kv_transfer_config=None, worker_cls='auto', worker_extension_cls='', generation_config='auto', override_generation_config=None, enable_sleep_mode=False, calculate_kv_scales=False, additional_config=None, enable_reasoning=False, reasoning_parser=None, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False, enable_server_load_tracking=False, dispatch_function=<function ServeSubcommand.cmd at 0x7f81cc92dee0>)
WARNING 03-23 05:00:57 [utils.py:2079] Found ulimit of 8192 and failed to automatically increase with error current limit exceeds maximum limit. This can cause fd limit errors like `OSError: [Errno 24] Too many open files`. Consider increasing with ulimit -n
INFO 03-23 05:00:57 [config.py:208] Replacing legacy 'type' key with 'rope_type'
WARNING 03-23 05:00:57 [config.py:215] Replacing legacy rope_type 'su' with 'longrope'
INFO 03-23 05:01:05 [config.py:1693] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO 03-23 05:01:10 [__init__.py:256] Automatically detected platform cuda.
INFO 03-23 05:01:12 [core.py:53] Initializing a V1 LLM engine (v0.8.1) with config: model='TIGER-Lab/VLM2Vec-Full', speculative_config=None, tokenizer='TIGER-Lab/VLM2Vec-Full', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=None, served_model_name=TIGER-Lab/VLM2Vec-Full, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":3,"custom_ops":["none"],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output"],"use_inductor":true,"compile_sizes":[],"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":512}
WARNING 03-23 05:01:13 [utils.py:2282] Methods determine_num_available_blocks,device_config,get_cache_block_size_bytes,initialize_cache not implemented in <vllm.v1.worker.gpu_worker.Worker object at 0x7faad654bb90>
INFO 03-23 05:01:14 [parallel_state.py:967] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0
INFO 03-23 05:01:14 [cuda.py:215] Using Flash Attention backend on V1 engine.
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.json'
[2025-03-23 05:01:14] ERROR file_download.py:1389: Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.json'
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.jinja'
[2025-03-23 05:01:14] ERROR file_download.py:1389: Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.jinja'
/home/ubuntu/myenv/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py:602: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
  warnings.warn(
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.json'
[2025-03-23 05:01:15] ERROR file_download.py:1389: Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.json'
Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.jinja'
[2025-03-23 05:01:15] ERROR file_download.py:1389: Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/.no_exist/0f078450a78f421b62630e4ade3da5778efd98fd/chat_template.jinja'
INFO 03-23 05:01:16 [gpu_model_runner.py:1164] Starting to load model TIGER-Lab/VLM2Vec-Full...
INFO 03-23 05:01:16 [config.py:3222] cudagraph sizes specified by model runner [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512] is overridden by config [512, 384, 256, 128, 4, 2, 1, 392, 264, 136, 8, 400, 272, 144, 16, 408, 280, 152, 24, 416, 288, 160, 32, 424, 296, 168, 40, 432, 304, 176, 48, 440, 312, 184, 56, 448, 320, 192, 64, 456, 328, 200, 72, 464, 336, 208, 80, 472, 344, 216, 88, 120, 480, 352, 248, 224, 96, 488, 504, 360, 232, 104, 496, 368, 240, 112, 376]
WARNING 03-23 05:01:16 [topk_topp_sampler.py:63] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
INFO 03-23 05:01:16 [weight_utils.py:257] Using model weights format ['*.safetensors']
Ignored error while writing commit hash to /home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/refs/main: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/refs/main'.
[2025-03-23 05:01:16] WARNING _snapshot_download.py:264: Ignored error while writing commit hash to /home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/refs/main: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub/models--TIGER-Lab--VLM2Vec-Full/refs/main'.
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:37<00:37, 37.10s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [01:02<00:00, 30.29s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [01:02<00:00, 31.31s/it]

INFO 03-23 05:02:19 [loader.py:429] Loading weights took 62.67 seconds
INFO 03-23 05:02:19 [gpu_model_runner.py:1176] Model loading took 7.9083 GB and 63.438900 seconds
INFO 03-23 05:02:19 [gpu_model_runner.py:1421] Encoder cache will be initialized with a budget of 2048 tokens, and profiled with 3 image items of the maximum feature size.
INFO 03-23 05:02:28 [backends.py:409] Using cache directory: /home/ubuntu/.cache/vllm/torch_compile_cache/dde0fd4b63/rank_0_0 for vLLM's torch.compile
INFO 03-23 05:02:28 [backends.py:419] Dynamo bytecode transform time: 7.30 s
INFO 03-23 05:02:31 [backends.py:132] Cache the graph of shape None for later use
INFO 03-23 05:02:56 [backends.py:144] Compiling a graph for general shape takes 27.31 s
INFO 03-23 05:03:04 [monitor.py:33] torch.compile takes 34.61 s in total
INFO 03-23 05:03:05 [kv_cache_utils.py:537] GPU KV cache size: 28,704 tokens
INFO 03-23 05:03:05 [kv_cache_utils.py:540] Maximum concurrency for 2,048 tokens per request: 14.02x
INFO 03-23 05:03:30 [gpu_model_runner.py:1499] Graph capturing finished in 25 secs, took 1.58 GiB
INFO 03-23 05:03:30 [core.py:138] init engine (profile, create kv cache, warmup model) took 70.27 seconds
INFO 03-23 05:03:30 [api_server.py:1024] Starting vLLM API server on http://0.0.0.0:8070
INFO 03-23 05:03:30 [launcher.py:26] Available routes are:
INFO 03-23 05:03:30 [launcher.py:34] Route: /openapi.json, Methods: HEAD, GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /docs, Methods: HEAD, GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /docs/oauth2-redirect, Methods: HEAD, GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /redoc, Methods: HEAD, GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /health, Methods: GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /load, Methods: GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /ping, Methods: POST, GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /tokenize, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /detokenize, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /v1/models, Methods: GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /version, Methods: GET
INFO 03-23 05:03:30 [launcher.py:34] Route: /v1/chat/completions, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /v1/completions, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /v1/embeddings, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /pooling, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /score, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /v1/score, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /v1/audio/transcriptions, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /rerank, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /v1/rerank, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /v2/rerank, Methods: POST
INFO 03-23 05:03:30 [launcher.py:34] Route: /invocations, Methods: POST
INFO:     Started server process [2239]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Depending on the settings of the server, you may or may not be able to do concurrency. An asyncio.Semaphore is a concurrency primitive in Python’s asyncio library that limits how many coroutines can run a particular section of code at once. When you do:

```python
semaphore = asyncio.Semaphore(1)
```
you’re creating a semaphore that allows only one coroutine to acquire it at a time. In other words, it enforces that any code wrapped with:

```python
async with semaphore:
    ...
```
runs sequentially – only one piece of code can be in that block at a time.

If you had something like Semaphore(5), up to 5 coroutines could enter that block concurrently, but no more. The purpose is usually to throttle or rate-limit some resource-intensive or external call so you don’t overwhelm a system (e.g., a rate-limited API) or cause an overly heavy load on your local resources.

Does asyncio create multiple threads or is it a single thread in an event loop? By default, asyncio runs its event loop in a single thread and manages concurrency by interleaving tasks (coroutines). This is cooperative concurrency, not parallelism. Each coroutine yields control (for example, when awaiting I/O) back to the event loop so other tasks can run.

You can, however, combine asyncio with libraries such as concurrent.futures.ThreadPoolExecutor or ProcessPoolExecutor to do work in multiple threads or processes, but pure asyncio uses an event loop that generally operates on a single thread. In this snippet:

```python
semaphore = asyncio.Semaphore(1)

async def process_document(document: Document):
    async with semaphore:
        batch_ids = await self._process_batch(
            [document], ttl_seconds, redis_client, **kwargs
        )
        return batch_ids

tasks = [
    asyncio.create_task(process_document(document))
    for document in documents
]
results = await asyncio.gather(*tasks)
```
- semaphore = asyncio.Semaphore(1) ensures that only one coroutine at a time can enter the async with semaphore: block.
- As a result, only one document is processed (i.e., embedded and stored in Redis) at a time in this particular piece of code.
- So, there are never two concurrent requests going out to the embedding server or to Redis in that specific code path, because each call to _process_batch() (and transitively the calls to aadd_documents() and redis_client.expire()) happens sequentially.


- KV cache tokens
Inside a Transformer, each layer stores “keys” and “values” for every token in the sequence, so it can quickly attend to previously processed tokens. This key/value data is the “KV cache.”

For each token, at each layer, the model stores “key” and “value” tensors. This is how the attention mechanism can refer back to previously processed tokens when generating or embedding the next token. If requests truly run in parallel (i.e., the GPU is switching back and forth between them at the layer or batch level), the model needs the KV data for all those active sequences available in GPU memory, so it can instantly access previous tokens’ keys/values. So if each request uses 4,096 tokens, 10 such requests in flight at once require space for 10×4,096=40,960 tokens in the KV cache.

What if I have 10 concurrent requests of 4096 tokens and my gpu memory can handle 40,960 tokens in KV cache total, and then the 10th request is followed by another request, is there enough time for the KV cache to be freed before the 11th requests comes in and demands another 4096 tokens of KV cache? Generally, yes—once a given request is finished (i.e., its prompt and generation/embedding work is done), the model no longer needs to store the key/value states for that request, and that portion of the KV cache is freed up. So if your GPU memory can handle exactly 40,960 tokens (10 parallel requests at 4,096 each) and you only add an 11th request after at least one of those earlier requests has fully completed, there would be new space available in the KV cache for that next request.

However, how smoothly that works depends on:
- Whether the earlier requests truly “finished”
  - If even one of those 10 requests is still in flight when the 11th arrives, you’re effectively going up to 11 concurrent requests. If your system only has capacity for 40,960 tokens in the KV cache, and you suddenly need 45,056 tokens (11 × 4,096), that’s beyond your limit. At that point, vLLM 
    - Queue or delay the 11th request until memory is freed,
    - Evict older requests’ KV states (causing a performance drop for those requests),
    - Or potentially fail with an out-of-memory error (depending on the server’s behavior).
Many server frameworks will serialize or queue new requests if the GPU is at capacity, ensuring you never exceed the KV memory limit. Others may attempt to serve all requests at once, risking out-of-memory if concurrency spikes beyond your KV cache capacity. vLLM also does “chunked prefill,” where long prompts might be processed incrementally, so the peak memory usage per request might not be at the full 4,096 tokens simultaneously—depending on the server’s scheduling.

Does vLLM do queueing of requests when concurrency is reached?
Yes—vLLM implements a scheduling policy and can effectively queue or batch incoming requests so they don’t exceed the GPU’s memory capacity. By default, vLLM uses a “First Come, First Served” (FCFS) scheduler. If your concurrency is very high (and you’re near the KV cache or GPU memory limit), new requests may be queued while the engine finishes processing earlier ones.

Highlights from vLLM’s scheduling:
- FCFS scheduling: Incoming requests are handled in the order they arrive, with each request being put in a queue if the system is at capacity.
- Chunked prefill: For large prompts, vLLM processes them incrementally, which helps reduce the maximum memory usage at any one moment.
- Per-request concurrency limit: If you exceed your total KV cache limit with simultaneous requests, vLLM may delay (queue) new requests until capacity is freed.




“GPU KV cache size: 28,688 tokens” means that with your GPU memory, the engine can store up to ~28.7k tokens of K/V cache at once. That “GPU KV cache size: 28,688 tokens” is telling you how many tokens’ worth of key-value cache can fit into your GPU memory across all active requests at runtime.  The GPU KV cache size of 28,688 tokens means that, across all sequences currently being processed in parallel, you can store up to ~28.7k tokens worth of K/V states in GPU memory. 



“Maximum concurrency for 4,096 tokens per request: 7.00x” means you can handle about 7 concurrent requests at the full 4096-token context each before running out of GPU memory for caching. In other words, 7 parallel requests of length 4096 tokens.


H100 box

g5.12xlarge
sg-036a8002c430fc904
ami-01c346d8c45cc6ae1
generative-ai

# run it along tei:
token=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
model=BAAI/bge-large-en-v1.5
volume=$PWD/data
docker run --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8071:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 32768 --max-client-batch-size 128 --max-batch-requests 64 --auto-truncate

 docker container ls
CONTAINER ID   IMAGE                                               COMMAND                  CREATED         STATUS         PORTS
                         NAMES
daee2eb610ac   ghcr.io/huggingface/text-embeddings-inference:1.5   "text-embeddings-rou…"   3 minutes ago   Up 3 minutes   0.0.0.0:8071->80/tcp, [::]:8071->80/tcp       beautiful_bohr
d55a1a476498   vllm/vllm-openai:latest                             "python3 -m vllm.ent…"   21 hours ago    Up 2 hours     0.0.0.0:8070->8000/tcp, [::]:8070->8000/tcp   kind_leakey



Like HF TGI, VLLM is an open-source library designed specifically for deploying and serving large language models (LLMs).

vLLM is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries. As of now, vLLM’s binaries are compiled with CUDA 12.1 and public PyTorch release versions by default. In order to be performant, vLLM has to compile many cuda kernels. The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides wheels for Linux running on x86 platform with cuda 12 for every commit since v0.5.3. You can download and install the latest one with the following command:

```shell
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

Another way to access the latest code is to use the docker images:

```shell
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}
```

These docker images are used for CI and testing only, and they are not intended for production use. They will be expired after several days. Latest code can contain bugs and may not be stable. Please use it with caution.

### Deploying with Docker

vLLM offers an official Docker image for deployment. The image can be used to run OpenAI compatible server and is available on Docker Hub as vllm/vllm-openai.

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```

You can either use the ipc=host flag or --shm-size flag to allow the container to access the host’s shared memory. vLLM uses PyTorch, which uses shared memory to share data between processes under the hood, particularly for tensor parallel inference.

By default vLLM will build for all GPU types for widest distribution. If you are just building for the current GPU type the machine is running on, you can add the argument --build-arg torch_cuda_arch_list="" for vLLM to find the current GPU type and build for that.

Before going into the details of distributed inference and serving, let’s first make it clear when to use distributed inference and what are the strategies available. The common practice is:

- Single GPU (no distributed inference): If your model fits in a single GPU, you probably don’t need to use distributed inference. Just use the single GPU to run the inference.

- Single-Node Multi-GPU (tensor parallel inference): If your model is too large to fit in a single GPU, but it can fit in a single node with multiple GPUs, you can use tensor parallelism. The tensor parallel size is the number of GPUs you want to use. For example, if you have 4 GPUs in a single node, you can set the tensor parallel size to 4.

- Multi-Node Multi-GPU (tensor parallel plus pipeline parallel inference): If your model is too large to fit in a single node, you can use tensor parallel together with pipeline parallelism. The tensor parallel size is the number of GPUs you want to use in each node, and the pipeline parallel size is the number of nodes you want to use. For example, if you have 16 GPUs in 2 nodes (8GPUs per node), you can set the tensor parallel size to 8 and the pipeline parallel size to 2.

In short, you should increase the number of GPUs and the number of nodes until you have enough GPU memory to hold the model. The tensor parallel size should be the number of GPUs in each node, and the pipeline parallel size should be the number of nodes.

After adding enough GPUs and nodes to hold the model, you can run vLLM first, which will print some logs like # GPU blocks: 790. Multiply the number by 16 (the block size), and you can get roughly the maximum number of tokens that can be served on the current configuration. If this number is not satisfying, e.g. you want higher throughput, you can further increase the number of GPUs or nodes, until the number of blocks is enough.

vLLM supports distributed tensor-parallel inference and serving. Currently, we support Megatron-LM’s tensor parallel algorithm. We also support pipeline parallel as a beta feature for online serving. We manage the distributed runtime with either Ray or python native multiprocessing. Multiprocessing can be used when deploying on a single node, multi-node inferencing currently requires Ray.

Multiprocessing will be used by default when not running in a Ray placement group and if there are sufficient GPUs available on the same node for the configured tensor_parallel_size, otherwise Ray will be used. 

If a single node does not have enough GPUs to hold the model, you can run the model using multiple nodes. It is important to make sure the execution environment is the same on all nodes, including the model path, the Python environment. The recommended way is to use docker images to ensure the same environment, and hide the heterogeneity of the host machines via mapping them into the same docker configuration.

### Deploying with Docker on AWS EC2

Since vLLM uses Python, Pytorch and Cuda drivers, we will use an instance that has Python, Pytorch and Cuda drivers pre-installed. We could choose the Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1 (Amazon Linux 2) AMI or the Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1 (Ubuntu 20.04) AMI. We will choose the later since Ubuntu reflects our local setup.

vLLM requires access to one of the following EC2 instances: G4dn, G5, G6, Gr6, P4, P4de, P5. This is needed given the intensive resources required to deploy model’s weights and to store KV cache. It is not recommended to use G4dn actually as it is uses the older NVIDIA Tesla T4 GPU, which was released in 2018 and build on the Turing architecture. For example, the HuggingFace TGI does not support it. The HuggingFace TGI supports the G5 instance, which uses the NVIDIA A10G Tensor Core GPU powered by the Ampere architecture. Follow the link to get a description of this Instance Type: https://aws.amazon.com/ec2/instance-types/g5/. Hence, for vLLM, we will also use the G5 instance.

You need to generate an API key so you can securely access the endpoint. To do so, se Python's secrets library which is built-in and designed for generating cryptographically strong random characters suitable for managing data such as authentication tokens. From within the TGI server, launch the Python repl with python3 command and execute this code to get the secret:

```python
import secrets

def generate_api_key(length=32):
    return secrets.token_urlsafe(length)

VLLM_API_KEY = generate_api_key()
print("Generated API Key:", VLLM_API_KEY)
# Generated API Key: qV57mQ7AlaFQAJhSJ0zq9wryKJio0yByxIS-7gr33tk
```

Next retrieve your HuggingFace token since we will be using the microsoft/Phi-3.5-vision-instruct LLM available on HuggingFace hub. Then launch the container:

```shell
# before running the container, verify you have 24GB of V-RAM
 nvidia-smi
Sat Oct  5 04:36:40 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    On  |   00000000:00:1E.0 Off |                    0 |
|  0%   25C    P8              9W /  300W |       1MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# Then run the container
 docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model microsoft/Phi-3.5-vision-instruct --trust-remote-code
Unable to find image 'vllm/vllm-openai:latest' locally
latest: Pulling from vllm/vllm-openai
43cfb69dbb46: Pull complete
fbcd35dc5bc3: Pull complete
c7232af9ae05: Pull complete
db6cdef1932a: Pull complete
56dc85502937: Pull complete
9f61b3db38d6: Pull complete
c12eb87e9588: Pull complete
62092a5c3164: Pull complete
824b951aa4fc: Pull complete
893502e49a0d: Downloading [==============================>                    ]   2.08GB/3.425GB
549334a085cb: Download complete
e56d5cf4292b: Download complete
INFO 10-04 21:59:34 api_server.py:526] vLLM API server version 0.6.1.dev238+ge2c6e0a82
INFO 10-04 21:59:34 api_server.py:527] args: Namespace(host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, model='microsoft/Phi-3.5-vision-instruct', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=True, download_dir=None, load_format='auto', config_format='auto', dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=None, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=False, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, enforce_eager=False, max_context_len_to_capture=None, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=False, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, override_neuron_config=None, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False)
INFO 10-04 21:59:34 api_server.py:164] Multiprocessing frontend to use ipc:///tmp/e35a7c54-5fe3-4c03-ba9a-03d4dc3d9864 for IPC Path.
INFO 10-04 21:59:34 api_server.py:177] Started engine process with PID 20
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-3.5-vision-instruct:
- configuration_phi3_v.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
WARNING 10-04 21:59:34 arg_utils.py:940] The model has a long context length (131072). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
WARNING 10-04 21:59:38 arg_utils.py:940] The model has a long context length (131072). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
INFO 10-04 21:59:38 llm_engine.py:226] Initializing an LLM engine (v0.6.1.dev238+ge2c6e0a82) with config: model='microsoft/Phi-3.5-vision-instruct', speculative_config=None, tokenizer='microsoft/Phi-3.5-vision-instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=microsoft/Phi-3.5-vision-instruct, use_v2_block_manager=False, num_scheduler_steps=1, multi_step_stream_outputs=False, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=True, mm_processor_kwargs=None)
INFO 10-04 21:59:38 selector.py:240] Cannot use FlashAttention-2 backend due to sliding window.
INFO 10-04 21:59:38 selector.py:116] Using XFormers backend.
/usr/local/lib/python3.12/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/usr/local/lib/python3.12/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
INFO 10-04 21:59:41 model_runner.py:1014] Starting to load model microsoft/Phi-3.5-vision-instruct...
INFO 10-04 21:59:41 selector.py:240] Cannot use FlashAttention-2 backend due to sliding window.
INFO 10-04 21:59:41 selector.py:116] Using XFormers backend.
INFO 10-04 21:59:41 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.59it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.84it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.80it/s]
INFO 10-04 22:00:12 model_runner.py:1025] Loading model weights took 7.7498 GB
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-3.5-vision-instruct:
- processing_phi3_v.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
/usr/local/lib/python3.12/dist-packages/transformers/models/auto/image_processing_auto.py:517: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
```

- --runtime nvidia: Specifies the NVIDIA runtime, enabling the use of NVIDIA GPUs within the Docker container for tasks that require significant computational power.
- --gpus all: Allows the container to access all available GPUs on the host machine, essential for GPU-accelerated processes.
- -v ~/.cache/huggingface:/root/.cache/huggingface: Mounts the host machine’s Hugging Face cache directory inside the container, facilitating faster access to frequently used model data.
- --env "HUGGING_FACE_HUB_TOKEN=<HF_TOKEN>": Sets an environment variable for the Hugging Face Hub token, necessary for authenticating and downloading models securely.
- --env "VLLM_API_KEY=<VLLM_API_KEY>": Configures an API key for accessing specific APIs that require authentication, ensuring secure API communication. This is optional.
- -p 8000:8000: Maps port 8000 on the host to port 8000 on the container, making the application hosted inside the container accessible at localhost:8000.
- --ipc=host: Uses the host’s IPC namespace, important for applications that need extensive process communication, like those running large-scale machine learning models.
- vLLM offers an official Docker image for deployment: vllm/vllm-openai:latest. The image can be used to run OpenAI compatible server and is available on Docker Hub as vllm/vllm-openai.
- --model microsoft/Phi-3.5-vision-instruct: Specifies which machine learning model to load and use within the application, targeting efficient and specific model operations.
- --trust-remote-code: The microsoft/Phi-3.5-vision-instruct model repository contains custom code that needs to be executed locally for the model to work correctly. By default, transformers library prevents executing untrusted code for security reasons, and so you must explicitly trust the remote code by setting trust_remote_code=True.

You may get the following error:

```shell
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.68 GiB. GPU 0 has a total capacity of 21.98 GiB of which 2.23 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 16.17 GiB is allocated by PyTorch, and 3.27 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

The error you're encountering is due to insufficient GPU memory, where PyTorch is unable to allocate the required memory on your GPU to run the model. There are a few potential solutions to resolve the "CUDA out of memory" issue.

- Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: As the error suggests, this configuration can help avoid memory fragmentation and make better use of available GPU memory. You can set this environment variable in your Docker run command:

```shell
docker run \
  --runtime nvidia \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model microsoft/Phi-3.5-vision-instruct \
  --trust-remote-code
```

- Reduce Model Batch Size or Max Tokens: Some models allow you to configure batch sizes or maximum token limits, which directly impact GPU memory usage. The TGI has an argument called --max_model_len. You can specify a value like --max_model_len 30000. This sets a parameter that limits the model to a maximum length of 30000, optimizing performance and resource allocation within operational constraints. Note that we are setting the maximum number of tokens to 30000 for experimental purposes using the parameter max_model_len. Although the model's context window is 128,000, we are still limited by the GPU memory of the available machine.

```shell
docker run \
  --runtime nvidia \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model microsoft/Phi-3.5-vision-instruct \
  --max_model_len 30000 \
  --trust-remote-code
```

You may get the following error now:

```shell
 File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker.py", line 483, in raise_if_cache_size_invalid
    raise ValueError(
ValueError: The model's max seq len (30000) is larger than the maximum number of tokens that can be stored in KV cache (19632). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

The error you're encountering is because the max_model_len you've set (30,000) exceeds the capacity of the Key-Value (KV) cache, which is used by the model to store intermediate activations. The KV cache's token limit is determined by available GPU memory, and since it's set at 19,632, this means you can't store 30,000 tokens in memory without adjusting some parameters.

Increase gpu_memory_utilization: You can allow the engine to use more GPU memory by increasing the gpu_memory_utilization parameter. This will let the model allocate more space for the KV cache, though it could limit how much memory is available for other processes. Here's how to specify this parameter:

```shell
docker run \
  --runtime nvidia \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model microsoft/Phi-3.5-vision-instruct \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.9 \
  --trust-remote-code
```

The --gpu_memory_utilization parameter defaults to 0.8, meaning 80% of the GPU memory is allocated for the model and its operations, leaving 20% as a buffer for system tasks or other processes. When you set --gpu_memory_utilization 0.9, you're increasing the allocation to 90% of the GPU's memory for the model, which allows for more memory to be used for operations such as the KV cache. This buffer is important to prevent GPU memory from being completely exhausted, which could cause out-of-memory (OOM) errors. However, by increasing it to 90%, you're giving the model more memory to work with at the risk of leaving less room for other processes running on the GPU.

No you should finally get successful output:

```shell
INFO 10-04 22:27:39 model_runner.py:1329] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-04 22:27:39 model_runner.py:1333] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-04 22:27:51 model_runner.py:1456] Graph capturing finished in 12 secs.
INFO 10-04 22:27:51 api_server.py:230] vLLM to use /tmp/tmpznccwmw8 as PROMETHEUS_MULTIPROC_DIR
WARNING 10-04 22:27:51 serving_embedding.py:189] embedding_mode is False. Embedding API will not work.
INFO 10-04 22:27:51 launcher.py:19] Available routes are:
INFO 10-04 22:27:51 launcher.py:27] Route: /openapi.json, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /docs, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /docs/oauth2-redirect, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /redoc, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /health, Methods: GET
INFO 10-04 22:27:51 launcher.py:27] Route: /tokenize, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /detokenize, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/models, Methods: GET
INFO 10-04 22:27:51 launcher.py:27] Route: /version, Methods: GET
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/chat/completions, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/completions, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/embeddings, Methods: POST
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Now let's explore the model.

Phi-3.5-vision is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

The model is intended for broad commercial and research use in English. The model provides uses for general purpose AI systems and applications with visual and text input capabilities which require:

- Memory/compute constrained environments
- Latency bound scenarios
- General image understanding
- Optical character recognition
- Chart and table understanding
- Multiple image comparison
- Multi-image or video clip summarization

After obtaining the Phi-3.5-vision-instruct model checkpoints, users can use this sample code for inference.

```python
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4
) 

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
for i in range(1,20):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f"<|image_{i}|>\n"

messages = [
    {"role": "user", "content": placeholder+"Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, 
  eos_token_id=processor.tokenizer.eos_token_id, 
  **generation_args
)

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
  skip_special_tokens=True, 
  clean_up_tokenization_spaces=False)[0] 

print(response)
```

Now let's try running meta-llama/Llama-3.2-11B-Vision-Instruct (as you will find out, it won't run on 24GB of GPU) 

```shell
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code

...

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.59 GiB. GPU 0 has a total capacity of 21.98 GiB of which 1.58 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 19.95 GiB is allocated by PyTorch, and 164.39 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

You might think the immediate solution is to quantize using bitsandbytes quantization algorithm. But let's also look at the context length of the model. The Llama-3.2-11B-Vision-Instruct model has a context length of 128,000 tokens, which allows it to process and understand extensive sequences of text and images during inference. This large context window supports more complex reasoning and detailed analysis, especially useful for tasks involving multimodal inputs such as image captioning and document-based queries.

How does reducing the context length reduce the memory that is required? Reducing the context length in a model like Llama-3.2-11B-Vision-Instruct reduces the amount of memory required by decreasing the number of tokens that the model needs to keep track of during computation. This has a direct impact on GPU memory usage for the following reasons:

- During forward and backward passes, the model has to store intermediate activations for each token in the context window. A longer context window means storing more activations, which consumes more memory. Reducing the number of tokens reduces the number of activations the model has to store.
- Self-Attention Mechanism: Transformers, like Llama, use self-attention, where each token interacts with all other tokens in the context window. The memory consumption of this process grows quadratically with the context length. Reducing the context length decreases the number of token interactions, thereby reducing the memory used by the self-attention mechanism.
- Buffer and Memory Allocations: Models allocate memory buffers based on the maximum sequence length, so a smaller context length requires smaller buffer allocations. This can save memory for both the model's forward computations and the backpropagation steps during fine-tuning.
- Batch Processing: With a shorter context length, it may be possible to fit larger batch sizes or more complex models into the same memory, as fewer resources are used per sequence.

Now we try a context length of 80000:

```shell
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 80000

...

WARNING 10-14 23:23:16 arg_utils.py:963] The model has a long context length (80000). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
WARNING 10-14 23:23:20 arg_utils.py:963] The model has a long context length (80000). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.

...

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.59 GiB. GPU 0 has a total capacity of 21.98 GiB of which 1.58 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 19.95 GiB is allocated by PyTorch, and 163.56 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

Notice the warning above of "The model has a long context length (80000)". 

The warning message you're seeing indicates that, even with a context length of 80,000 tokens, the system is concerned about potential out-of-memory (OOM) issues or performance degradation due to small Key-Value (KV) cache space.

Here’s what’s happening:

- KV Cache: This stores intermediate states during the model's self-attention mechanism. A longer context length increases the memory required for the KV cache, which can exhaust the available GPU memory.
- Memory Profiling Phase: During the initialization of vLLM, it profiles the memory usage to determine how much can be allocated for various components (e.g., KV cache). With such a large context length, it may require more memory than the GPU can provide.

The solution is to reduce Context Length Further: The warning suggests lowering the context length further. Try a value lower than 80,000 (e.g., 50,000 or 40,000 tokens), which can help reduce memory pressure.

```shell
# 50,000 fails too:
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 50000

# 30,000 fails too:
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000

# and notice it tries to allocate the same memory as in the case of 120,000k:
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.59 GiB. GPU 0 has a total capacity of 21.98 GiB of which 1.58 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 19.94 GiB is allocated by PyTorch, and 164.71 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

# So it appears adjusting context length is not reducing the memory requirements! 

# Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: Since the error mentions fragmentation issues, you should try setting the environment variable that PyTorch recommends:

docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000
```

### --gpu-memory-utilization
The default --gpu-memory-utilization value for the vllm/vllm-openai Docker image is 0.8, which means it uses 80% of the available GPU memory by default. This configuration allows some buffer for other processes, reducing the risk of out-of-memory (OOM) errors. You can adjust this value if needed by passing a different value when starting the container to utilize more or less GPU memory depending on the requirements of your model. This increases the memory utilization to 90%, giving your model more memory to work with while leaving a smaller buffer for system operations:

```shell
# using 90 percent GPU still caused OOM error
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.9

# using 95 percent GPU still caused OOM error
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.95
```

Mixed Precision (float16) is a technique in deep learning that involves using lower precision data types, specifically 16-bit floating point (float16), alongside 32-bit floating point (float32), to accelerate training and inference while reducing memory usage. This method leverages the computational efficiency of float16 without significantly impacting the accuracy of the model.

```shell
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.95 --dtype float16

# but even with all these options passed in now, it still tries to use the same amount of memory when loading the docker container:
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.59 GiB. GPU 0 has a total capacity of 21.98 GiB of which 1.72 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 19.94 GiB is allocated by PyTorch, and 21.49 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

The model may still require more memory than what's available on the GPU, despite optimizations. Even though float16 reduces memory consumption during runtime, the INITIAL MODEL LOADING still demands a significant amount of memory, especially for large models like Llama 3.2 11B Vision-Instruct.

### --max-model-len
In vLLM, the --max-model-len argument defines the maximum context length the model can handle, i.e., how much text or tokens the model can process within a single session. This is a critical parameter when working with large models and affects memory allocation for the key-value (KV) cache. However, there is no direct --max-seq-len argument in vLLM. Instead, --max-model-len functions similarly to what other frameworks might refer to as a maximum sequence length. This parameter limits the context window size, affecting the length of text the model can "remember" and reference during inference. Reducing this value helps decrease memory usage.

Model context length in a Large Language Model (LLM) is the maximum number of tokens or words that a model can consider when making predictions. It's similar to the model's memory or attention span. Context length is important because it allows models to understand and respond to complex information, such as in question answering, dialogue generation, and text summarization.

### --tensor-parallel-size flag
The --tensor-parallel-size flag in vLLM specifies the number of tensor parallel replicas that the model will use during execution. Tensor parallelism is a technique used to split a model's operations across multiple GPUs, allowing each GPU to process a portion of the model's tensors (e.g., weight matrices) simultaneously. This enables the model to handle larger models or workloads than a single GPU could manage on its own, by distributing the computation across multiple GPUs.

Key aspects of --tensor-parallel-size:

- Value: The value you specify represents how many GPUs will share the model's tensor computation load. For example, if you set --tensor-parallel-size 2, the model will be divided into two parts, each handled by a separate GPU.
- Purpose: It helps to maximize the use of available GPU memory and compute power when running large models. By dividing the work across GPUs, you can efficiently scale up the model size and execution speed.
- Trade-off: Using a larger tensor parallel size increases inter-GPU communication overhead but enables handling larger models that may not fit in the memory of a single GPU.

```shell
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.95 --dtype float16 --tensor-parallel-size 2

...

ValueError: The number of required GPUs exceeds the total number of available GPUs in the placement group.
Traceback (most recent call last):
```

The error message ValueError: The number of required GPUs exceeds the total number of available GPUs in the placement group indicates that the --tensor-parallel-size 2 parameter requires two GPUs, but your current setup has fewer than two available GPUs.

### --cpu-offload-gb argument
The --cpu-offload-gb argument in vLLM allows you to offload a portion of the model's data to the CPU, essentially extending the effective memory available for the GPU. This argument is particularly useful when your model size exceeds the available GPU memory.

- Offloading to CPU: By setting a value for --cpu-offload-gb, you offload part of the model's data to the CPU memory. For example, if your GPU has 24 GB of memory and you set this parameter to 10 GB, the system will utilize 10 GB of CPU memory in addition to the 24 GB of GPU memory, virtually giving you 34 GB of total memory for the model.
- Trade-offs: While this method allows for larger models to be loaded, it introduces some overhead due to the slower CPU-GPU memory transfer. Therefore, a fast CPU-GPU interconnect (such as NVLink or PCIe) is recommended to minimize the performance penalty.
- Use case: It is especially useful when you are working with models that require more memory than your GPU can handle. For example, a 13B parameter model with BF16 weights that needs around 26 GB of memory can run on a 24 GB GPU by offloading 2 GB to the CPU.
- Limitations: The performance will depend on how frequently data needs to be transferred between the CPU and GPU during inference or training. Frequent transfers may lead to significant slowdowns, so it's best suited for workloads where such transfers are minimal.

```shell
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.90 --dtype float16 --cpu-offload-gb 4 --quantization awq
```

The --cpu-offload-gb argument is not being honored: https://github.com/vllm-project/vllm/issues/9367

### --quantization

Nothing has worked insofar. I have resorted to quantization. This is the option available on vLLM website: 

--quantization, -q 
Possible choices: aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, modelopt, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, experts_int8, neuron_quant, ipex, None

Which quantization option should I use which doesn't do too much sacrifice to the data? After all, It complains it needs 3 more GiB of GPU where I only have 1 GiB remaining. So it seems close but 3 GiB more it needs.

Since your GPU memory is only a few GiB short of the requirement, and you want to minimize the loss in model performance, AWQ (Activation-aware Weight Quantization) or GPTQ (Quantized Kernels for Transformer Models) are good options. These methods tend to reduce memory usage with minimal impact on accuracy.

```shell
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.90 --dtype float16 --cpu-offload-gb 4 --quantization awq

    raise ValueError(
ValueError: Cannot find the config file for awq
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 585, in <module>
    uvloop.run(run_server(args))
  File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 109, in run
    return __asyncio.run(
           ^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "uvloop/loop.pyx", line 1517, in uvloop.loop.Loop.run_until_complete
  File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 61, in wrapper
    return await main
           ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 552, in run_server
    async with build_async_engine_client(args) as engine_client:
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 107, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 194, in build_async_engine_client_from_engine_args
    raise RuntimeError(
RuntimeError: Engine process failed to start
```

The error you're encountering regarding the "Cannot find the config file for awq" is related to vLLM expecting that the model weights are already quantized using the AWQ quantization method. vLLM assumes that when you specify a quantization method like "awq," the model you're trying to load has already been pre-quantized and includes a configuration file for that quantization method.

The post from the vLLM GitHub page explains that the model directory must contain a configuration file for the AWQ quantization. To resolve this, you would need to quantize the model using a tool like AutoAWQ, which is specifically designed to apply AWQ quantization to LLaMA and other large language models. Once the model is quantized, it will contain the necessary files that vLLM can recognize and load without errors.

Here's a brief outline of the solution:

- Quantize the model: Use a tool like AutoAWQ to apply AWQ quantization to your model.
- Ensure the config file is present: After quantization, the model directory should contain the required quantization configuration file.
- Load the quantized model: With the model properly quantized, vLLM should now be able to load the model without throwing the "Cannot find the config file" error.

```shell
pip install autoawq

ERROR: Could not find a version that satisfies the requirement autoawq (from versions: none)
ERROR: No matching distribution found for autoawq

pip install --upgrade pip setuptools wheel
pip install autoawq

ERROR: Could not find a version that satisfies the requirement autoawq (from versions: none)
ERROR: No matching distribution found for autoawq

pip config list

pip config set global.index-url https://pypi.org/simple

pip config list
global.index-url='https://pypi.org/simple'

pip install autoawq

ERROR: Could not find a version that satisfies the requirement autoawq (from versions: none)
ERROR: No matching distribution found for autoawq

# First, ensure that autoawq is compatible with your Python version (3.12). Since autoawq is a relatively new package, it might not yet have been built for Python 3.12.

# similar errors
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.90 --dtype float16 --cpu-offload-gb 4 --quantization aqlm

# similar errors
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.90 --dtype float16 --cpu-offload-gb 4 --quantization gptq

# similar errors
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.90 --dtype float16 --cpu-offload-gb 4 --quantization bitsandbytes --load-format bitsandbytes
```

After a 5 hour struggle, I tried the g5g.16xlarge which offers 4 GPUS, each with 16GiB of memory. But it didn't distribute the workload correctly. I tried adding --disable-custom-all-reduce but that was to no avail:


```shell
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.90 --dtype float16 --disable-custom-all-reduce --tensor-parallel-size 2


watch -n 1 nvidia-smi

Tue Oct 15 10:18:48 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000000:00:1B.0 Off |                    0 |
| N/A   30C    P0             31W /   70W |    5521MiB /  15360MiB |     22%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       On  |   00000000:00:1C.0 Off |                    0 |
| N/A   28C    P0             31W /   70W |    5521MiB /  15360MiB |     28%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  Tesla T4                       On  |   00000000:00:1D.0 Off |                    0 |
| N/A   28C    P0             32W /   70W |    5521MiB /  15360MiB |     27%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
| N/A   28C    P0             32W /   70W |    5521MiB /  15360MiB |     15%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     33171      C   /usr/bin/python3                             5518MiB |
|    1   N/A  N/A     33435      C   /usr/bin/python3                             5518MiB |
|    2   N/A  N/A     33436      C   /usr/bin/python3                             5518MiB |
|    3   N/A  N/A     33437      C   /usr/bin/python3                             5518MiB |
+-------------------------------------------

# But then at the end it uses only 1 and crashes out at 100 percent

```

g5.12xlarge

Visit Instances > Spot Requests

Total Spot cost (USD)
$2.70
You saved 52%

 Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1

docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --max-model-len 30000 --gpu-memory-utilization 0.90 --dtype float16 --disable-custom-all-reduce --tensor-parallel-size 4

Every 1.0s: nvidia-smi                                                             ip-172-31-64-123: Tue Oct 15 10:55:23 2024

Tue Oct 15 10:55:23 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    On  |   00000000:00:1B.0 Off |                    0 |
|  0%   23C    P8             10W /  300W |       1MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A10G                    On  |   00000000:00:1C.0 Off |                    0 |
|  0%   23C    P8             41W /  300W |       1MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A10G                    On  |   00000000:00:1D.0 Off |                    0 |
|  0%   23C    P0             40W /  300W |       1MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A10G                    On  |   00000000:00:1E.0 Off |                    0 |
|  0%   22C    P0             28W /  300W |       1MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

