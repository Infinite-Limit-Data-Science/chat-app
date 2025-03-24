### Overview

Using embedding models with larger model weights will continue to pose challenges and invite novel architectural considerations when managing finite memory limits. This essay covers memory considerations with the ever growing demand for larger language models.

Classical statistical tests use precise mathematical calculations to provide direct answers to real-world questions, e.g. does x significantly affect the output of y? You create a null hypothesis, perhaps an alternative hypothesis, for a two-tail test and rely on well-established probability distributions, such as normal distribution, standard normal distribution (e.g. for z-tests), and so on. Determine a confidence interval, significant level, and p-value and quickly get answers. To execute a classical test, from chai-squared to anova or from z to t test would take seconds on any PC. Yes, it involves deep analytical rigor but the data sets are often small and well-structured (e.g. population mean, population standard deviation, z-score, p-value, etc).

But transformer models are intended to engage a different kind of problem. Even beyond classical statistical testing, there are a plethora of traditional machine-learning approaches, including, but not limited to, linear regression, logistic regression, Naive Bayes, random forest, k-means, hierarchical clustering, etc. Transformers are built to handle massive numbers of parameters. Unlike classical statistical testing, they make fewer explicit assumotions about data distribution and instead rely on large datasets to build generalizations. 

### Training

Transformers have two phases, training and inference. Training relies on iterative fitting. With any neural network, iteration occurs in the forward pass of the network layers. In transformers, the forward pass is composed of a multi-head self-attention mechanism and a position-wise feed forward network. In training, the goal is to find model parameters that produce accurate outputs on new, unseen data. Transformers process entire sequences (or batches of sequences) in a single forward pass rather than stepping through tokens one by one (as a traditional Recurrent Neural Network (RNN) would).

In Transformers, each token is mapped to a high dimensional vector wherein the features of the vector are learned over iterative fitting. This high dimensional vector is stored in the token embeddings table and associated with the token id in that table. When a high dimensional vector is learned during the embedding phase of a forward pass, a positional embedding (or encoding) is also added in a positional embedding table. This process also happens during the embedding layer of the forward pass of the neural network in training. The result is a continuous vector representation of the token, often in hundreds or thousands of dimensions. 

Because Transformers do not have a built-in notion of sequence order (unlike RNNs), we add positional information to each token’s embedding. This may be done by a learned positional embedding or a fixed sinusoidal positional encoding. Either way, it injects positional context.

The sum of the token embedding and the positional embedding is passed into the Transformer blocks (self-attention + feedforward), all as part of the forward pass. Throughout training, these embedding parameters (both token and positional, if the positional embeddings are learned) are updated along with the rest of the network’s parameters via backpropagation. 

Backpropagation itself is the algorithm for computing gradients with respect to the network parameters. Backpropagation is simply a systematic way of computing how much each individual parameter (weights, biases, etc.) in a neural network contributes to the overall error (the loss). Since modern neural networks often have millions (or even billions) of parameters, we need an efficient procedure to figure out the gradient of the loss with respect to each parameter. This is exactly what backpropagation does.

The Core Idea: The Chain Rule
START HERE

By itself, it doesn’t inherently “rely” on a particular loss function or optimization method, but in practice:
- Loss Function (often cross-entropy)
    - You define a loss function (e.g., cross-entropy) that measures how far off the model’s predictions are from the target.
    - Backpropagation requires a differentiable loss function so it can compute partial derivatives.
- Optimization Method (often SGD or a variant)
    - Once backpropagation calculates the gradients, you apply an optimizer (e.g., SGD or Adam) to update the parameters.
    - These optimizers use the gradients to adjust the parameters in a direction that reduces the loss.

Cross-Entropy loss is a measure of how “wrong” the model’s probability distribution is compared to the actual outcomes (labels) in your data.
It isn't using z-tests, z-scores, confidence interval, significance level, p-values, central limit theorem to know whether to reject or fail to reject the null hypothesis. Cross-entropy does not use classical hypothesis test, such as z-tests, t-tests, and chi-squared tests. Instead, it’s a direct measure of the difference between two probability distributions:
- The model’s predicted distribution over possible outcomes (for instance, which token comes next?), and
- The true distribution (which in a supervised dataset is typically 100% on the correct label/token).

Formally, if the model outputs a probability distribution p for the next token and the ground truth distribution is q (e.g., the correct token with probability 1, or a “one-hot” vector), then the cross-entropy is:

$$
H(q,p) = - \sum_{x} q(x) \log p(x)
$$

No p-values, z-scores, or explicit use of the central limit theorem are involved here. Instead, the model just directly sees how well its predicted probabilities match reality:
- If p(x) is high for the actual label x, the cross-entropy is lower (the model is doing well).
- If p(x) is low for the actual label x, the cross-entropy is higher (the model is doing poorly).

In a z-test, you usually have one main statistic (the difference of means, for instance) and you know (or assume) its approximate distribution if the null hypothesis is true. In deep learning, you have millions (or billions) of parameters. The cross-entropy is simply a scalar objective that the optimizer tries to push down, but there’s no simple, “this is distributed as a Z under the null” derivation. We don’t do a one-time test; we do a long optimization procedure over enormous data. Neural network training is an iterative process: in each gradient update, you see a small batch, compute cross-entropy, adjust parameters. Over many epochs, you (hopefully) converge to a model that generalizes well. There’s no single “reject/fail to reject” moment. 

In a typical Transformer training loop, the steps are:
- Forward pass: Feed input through the model; compute cross-entropy loss.
- Backward pass (backpropagation): Compute the gradient of the loss w.r.t. all model parameters.
- Parameter update (via SGD or similar): Adjust parameters using the gradient information.

Multi-head self-attention in each layer considers all tokens in the sequence at once to compute attention-weighted representations.

Depending on the task (e.g., next-token prediction, sequence classification), the model outputs a probability distribution or some other target for each token or for the entire sequence. A loss function (commonly cross-entropy) is then computed over the model’s outputs for all tokens in the sequence (or all sequences in the batch).

Gradients from the loss are propagated back through all the Transformer layers. Parameters are updated (e.g., via stochastic gradient descent or Adam). This is iterative fitting. Iterative fitting is the process of repeatedly adjusting a model’s parameters so that its predictions get closer and closer to the actual data. Here’s a simple analogy:
- Initial Guess: You start with random guesses for how the model should behave (like randomly chosen parameters).
- Check Error: You see how far off the model’s predictions are from the real answers.
- Adjust: You tweak the parameters to reduce that error slightly.
- Repeat: You go back to Step 2 with the updated model and do it again on more data.
- Over many rounds of this “guess, check, adjust” cycle (like practicing a skill repeatedly), the model’s parameters slowly converge to values that produce better and better predictions. This is fundamentally what happens in gradient descent training for neural networks.

Because the Transformer’s self-attention mechanism scales with the length of the sequence, the whole sequence must be present to compute the attention distribution among tokens during that forward pass. This parallel sequence processing—rather than iterating token-by-token—makes Transformers extremely efficient on modern hardware, though at the cost of needing sufficient memory to handle the entire sequence at once.

The memory used in training of course is not applicable for inference. But we still need to load the model learned weights into memory during inference. This includes token embedding, positional embedding, forward pass weights, such as self-attention and feed forward, as well as final output projections. Larger models will require more model weights which increase throughput and latency during inference.

### Model Weights: Token Embeddings

The Token Embedding Table maps each token ID from the vocabulary to a continuous vector representation. The token id is often a word or subword from the vocabulary. For example, the token “the” may be mapped to the ID 464. Platforms like vLLM and HuggingFace TGI load a transformer tokenizer in the model to convert raw text to token ids and back. The trained model will only work with the token ids and not raw text. Hence, when inference platforms like vLLM or Hugging Face TGI run, they download model card configs from hugging face and store them internally. These configs include config.json, special_tokens_map.json, tokenizer_config.json, and tokenizer.json. The tokenizer must convert all text to token ids. When the model sees a token ID such as 464 at runtime, that will correspond to an English word like “the”, even though the mode doesn't understand "the". 

The token ids in the token emebdding table do not need to be in any particular order, since it is the embedding vectors that convey meaning and semantic similarity. However, different tokenizers order token ids differently. For example, in a Byte-Pair Encoding (BPE) or WordPiece tokenizer, the algorithm might merge subwords by frequency or other heuristics, and assign lower IDs to more frequent tokens. But there is no strict rule that “token 0 must be [PAD]” or “token 464 must be the” across all models—each tokenizer might have a different mapping.

It is the role of the token embedding table to map all tokens to vectors whose dimensionality matches the model’s internal hidden dimension. In the case of vlm2vec-full, the hidden dimension is 3072. In effect, all tokens will have an associated 3072-dimensional embedding vector. The fact that 3072 is quite large just reflects the architecture’s capacity. It allows the model to store richer contextual and semantic information for each token than if the dimension were smaller.
 
While each individual dimension isn’t interpretable on its own, taken together, the dimensions encode patterns of usage and context—syntactic, semantic, even morphological traits—that help the model distinguish how tokens behave in language. Essentially:
- High Dimensionality: The large number of dimensions gives the model a broad “budget” of representational capacity
- Learned, Not Predefined: The vectors are learned from data
- Contextual and Semantic Nuances: Tokens with similar usage and meaning often end up with similar embeddings, because the model groups them according to how they co-occur in context.
Hence, although the embedding matrix itself is just a big table indexed by token IDs, each row (embedding) is a rich, learned representation that encodes diverse nuance about that token’s role in language.

Each model has its on vocabulary length. The length of the vocabulary is determined by the vocab_size in the transformer's model card config.json file. For example, for vlm2vec-full, vocab_size is 32064. The shape of the matrix of the token embedding table is [32064, 3072] for vlm2-vec full. Index i in this matrix corresponds to the embedding for the i-th token in the vocabulary. The table does not store the literal text “the” (or any other strings). Instead, each row in the table is just a vector of learned numbers (floating-point parameters). Row index i (for example, 464 for “the”) corresponds to a 3072-dimensional vector (like [0.12, -0.03, 0.78, ...]) that the model learns to represent the concept of that token.

### Model Weights: Positioanl Embeddings

But the token embedding table only provides semantic structure to individual tokens. It does not provide sequence ordering. Token embeddings answer the question “what token is this?” but not “where does this token appear in the sentence?” Sequence ordering captures its own meanings. For example, we can say “I love cats” or “cats love me”? Both share the tokens “I,” “love,” “cats,” but they have fundamentally different meanings because the tokens appear in a different order.

There are so many ways to arrange words in a sentence that express different meanings. If we were to arrange words all the different ways possible, we will have an infinite number of permutations. Indeed, there are infinitely many possible word permutations—but a Transformer (or any language model) does not store every permutation. Instead, it learns generalizable patterns of how word order affects meaning. In fact, the positional emebdding matrix of vlm2vec-full, for example, is [131072, 3072]. Thus, a lot of features are captured to understand what it is like to be in position i in the input sequence.

Transformers are permutation-invariant. Thus, another reference table, the positional embedding, is exercised and has its own model weights that must be loaded into memory. The positional embedding answers the question “where does the token appear in the input sequence?”  For vlm2vec-full, the positional embeddings is a matrix of type [131072, 3072]. Thus, a lot of features are captured to understand what it is like to be in i-th position in the input sequence.

2. Computing Attention Scores: START HERE

Ultimately, an additive property merges token embedding + positional embedding so each token knows both what it is and where it’s located in the sentence. These token embedding representations are later referenced in the self-attention mechanism and feed forward hidden layers in a forward pass of the neural network.
In addition to the frozen model weights from training, self-attention, during inference, has its own weights. The tokenizer loaded in the inference platform, in this case vLLM, converts the text to tokens. These tokens are passed into the neural network. At the very start of the forward pass, the model performs a summation of the token embedding and positional embedding using the relevant input token. Hence, the input token assumes a new representation based on the model’s trained token and positional embedding weights. In effect, first the model looks up each token’s token embedding (from the token embedding table) and corresponding positional embedding (from the positional embedding table) and then performs the summation of the two embeddings to create a new embedding representation of the token. This process is known as the embedding layer or embedding step of the forward pass.
This sum is the initial representation of that token embedding (including content + position). This initial representation is a vector - specifically, a high-dimensional float vector of size hidden_dim. For instance, if your model’s hidden_dim is 3072, then the initial representation for each token will be a 3072-dimensional vector. The dimensions of the vector never changes during the lifecycle of the neural network. The newly-represented embedding vector is what is passed into the first transformer block. All tokens in the input sequence are converted to their vector representations at the same time and all pass into the first transformer block in parallel. All in all, the embeddings are summed elementwise for each token, resulting in a batch of new token representations (each a 3072-dimensional vector) that proceed into the first Transformer block.
What is the “transformer block”? A Transformer block, whether in an encoder or decoder setting, consists of two main sublayers, Multi-Head Self-Attention and Feed-Forward Network (FFN).
Why do we need to do self-attention in the forward pass during inference? Self-attention allows each token in the sequence to “look at” (or attend to) other tokens to gather context. The model computes self-attention to update each token’s representation based on all the other tokens. Simply looking up token embeddings (and positional embeddings) only gives you a basic representation of “which token, at which position.” Self-attention goes deeper. It captures how each token relates to other tokens in the input sequence, not just in the token embedding and positional embedding tables, e.g. “Does ‘cat’ modify ‘sat,’ or does something else happen?” The input sequence has this information. Remember the frozen weights in our token and positional embeddings table gathered during training cannot capture every possible permutation of word ordering. The weights represent generalized patterns and probabilities. The input sequence captures what the user asked.
In self-attention, each token’s current representation (derived from the token and positional embeddings table) is transformed into three distinct vectors called a “query” (Q), a “key” (K), and a “value” (V). These vectors are crucial to letting each token “look at” other tokens in the sequence and decide how much information to pull from them. The query vector represents the information you are looking for. The key vector represents what content you contain. The value vector represents the actual information to be contributed if the attention mechanism decides the key vector is relevant to the query vector. For each token i, we compare its query vector with every other token’s key vector using a dot product:
