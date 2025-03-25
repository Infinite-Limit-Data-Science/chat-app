### Overview

Using embedding models with larger model weights will continue to pose challenges and invite novel architectural considerations when managing finite memory limits. This essay covers memory considerations with the ever growing demand for larger language models.

Classical statistical tests use precise mathematical calculations to provide direct answers to real-world questions, e.g. does x significantly affect the output of y? You create a null hypothesis, perhaps an alternative hypothesis, for a two-tail test and rely on well-established probability distributions, such as normal distribution, standard normal distribution (e.g. for z-tests), and so on. Determine a confidence interval, significant level, and p-value and quickly get answers. To execute a classical test, from chai-squared to anova or from z to t test would take seconds on any PC. Yes, it involves deep analytical rigor but the data sets are often small and well-structured (e.g. population mean, population standard deviation, z-score, p-value, etc).

But transformer models are intended to engage a different kind of problem. Even beyond classical statistical testing, there are a plethora of traditional machine-learning approaches, including, but not limited to, linear regression, logistic regression, Naive Bayes, random forest, k-means, hierarchical clustering, etc. Transformers are built to handle massive numbers of parameters. Unlike classical statistical testing, they make fewer explicit assumotions about data distribution and instead rely on large datasets to build generalizations. 

### Training

Transformers have two phases, training and inference. Training relies on iterative fitting. With any neural network, iteration occurs in the forward pass of the network layers. In training, the goal is to find model parameters that produce accurate outputs on new, unseen data. Transformers process entire sequences (or batches of sequences) in a single forward pass rather than stepping through tokens one by one (as a traditional Recurrent Neural Network (RNN) would).

Training comprising three critical steps:
- Model Initialization
- Forward Pass
- Backward Pass and Parameter Update

### Training: Tokenization

Tokenization is a preprocessing step that converts raw text into token IDs (integers) using a particular vocabulary. This is done independently of the model. You can do this any time you have raw text ready—before passing it into the model’s forward pass.

vLLM and Hugging Face TGI/TEI use a transformer tokenizer loaded from hugging face hub, in the case of a hugging face model. They download the model card files. The model card file specifies the exact tokenizer to use. The tokenizer loaded in the inference platform, in this case vLLM, converts the text to tokens. These tokens are passed into the neural network, since the model itself only understands token ids and not raw text.

### Training: Model Initialization

When you first instantiate the Transformer (or any neural network), you allocate memory for all trainable parameters—including the token and positional embedding tables. These tables are usually randomly initialized or loaded from a pretrained checkpoint (e.g., if you’re using a pretrained language model). If you’re training a model from scratch, those initial token and positional embedding values are typically randomized and don’t carry any semantic meaning. Over the course of training, backpropagation will adjust these vectors so that tokens with similar meanings end up with similar embedding vectors. By the end, the once-random embeddings become rich, learned representations that help the network minimize its loss. However, if you’re loading a pretrained model, the token and positional embeddings will already contain meaningful values learned in a previous training phase. In that case, they’re not random but come from the pretrained checkpoint.

Each token id in the embedding tables is a high-dimensional vector. In the case of vlm2vec-full, it is a 3072-dimensional vector. Eventually, these vectors will contain semantic information about the token ids. The positional embedding table is indexed by position (sometimes called position IDs), not by token IDs. Think of it this way:
- Token Embedding Table
    - Index: Token ID (from the vocabulary).
    - Output: A learned vector that captures what the token is.
- Positional Embedding Table
    - Index: Position ID (0, 1, 2, … up to the maximum sequence length).
    - Output: A learned vector that captures where in the sequence the token occurs.
So, if a token is at position i in the input, we do a lookup in the positional embedding table at index i. This returns the vector that says “this is the representation of being at position i.” That vector is then added to the token embedding vector.

### Training: Model Initialization: Token Embedding Table

The Token Embedding Table maps each token ID from the vocabulary to a continuous vector representation. The token id is often a word or subword from the vocabulary. For example, the token “the” may be mapped to the ID 464. Platforms like vLLM and HuggingFace TGI load a transformer tokenizer in the model to convert raw text to token ids and back. The trained model will only work with the token ids and not raw text. Hence, when inference platforms like vLLM or Hugging Face TGI run, they download model card configs from hugging face and store them internally. These configs include config.json, special_tokens_map.json, tokenizer_config.json, and tokenizer.json. The tokenizer must convert all text to token ids. When the model sees a token ID such as 464 at runtime, that will correspond to an English word like “the”, even though the mode doesn't understand "the". 

The token ids in the token emebdding table do not need to be in any particular order, since it is the embedding vectors that convey meaning and semantic similarity. However, different tokenizers order token ids differently. For example, in a Byte-Pair Encoding (BPE) or WordPiece tokenizer, the algorithm might merge subwords by frequency or other heuristics, and assign lower IDs to more frequent tokens. But there is no strict rule that “token 0 must be [PAD]” or “token 464 must be the” across all models—each tokenizer might have a different mapping.

It is the role of the token embedding table to map all tokens to vectors whose dimensionality matches the model’s internal hidden dimension. In the case of vlm2vec-full, the hidden dimension is 3072. In effect, all tokens will have an associated 3072-dimensional embedding vector. The fact that 3072 is quite large just reflects the architecture’s capacity. It allows the model to store richer contextual and semantic information for each token than if the dimension were smaller.
 
While each individual dimension isn’t interpretable on its own, taken together, the dimensions encode patterns of usage and context—syntactic, semantic, even morphological traits—that help the model distinguish how tokens behave in language. Essentially:
- High Dimensionality: The large number of dimensions gives the model a broad “budget” of representational capacity
- Learned, Not Predefined: The vectors are learned from data
- Contextual and Semantic Nuances: Tokens with similar usage and meaning often end up with similar embeddings, because the model groups them according to how they co-occur in context.
Hence, although the embedding matrix itself is just a big table indexed by token IDs, each row (embedding) is a rich, learned representation that encodes diverse nuance about that token’s role in language.

Each model has its on vocabulary length. The length of the vocabulary is determined by the vocab_size in the transformer's model card config.json file. For example, for vlm2vec-full, vocab_size is 32064. The shape of the matrix of the token embedding table is [32064, 3072] for vlm2-vec full. Index i in this matrix corresponds to the embedding for the i-th token in the vocabulary. The table does not store the literal text “the” (or any other strings). Instead, each row in the table is just a vector of learned numbers (floating-point parameters). Row index i (for example, 464 for “the”) corresponds to a 3072-dimensional vector (like [0.12, -0.03, 0.78, ...]) that the model learns to represent the concept of that token.

### Training: Model Initialization: Positional Embedding Table

Transformers are permutation-invariant. They don't capture sequence ordering of inputs by default. At the same time, the token embedding table only provides semantic structure to individual tokens. It does not provide sequence ordering. Token embeddings answer the question “what token is this?” but not “where does this token appear in the sentence?” Sequence ordering captures its own meanings. For example, we can say “I love cats” or “cats love me”? Both share the tokens “I,” “love,” “cats,” but they have fundamentally different meanings because the tokens appear in a different order.

There are so many ways to arrange words in a sentence that express different meanings. If we were to arrange words all the different ways possible, we will have an infinite number of permutations. Indeed, there are infinitely many possible word permutations—but a Transformer (or any language model) does not store every permutation. Instead, it learns generalizable patterns of how word order affects meaning. In fact, the positional emebdding matrix of vlm2vec-full, for example, is [131072, 3072]. Thus, a lot of features are captured to understand what it is like to be in position i in the input sequence.

Just as with the input embedding table, the positional embedding has its own model weights that must be loaded into memory. The positional embedding answers the question “where does the token appear in the input sequence?”  For vlm2vec-full, the positional embeddings is a matrix of type [131072, 3072]. Thus, a lot of features are captured to understand what it is like to be in i-th position in the input sequence.

### Training: Forward Pass

**The Forward Pass uses the current values of those embedding tables, even when they start with random nonsense values. Those "current values" are our model weights. For each token in your input, the forward pass looks up the embedding vector from the token table and the positional table, sums them, and sends that to the Transformer blocks.** Let's be more specific. Because the embedding tables are indexed by token ID (and position ID), the model always knows which row to use for a given input token—even if those rows initially contain “nonsense” values. Here are the steps:
- Tokenizer Assigns IDs
    - For each word/subword in your input, the tokenizer maps it to a specific integer token ID.
- Index into the Embedding Table
    - The token ID serves as an index into the token embedding matrix. Even if the vector stored there is initially random, the model will fetch that exact row by index.
- Index into the Positional Embedding Table
    - The position i (0, 1, 2, …) in the input sequence (remember the input sequence represents the "user prompt") also serves as an index into the positional embedding matrix. In fact, it isn't even a user prompt. It is a corpus of training data we use to train the model to understand word meaning and position meaning in sentences. As we pass more and more input sequences, we can begin to develop the definition of what it means to be in position i.
- Sum the Two Vectors
    - The token vector + positional vector becomes your “initial representation” of that token at position i so each token knows both what it is and where it’s located in the sentence. Effectively, a new vector is created representing the input token.
    - The new vector will always be the same dimension as what is in the embedding tables.  For instance, if your model’s hidden_dim is 3072, then the initial representation for each token will be a 3072-dimensional vector. The dimensions of the vector never changes during the lifecycle of the neural network. 
In other words, the process is purely positional lookup—the model uses the token ID and position ID as direct indices. It does not rely on any pre-existing “semantic meaning” at initialization. Over many training iterations, backpropagation adjusts these “nonsense” embeddings into meaningful vectors that help minimize the loss. 

The newly-represented embedding vector is what is passed into the first transformer block. All tokens in the input sequence are converted to their vector representations at the same time using the token and positioanl embedding tables and all pass into the first transformer block in parallel. **It is important to emphasize this: the Summation of TokenEmbedding(tokenID) + PositionalEmbedding(i), this summed vector, is what enters the Transformer blocks, nothing else.**

What is the “transformer block”? A Transformer block, whether in an encoder or decoder setting, consists of two main sublayers, Multi-Head Self-Attention and Feed-Forward Network (FFN).

### Training: Forward Pass: Multi-Head Self-Attention

Multi-Head Self-Attention is the first processing of input vectors in the Forward Pass of a transformer block. Multi-Head Self-Attention can be broken into two parts: (1) computing “Scaled Dot-Product Attention” for one head, and (2) repeating this process over multiple heads in parallel.

1. Scaled Dot-Product Attention (Single Head)

a) Input Embeddings

Each token in your sequence starts with some embedding vector $x_i$. In an encoder or decoder block, these embeddings could be:
- The original token embedding + positional embedding (in the first layer)
- The output of the previous layer (in deeper layers).

Throughout these steps, we’ll assume you have a sequence of length n: $x_1, x_2, \ldots, x_n$. 

b) Linear Projections to Q, K, and V

For each token i, we produce three vectors:
$$
Q_i = x_i W_Q,\quad
K_i = x_i W_K,\quad
V_i = x_i W_V.
$$
$W_Q, W_K, W_V$ are learned parameter matrices. Each $Q_i, K_i, V_i$ typically has a dimension called $d_k$ which might be smaller than the original embedding dimension.

Intuition
- $Q_i$ (Query) represents “what information token i is seeking.”
- $K_j$ (Key) represents “the content token j holds.”
- $V_j$ (Value) is “the information token j will actually contribute if it’s attended to.”

c) Compute Attention Scores (Query ⋅ Key)

For each token i, we compare its query $Q_i$ to all keys in the sequence $K_1, K_2, \ldots, K_n$ via dot products:
$$
\text{score}(i, j) = Q_i \cdot K_j
$$ 
These raw similarity scores tell us how relevant token j is to token i.

d) Scale and Softmax

To keep the dot products in a stable range, we typically divide by $\sqrt{d_k}$ (the dimension of $k_j$):
$$
\alpha_{i,j} 
= \mathrm{Softmax}\!\Bigl(
  \frac{Q_i \cdot K_j}{\sqrt{d_k}}
\Bigr).
$$
- The Softmax ensures that $\sum_{j} \alpha_{i,j} = 1$
- Each $\alpha_{i,j}$ is called the “attention weight” from token i to token j.

**Wait, hold on a second. What is Softmax doing? When we compare a particular token’s query vector $Q_i$ with the key vectors $K_j$ for all tokens j in the sequence, we get a set of raw similarity scores ($Q_i ⋅ K_j$). The softmax function is then applied to these scores to transform them into a set of nonnegative weights that sum to 1. In other words, we are creating probabilties. After applying softmax, each score $\alpha_{i,j}$ becomes a number in the range [0,1]. Sum to 1: $\sum_{j} \alpha_{i,j} = 1$. You can interpret  $\alpha_{i,j}$ like a “probability” that token i attends to token j.**

e) Weighted Sum of Values

Finally, we create the new representation for token i, denoted $z_i$, by weighting each value $V_j$ by $\alpha_{i,j}$.
$$
z_i = \sum_{j=1}^{n} \alpha_{i,j} \, V_j.
$$
This $z_i$ is a contextually updated vector for token i. It blends information from all other tokens based on their relevance to i.

2. Multi-Head Attention

In practice, Transformers do Scaled Dot-Product Attention several times in parallel “heads,” each with different projection matrices $W_Q^{(h)}, \quad W_K^{(h)}, \quad W_V^{(h)}.$

a) Multiple Heads

Each head h produces its own set of $\{\, z_1^{(h)}, \dots, z_n^{(h)} \}$. The intuition: Different heads can learn to focus on different types of relationships. For example, one head might learn to track verb–object relations, another might focus on synonyms or long-range dependencies, etc.

b) Concatenate and Project

After all heads compute their $z_i^{(h)}$ vectors, we concatenate the results for each token i:
$$
z_i^{\mathrm{multi}} = \mathrm{Concat}\bigl(z_i^{(1)},\, z_i^{(2)},\, \ldots,\, z_i^{(H)}\bigr).
$$
Then we pass this concatenated vector through another linear projection (often denoted $W_O$):
$$
z_i^{\mathrm{output}} = z_i^{\mathrm{multi}} \, W_O.
$$
That final $z_i^{\mathrm{output}}$ is what leaves the multi-head attention sublayer for token i. It then goes on to the residual connection + layer normalization, and ultimately to the feed-forward sublayer in the Transformer block.

Summary:
- Queries, Keys, Values: Each token embedding is projected into three different vectors, capturing:
    - Query: “What info am I looking for?”
    - Key: “What info do I contain?”
    - Value: “The actual info to use if my key matches a query.”
- **Scaled Dot-Product Attention: For each token’s query, we measure similarity with every token’s key, use softmax to get attention weights, and then create a weighted sum of the values.**
- **Multi-Head: We do the above process multiple times (heads) in parallel, each with its own set of trainable parameters. This enables the model to capture different types of relationships at once.**
- Concatenate & Project: We stack each head’s output, then do a final linear projection to produce the sublayer’s output.

Why do we do all this? So each token’s representation can integrate information from the entire sequence, focusing on what’s most relevant in context—thus enabling the model to capture rich, long-range dependencies without relying purely on recurrence or convolutions. In the end, the model computes self-attention to update each token’s representation based on all the other tokens. It captures how each token relates to other tokens in the input sequence. Note it is actually backward propagation that updates the token and positional embedding tables to give meaning to vectors. While self-attention is the mechanism that discovers relationships among tokens (and thereby influences what good embeddings might look like), the actual updating of token/positional embeddings happens because backpropagation flows error signals all the way back to those embedding parameters. Over time, these updates make the embeddings more “meaningful.”

### Training: Forward Pass: Residual Connection + Layer Normalization (Post-Attention)

A residual connection (also called a skip connection) simply means we add the original input to the sublayer’s output, rather than relying on the sublayer’s output alone. In the case of multi-head attention, if:
- $x_i$ is the input representation for token i (which might be either the original token embedding + positional embedding in the very first layer, or the output of the previous layer in deeper layers),
- $z_i^{(\mathrm{multi\text{-}head})}$ is the result of the multi-head attention sublayer for token i, then the residual connection is:
$$
\text{AttentionOutput}_i = z_i^{(\mathrm{multi\text{-}head})} + x_i.
$$

Why do we add $x_i$ back? 
- Preventing “forgetting” of the original representation
    - Without a skip connection, the only signal token i would send forward is the output from the multi-head attention sublayer. If the sublayer’s transformation is not beneficial (or partially harmful) for a particular token, the network might lose the crucial original information $x_i$. 
    - By addding $x_i$ back in, the model always retains a direct path to the original representation—even if the attention output is weak or unhelpful. This stabilizes learning and preserves more information.
- Improving gradient flow (ease of training)
    - Deep neural networks can suffer from vanishing or exploding gradients, especially when stacking many layers. Residual connections give gradients a more direct route back through layers.
    - Instead of each layer being responsible for completely transforming its input, the layer effectively outputs “input + some learned correction.” This makes training deeper networks more feasible.
- Empirical performance
    - Residual connections (popularized by ResNet in computer vision) are now standard practice in Transformers because they significantly improve both training stability and final accuracy.

In addition to the REsidual Connection, we have Layer Normalization. Layer normalization: Normalize the result across the hidden dimension to stabilize training and keep values in a consistent range:
$$
\text{NormedOutput} = \mathrm{LayerNorm}\bigl(\text{AttentionOutput}\bigr).
$$

### Training: Forward Pass: Position-wise Feed-Forward

In the attention sublayer, we do a lot of “mixing” across tokens (each token attends to every other token). In contrast, the feed-forward sublayer is position-wise—meaning it applies the same MLP to each token separately. “MLP” stands for Multi-Layer Perceptron, which is essentially a feed-forward neural network composed of one or more fully connected (dense) layers. In an MLP:
- Inputs are multiplied by a weight matrix (plus a bias) and then passed through a nonlinear activation function (e.g., ReLU).
- Multiple layers can be stacked, where each layer’s output is the next layer’s input.
- No recurrence or convolution is used, so it’s called “feed-forward”—information flows strictly from the input layer to the output layer.

"Inputs are multiplied by a weight matrix (plus a bias)". When you define your model (e.g., a torch.nn.Linear layer in PyTorch), the framework creates a matrix (for $W_1$) and a vector (for $b_1$) of the specified sizes. These numbers are initialized (e.g., randomly) but have no intrinsic meaning yet. But these weights are associated specifically with the feed forward.
- “Feed-forward weights” simply means: “the trainable parameters used in the feed-forward sublayer.”
- “Backpropagation” is how all trainable parameters in the model (including those feed-forward weights) get updated during training.
In other words, the name “feed-forward weights” refers to where in the network those parameters live—namely, the part of the Transformer that performs the feed-forward transformation. The mechanism for how they are learned (i.e., through backpropagation and gradient-based optimization) is a separate issue.

Suppose after layer normalization, each token in your sequence has a representation $h_i \in \mathbb{R}^d$.
- $h_i$ is a generic symbol for “the current representation of token i". For example, the output of a self-attention sublayer. Original Input Vector $x_i$ denotes the token + positional embedding for token i. After Self-Attention: That same token i gets a new representation $h_i$ by combining information from other tokens. Because self-attention “mixes” content from all tokens, we can think of $h_i$ as the “mutated” or updated version of $x_i$ enriched with context.
- ∈ is just “belongs to” or “is an element of.”
- $R^d$ means “the set of all d-dimensional vectors with real-valued entries. For example, if d is 3072, then $h_i ∈ R^3072$ is just a 3072-dimensional real vector. 

The feed-forward sublayer applies the same 2-layer MLP to each $h_i$, resulting in a new token vector $(f(h_i))$. No token-to-token interaction happens here; each $h_i$ is processed in isolation. A typical feed-forward sublayer can be written as:
$$
f(h) 
= \max\bigl(0,\, h W_1 + b_1\bigr)\, W_2 + b_2,
$$
- $W_1$ and $b_1$ are trainable weights and biases for the first linear layer. “Trainable weights” (often just called parameters) are the numbers in a neural network that the training process adjusts via backpropagation to minimize the loss. In the above formula, $W_1$ and $b_1$ are one set of trainable weights (a matrix and a bias vector). 
- $W_2$ and $b_2$ are another set of trainable weights.

During training, you run a forward pass to compute predictions, then compute the loss (how far off you are from the target). Backpropagation calculates how much each entry in $W_1$ and $b_1$ and $W_2$ and $b_2$ contributed to the error. An optimizer (like SGD or Adam) updates these parameters in the direction that reduces the loss. Over many training steps, $W_1$ and $b_1$ and $W_2$ and $b_2$ converge to values that make the model perform well on the training data. Once trained, $W_1$ and $b_1$ and $W_2$ and $b_2$ stay fixed (or “frozen”) to produce predictions for new inputs. They now hold the knowledge the model “learned.” Because the feed-forward sublayer (the 2-layer MLP) needs to learn how to transform each token’s representation in a way that improves the model’s final performance on the training objective, it must also have trainable parameters. 

"Inputs are multiplied by a weight matrix (plus a bias)". So in feed forward, the inputs are multiplied by the weight matrix. Remember the purpose of the feed-forward step is to transform each token’s representation through a nonlinear learned mapping. Self-attention by itself is largely a weighted averaging of different token “value” vectors. Without an additional nonlinear step, the model’s representational power is limited. The feed-forward layers introduce extra transformations that can model richer, more complex relationships.

Those feed-forward weights help the model capture patterns that self-attention alone might not fully exploit, such as certain syntactic or semantic transformations in language.

Nonlinear Transformation

The first linear transformation $hW_1 + b_1$ is followed by a nonlinear activation (e.g., ReLU or GELU). 
- A second linear transformation  $hW_2 + b_2$ brings the dimension back to d. This “expand → nonlinearity → contract” pattern gives the network more representational power.

Because the model expects each token’s output representation to remain the same dimension d (so it can pass to the next Transformer block without any shape mismatch), that second linear transformation “brings the dimension back to d.” This “expand → activate → contract” pattern is a standard way to increase representational power while still returning to the original hidden dimension after the feed-forward sublayer.

Once training finishes, these feed-forward weight values are fixed (unless you’re fine-tuning) and used as is to transform new inputs during inference.

### Training: Forward Pass Review

Below is a high-level overview of the forward pass in a Transformer, including its core components:
- Step 1: Input Embeddings
- Step 2: Multi-Head Self-Attention Mechanism
- Step 3: Residual Connection + Layer Normalization (Post-Attention)
- Step 4: Position-wise Feed-Forward Neural Network
- Step 5: Residual Connection + Layer Normalization (Post-Feed-Forward)
- Step 6: Stacking Multiple Blocks

### Training Backpropagation

**Backpropagation itself is the algorithm for computing gradients with respect to the network parameters. “Computing gradients” means finding how changes in each parameter of the network affect the model’s loss (or error) function. Formally, you are computing the partial derivative of the loss with respect to each parameter in the network. A partial derivative is a way of measuring how a multivariable function changes when you vary just one of its variables, while keeping the other variables constant. Suppose you have a function f(x,y,z) that depends on multiple variables—here x,y, and z. The partial derivative of f with respect to x​, tells you how f changes if you make a tiny change to x but keep y and z fixed. When training a neural network, the loss function is a single number that indicates how far off the model’s predictions are from the true labels (for a given batch of data). However, this function depends on many parameters (weights and biases) inside the model. Each parameter influences the loss in its own way, and we want to know: If I tweak this specific parameter slightly, does the loss go up or down—and by how much? A partial derivative answers exactly that question for each individual parameter, treating all other parameters as fixed. Ultimately, Backpropagation is simply a systematic way of computing how much each individual parameter (weights, biases, etc.) in a neural network contributes to the overall error (the loss). Since modern neural networks often have millions (or even billions) of parameters, we need an efficient procedure to figure out the gradient of the loss with respect to each parameter. This is exactly what backpropagation does.**

**Backpropagation is built on the principle of the Chain Rule in Calculus. When we do a forward pass, we feed an input x through the network layer by layer to produce an output y-hat. Then we compute a loss function L(y-hat, y) comparing y-hat to the true target y. To train the network (i.e., tune its parameters), we want to know how a tiny change in each parameter would affect the final loss. In calculus terms, we need the partial derivative of the loss L with respect to each parameter θ. In neural network training, the parameters (often denoted θ) are the learnable quantities—the weights and biases in each layer. These are the values we tune or optimize during training to reduce the loss.**

**Importantly, we must distinguish the variable θ and x. They are totally different. x represents the input data, which is fixed for any given example (you don’t “update” your training inputs). θ (weights, biases) are the variables you can change, so when we say “compute gradients,” we usually mean:**

$$
\frac{\partial \mathcal{L}}{\partial \theta}
$$
**The calculation above is the partial derivative. $\frac{\partial \mathcal{L}}{\partial \theta}$ tells us how 𝐿 changes if we make a tiny adjustment to parameter $\theta_i$ while keeping every other parameter fixed. This “rate of change” is crucial because it tells the optimizer which direction to move that parameter in order to reduce 𝐿. Gradient descent: After computing the partial derivatives for every parameter (the gradient), the optimizer updates each parameter in the direction that decreases the loss the most. This process is repeated over many iterations, and the network learns the optimal values for all parameters.**

Some additional key concepts of backpropagation: cross-entropy (a type of loss function), a loss function that measures how far off the model’s predictions are from the target.. Stochastic Gradient Descent (SGD) is a type of optimization method. Once backpropagation calculates the gradients, you apply an optimizer (e.g., SGD or Adam) to update the parameters. These optimizers use the gradients to adjust the parameters in a direction that reduces the loss.

As you have seen, the loss function isn't using z-tests, z-scores, confidence interval, significance level, p-values, central limit theorem to know whether to reject or fail to reject the null hypothesis. Cross-entropy does not use classical hypothesis test, such as z-tests, t-tests, and chi-squared tests. Instead, it’s a direct measure of the difference between two probability distributions:
- The model’s predicted distribution over possible outcomes (for instance, which token comes next?), and
- The true distribution (which in a supervised dataset is typically 100% on the correct label/token).

Formally, if the model outputs a probability distribution p for the next token and the ground truth distribution is q (e.g., the correct token with probability 1, or a “one-hot” vector), then the cross-entropy is:

$$
H(q,p) = - \sum_{x} q(x) \log p(x)
$$

No p-values, z-scores, or explicit use of the central limit theorem are involved here. Instead, the model just directly sees how well its predicted probabilities match reality:
- If p(x) is high for the actual label x, the cross-entropy is lower (the model is doing well).
- If p(x) is low for the actual label x, the cross-entropy is higher (the model is doing poorly).

Gradients from the loss are propagated back through all the Transformer layers. Parameters are updated (e.g., via stochastic gradient descent or Adam). This is iterative fitting. Iterative fitting is the process of repeatedly adjusting a model’s parameters so that its predictions get closer and closer to the actual data. Here’s a simple analogy:
- Initial Guess: You start with random guesses for how the model should behave (like randomly chosen parameters).
- Check Error: You see how far off the model’s predictions are from the real answers.
- Adjust: You tweak the parameters to reduce that error slightly.
- Repeat: You go back to Step 2 with the updated model and do it again on more data.
- Over many rounds of this “guess, check, adjust” cycle (like practicing a skill repeatedly), the model’s parameters slowly converge to values that produce better and better predictions. This is fundamentally what happens in gradient descent training for neural networks.

### Training Summary

The memory used in training of course is not applicable for inference. But we still need to load the model learned weights into memory during inference. This includes token embedding, positional embedding, forward pass weights, such as self-attention and feed forward, as well as final output projections. Larger models will require more model weights which increase throughput and latency during inference.

Where does this weight matrix (plus a bias) come from? They’re simply another set of trainable parameters that the model architecture defines as part of its feed-forward sublayer, distinct from the embedding tables. In other words, when you build (or load) a Transformer model, you allocate parameter matrices for:
- Token embeddings
- Positional embeddings
- Query/Key/Value projections for attention
- Feed-forward (MLP) weights
Each of these is declared in the code as a separate parameter or parameter set.

All these model weights get loaded in inference? Yes. When you load a trained Transformer model for inference, all its learned parameters (token embeddings, positional embeddings, attention projections, feed-forward weights, etc.) are loaded into memory. Although no gradient updates happen at inference time (i.e., no backprop), the forward pass still needs every parameter to compute the model’s output.
- Token & Positional Embeddings are needed to map input token IDs and positions to their initial vectors.
- Attention Projections ($W_Q, W_K, W_V$) and the feed-forward sublayers require their respective weight matrices for the forward pass calculations.
- Final Output Layers (like a projection to vocabulary logits in a language model) also need their learned weights.

Without these parameters, the model can’t perform the same transformations it learned during training. So, all of them must be loaded (though typically frozen) at inference.

### vLLM max model model length

In vLLM, the parameter --max-model-len (often referred to as “maximum model length” or “maximum context length”) tells the system how many tokens the model can handle in a single forward pass. Put differently, it is the maximum sequence length that vLLM will allow when performing inference.

When you set --max-model-len, you are explicitly telling vLLM “treat the model’s maximum allowable sequence length as this number.” Internally, vLLM will not process input sequences longer than this value. If you have an input that exceeds --max-model-len tokens, vLLM will typically truncate it or refuse to generate beyond that limit (depending on its exact implementation or the flags you provide).

If you do not set --max-model-len, vLLM will attempt to look at the model’s configuration (the same config.json you see in Hugging Face model repositories) to determine the correct context limit. Concretely, it checks fields like max_position_embeddings or a comparable attribute in the model config. So if your model is configured for 131072 tokens of positional embeddings, then vLLM will set the maximum context length to 131072 by default.

Why it matters:
- Memory usage: A larger maximum context length consumes more GPU (or CPU) memory because self-attention scales with the sequence length. This is another benefit of restricting the max model length. Imagine multiple consumers are using it, and one of those platforms is using a high max model length, then the other platform (like guidewell chat) will suffer as all that memory is being loaded for that other platform and it is using so much gpu.
- Performance: The runtime for attention scales roughly with the square of the sequence length. A higher --max-model-len can make inference slower if you actually push the model to handle very long contexts.
- Compatibility: Some model weights (especially the positional embedding matrices) are specifically trained for a fixed length. If you push --max-model-len beyond what the model was trained for, you may encounter unexpected behavior or degrade model quality unless the model supports (and was trained for) extrapolation.

If your model’s positional embedding matrix has shape [131072, hidden_dim], that typically indicates a maximum sequence length of 131072. In that case, if you do not provide --max-model-len on the command line, vLLM’s default is to look at the model config’s max_position_embeddings (131072) and set the context length to that number. If you pass a smaller number to --max-model-len, vLLM will enforce that smaller limit at inference time (possibly lowering memory usage, but restricting how long an input it will process).

### vLLM max num batch tokens

--max-model-len and --max-num-batched-tokens control two different things. --max-model-len is a per-input-sequence cap on how many tokens one input sequence can include. For example, if --max-model-len=16384, then any single input sequence  cannot exceed 16,384 tokens.

--max-num-batched-tokens A per-batch cap on how many tokens in total can be grouped in one forward-pass batch. For instance, if --max-num-batched-tokens=65536, and you have several shorter requests pending (like four 16k-token requests in the queue), vLLM can batch them all together in a single forward pass (16k × 4 = 64k). But if you tried to batch five such requests (which would total 80k tokens), it would exceed --max-num-batched-tokens, so vLLM would split them into multiple forward passes.

In short, --max-model-len ensures that no single sequence is too long, while --max-num-batched-tokens keeps the sum of tokens (across all requests in a batch) from getting too large. Both are memory/performance safeguards, just at different levels (individual request vs. group of requests).

**So max-num-batched-tokens must always be less than or equal to max-model-len? No, there’s no strict requirement that --max-num-batched-tokens be less than (or greater than) --max-model-len. They each put limits on different things**:
- --max-model-len (per-sequence limit):
    - Caps how many tokens a single sequence (one prompt/request) can contain.
    - Example: If --max-model-len=16384, no individual sequence can exceed 16,384 tokens.
- --max-num-batched-tokens (per-batch limit):
    - Caps the sum of tokens across all sequences batched together in one forward pass.
    - Example: If --max-num-batched-tokens=65536, then a batch can have up to 65,536 tokens in total. This might be 4 sequences of 16,384 tokens each, or 16 sequences of ~4,000 tokens each, etc.

But how does vLLM define a batch? multiple inputs passed into a single http request to its fastapi app or could be one input per request where many requests are passed to its fastapi app concurrently?

vLLM internally defines a “batch” as any group of token sequences that it combines into a single forward pass for efficiency. This batching can happen in multiple ways:
- Multiple input sequences within one HTTP request
    - If an API call itself includes multiple prompts (for example, if you programmatically send a list of input sequences in one request), vLLM can group those into a single batch in one forward pass.
- Multiple concurrent requests
    - In a typical server scenario, multiple users (or threads) might be sending separate HTTP requests—each with one prompt. vLLM can internally queue and merge those individual prompts into a batch.
    - The library “waits” for enough concurrent requests to come in, then combines them (subject to the constraints of --max-num-batched-tokens), and runs one forward pass for all of them at once.

A “batch” is not necessarily tied to a single HTTP request. It can represent:
    - multiple prompts in one API call, or
    - multiple separate API calls (requests) that vLLM merges internally.

The point of batching is to use the hardware more efficiently by running one larger forward pass, rather than many small forward passes. The total number of tokens in the batch (summed across all sequences in that batch) is capped by --max-num-batched-tokens. And regardless of batching, each individual sequence is still capped by --max-model-len.

We are embedding 2000 tokens per batch, so there is no need to have a max model length above 2024. 

vllm serve --port 8070 --host 0.0.0.0 --trust-remote-code --tensor-parallel-size 1 --max-model-len 2024 --max-num-batched-tokens 8192 --task embed TIGER-Lab/VLM2Vec-Full
Ingestion took 146.54 seconds

vllm serve --port 8070 --host 0.0.0.0 --trust-remote-code --tensor-parallel-size 1 --max-model-len 2024 --max-num-batched-tokens 16192 --task embed TIGER-Lab/VLM2Vec-Full
Ingestion took 148.01 seconds

vllm serve --port 8070 --host 0.0.0.0 --trust-remote-code --tensor-parallel-size 1 --max-model-len 2024 --max-num-batched-tokens 24288 --task embed TIGER-Lab/VLM2Vec-Full
Ingestion took 145.25 seconds

### Dynamic Batching in vLLM’s Inference Engine

Yes – vLLM does merge concurrent requests into one batch. The core engine is designed for high-throughput serving by dynamically batching multiple requests together per model forward pass​
MEDIUM.COM
​
MEDIUM.COM
. In each iteration (step) of the engine loop, vLLM’s scheduler can pick up multiple incoming prompts (if available) and process them in a single batch, subject to configured limits. This means that if several HTTP requests arrive around the same time, their prompts can be encoded and fed through the model together in one forward pass, rather than one-by-one.
Components for Queuing and Batching Requests
Several modules work together to queue incoming requests, schedule them, and form batches:
AsyncLLMEngine & RequestTracker: The FastAPI server creates an AsyncLLMEngine to handle requests asynchronously​
MEDIUM.COM
​
GITHUB.COM
. Incoming HTTP requests (via the /generate endpoint) are passed to engine.generate(...), which uses a RequestTracker to queue the request internally​
GITHUB.COM
. The AsyncLLMEngine runs a background loop that waits for new requests and kicks the engine when requests are queued​
MEDIUM.COM
​
MEDIUM.COM
. It pulls new requests from the tracker and adds them to the core engine’s queue by calling LLMEngine.add_request(...) (or its async variant)​
MEDIUM.COM
​
MEDIUM.COM
.
LLMEngine & Scheduler: The LLMEngine is the core inference engine that owns the model, tokenizer, and scheduler. When a new request is added, the engine encodes the prompt into token IDs and wraps it in a SequenceGroup (which represents one request’s sequences)​
MEDIUM.COM
. It then hands this SequenceGroup to the Scheduler, which maintains internal queues. The scheduler has a waiting queue for new or preempted SequenceGroups and a running queue for those already generating tokens​
GITHUB.COM
. Each engine iteration, LLMEngine.step() calls scheduler.schedule() to decide which requests to execute next​
MEDIUM.COM
. This scheduler is responsible for batching logic: it may dequeue multiple SequenceGroups from the waiting queue to process together (a prefill step) or advance all active ones by one token (a decode step), depending on the state.
Batch Execution: Based on the scheduler’s decision, the engine invokes the model forward pass on the selected batch. In code, after scheduler.schedule() returns the selected SequenceGroups (and any memory swap actions), the engine calls model_executor.execute_model(...) with that batch​
MEDIUM.COM
. All prompts or next-token computations in the batch are run in parallel on the GPU. The engine then collects the results and updates each request’s state. The use of sequence groups and the scheduler abstracts batching so that multiple requests can be served in one go. For example, if three requests are in the waiting queue, the scheduler might schedule all three SequenceGroups in one iteration (if within limits), causing one big forward pass for all their prompt tokens rather than three separate passes.
How Batching Decisions Are Made (Token and Sequence Limits)
The vLLM scheduler decides how many concurrent requests to batch based on configurable limits, notably max_num_batched_tokens and max_num_seqs. These define the scheduling budget per iteration. In each scheduler cycle, it will include as many waiting requests as possible until adding another would exceed either the total token budget or the max number of sequences​
GITHUB.COM
. The SchedulingBudget check ensures that the sum of new prompt tokens (or generated tokens) in the batch does not exceed token_budget (derived from --max-num-batched-tokens) and that the number of sequences does not exceed max_num_seqs​
GITHUB.COM
. --max-num-batched-tokens is a key limit: it caps the total tokens processed in one forward pass. For instance, if the limit is 25,000 tokens, the scheduler will accumulate prompts up to that amount. Suppose five new requests arrive, each with a 10k-token prompt. The scheduler will batch the first two requests (20k total tokens) in the same prefill step and leave the others waiting, since adding a third (30k total) would break the 25k token budget​
MEDIUM.COM
. The batching decision is strictly first-come-first-served (FCFS) – the scheduler does not skip over a large request to take later smaller ones. In the previous example, even if requests 4 and 5 were small, the engine stops at request 3 (10k tokens) when it no longer fits, rather than bypassing it​
MEDIUM.COM
. This FCFS policy means the order of requests in the queue is preserved in batching: vLLM will not reorder requests to optimize throughput, it will simply cut off when a limit is hit. In summary, a batch can contain multiple prompts, constrained by the sum of their token lengths (max_num_batched_tokens) and an upper bound on number of parallel sequences (max_num_seqs, e.g. 256 by default).
FastAPI Server Hand-off to the vLLM Engine
The FastAPI HTTP server integrated with vLLM acts as a thin wrapper that passes requests to the engine. In vLLM’s provided API server, the /generate endpoint reads the JSON request, pulls out the prompt and parameters, and then calls engine.generate(prompt, sampling_params, request_id) on the AsyncLLMEngine​
GITHUB.COM
. The AsyncLLMEngine.generate method (an async generator) immediately enqueues the request (with a new UUID) into the engine’s queue and returns an iterator over the result stream​
GITHUB.COM
. Internally, as noted above, this triggers the background engine loop to pick up the request and add it to the scheduler’s waiting queue on the next iteration​
GITHUB.COM
. The FastAPI handler then simply iterates over results_generator to stream out tokens or collects the final output. In essence, the hand-off is accomplished by calling the engine’s async API – no heavy lifting is done in the FastAPI layer beyond JSON parsing. All concurrency and batching is managed by the vLLM engine itself. This design allows multiple HTTP requests to pile up in the engine’s waiting queue, where the scheduler can batch them together. The FastAPI server and engine run in the same process, so the overhead of handing off requests is minimal – the server code obtains the AsyncLLMEngine’s singleton instance and submits work to it directly​
GITHUB.COM
. Each request is identified by an ID so that when outputs are produced by the engine, they can be streamed back on the correct response. The key point is that the FastAPI interface does not call the model directly for each request; instead it funnels all requests into vLLM’s scheduling system, which merges them into batched forward passes as described above. This is how vLLM serves many concurrent prompts efficiently within one model inference loop​
