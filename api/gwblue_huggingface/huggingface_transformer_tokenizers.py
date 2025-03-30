import os
import json
import re
import importlib
from pathlib import Path
from typing import TypeVar, Optional, Annotated, Tuple, Dict, Any, List
from typing_extensions import Doc
from abc import ABC
from transformers import PreTrainedTokenizerBase, AutoTokenizer

T = TypeVar("T")
U = TypeVar("U")

"""
Using models with larger model weights will continue to pose challenges and invite novel architectural considerations. Fortunately, transformers don't rely on simple classical statistical tests like chi-squared or anova. They rely on iterative fitting. This concept involves training, using techniques such as cross-entropy loss and stochastic gradient descent (SGD), to iteratively shape predictions to match closely with ground truth. Since our platform is built around inference and not training, we are not concerned with the model weights required for training. Backpropagation and gradient updates (which update the loss function during training to set predictors for token and positional embeddings) are not exercised in inference and those learned token and positional weights are frozen during inference.

But we still need to load model weights into memory during inference. This includes token embedding, positional embedding, forward pass weights, such as self-attention and feed forward, as well as final output projections.

This module wraps features of the transformers package, namely the model tokenizer with additional attributes that specify overriden weights for local deployments, where resources may not match full model training capacity. It also simplifies token management, e.g. `apply_chat_template`.
"""

TRANSFORMER_TOKENIZER_CACHE = "transformers/tokenizers/cache"

def transform_name(s: str) -> str:
    parts = s.split("/", 1)
    remainder = parts[1] if len(parts) > 1 else parts[0]

    result = re.sub(r"[.\-]", "", remainder)
    return result

def get_tokenizer_by_prefix(prefix: str):
    class_name = f"{transform_name(prefix)}PretrainedTokenizer"

    if class_name in globals():
        return globals()[class_name](prefix)
    
    cls = type(
        class_name,
        (BaseLocalTokenizer,),
        {}
    )

    globals()[class_name] = cls
    return cls(prefix)

def get_chat_tokenizer_by_prefix(prefix: str):
    class_name = f"{transform_name(prefix)}ChatPretrainedTokenizer"

    if class_name in globals():
        return globals()[class_name](prefix)
    
    cls = type(
        class_name,
        (BaseChatTokenizer,),
        {}
    )

    globals()[class_name] = cls
    return cls(prefix)

def _load_tokenizer_and_config(name: str) -> Tuple[PreTrainedTokenizerBase, dict]:
    """
    Load a tokenizer + its config from the local cache directory,
    falling back to AutoTokenizer if the explicit class fails.
    Returns (tokenizer, config dict).
    """
    current_file_dir = Path(__file__).resolve().parent
    local_dir = current_file_dir / TRANSFORMER_TOKENIZER_CACHE / name

    tokenizer_config_path = local_dir / "tokenizer_config.json"
    with tokenizer_config_path.open("r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    tokenizer_class_name: str = tokenizer_config.get("tokenizer_class", "")

    tokenizer = None
    try:
        transformers_module = importlib.import_module("transformers")
        cls = getattr(transformers_module, tokenizer_class_name)
        tokenizer = cls.from_pretrained(
            local_dir, local_files_only=True, trust_remote_code=True
        )
    except Exception as e:
        print(
            f"Failed to load tokenizer class '{tokenizer_class_name}' from config. "
            f"Falling back to AutoTokenizer.\nError: {e}"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir, local_files_only=True, trust_remote_code=True
        )

    config_path = local_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return tokenizer, config

def _load_local_config(name: str) -> Dict[str, Any]:
    """
    Model training is done on servers that may not match the servers
    you run on your on-premise setup. Therefore, you may not be able
    to run all the model weights at the full context length. 

    The local_config configuration allows you to override model weights.
    """
    current_file_dir = Path(__file__).resolve().parent
    local_dir = current_file_dir / TRANSFORMER_TOKENIZER_CACHE / name

    local_config_path = local_dir / "local_config.json"
    with local_config_path.open("r", encoding="utf-8") as f:
        local_config = json.load(f)

    return local_config


class BaseLocalTokenizer(ABC):
    def __init__(self, name: str):
        tokenizer, config = _load_tokenizer_and_config(name)
        local_config = _load_local_config(name)

        self.name = name
        self.tokenizer = tokenizer
        self.config = config
        self.local_config = local_config
        self.max_sequence_length = tokenizer.model_max_length

    @property
    def vector_dimension_length(self) -> int:
        """
        Transformer-based model training relies on iterative fitting. With any 
        neural network, iteration occurs in the forward pass of the network layers. 
        In training, the goal is to find model parameters that produce accurate 
        outputs on new, unseen data. Transformers process entire sequences (or 
        batches of sequences) in a single forward pass rather than stepping through 
        tokens one by one (as a traditional Recurrent Neural Network (RNN) would). 

        When you first instantiate the Transformer (or any neural network), you 
        allocate memory for all trainable parameters—including the token and 
        positional embedding tables. These tables are usually randomly initialized 
        or loaded from a pretrained checkpoint (e.g., if you're using a pretrained 
        language model). If you're training a model from scratch, those initial 
        token and positional embedding values are typically randomized and don't 
        carry any semantic meaning. Over the course of training, backpropagation 
        will adjust these vectors so that tokens with similar meanings end up with 
        similar embedding vectors. By the end, the once-random embeddings become 
        rich, learned representations that help the network minimize its loss. 
        However, if you're loading a pretrained model, the token and positional 
        embeddings will already contain meaningful values learned in a previous 
        training phase. In that case, they're not random but come from the 
        pretrained checkpoint.

        Each token id in the embedding tables is a high-dimensional vector. In the 
        case of vlm2vec-full, it is a 3072-dimensional vector. Eventually, these 
        vectors will contain semantic information about the token ids. The positional 
        embedding table is indexed by position (sometimes called position IDs), not 
        by token IDs.
        """
        return self.config.get("hidden_size") or self.config.get("n_embd")

    @property
    def token_embeddings(self) -> Tuple[int, int]:
        """
        The Token Embedding Table maps each token ID from the vocabulary to a continuous 
        vector representation. The token id is often a word or subword from the vocabulary. 
        For example, the token "the" may be mapped to the ID 464. Platforms like vLLM and 
        HuggingFace TGI load a transformer tokenizer in their engine to convert raw text to 
        token ids and back. The trained model will only work with the token ids and not 
        raw text. Hence, when inference platforms like vLLM or Hugging Face TGI run, they 
        download model card configs from hugging face and store them internally. These 
        configs include config.json, special_tokens_map.json, tokenizer_config.json, and 
        tokenizer.json. The tokenizer must convert all text to token ids. When the model 
        sees a token ID such as 464 at runtime, that will correspond to an English word 
        like "the", even though the mode doesn't understand "the".

        The token ids in the token emebdding table do not need to be in any particular 
        order, since it is the embedding vectors that convey meaning and semantic similarity. 
        However, different tokenizers order token ids differently. For example, in a Byte-
        Pair Encoding (BPE) or WordPiece tokenizer, the algorithm might merge subwords by 
        frequency or other heuristics, and assign lower IDs to more frequent tokens. But 
        there is no strict rule that "token 0 must be [PAD]" or "token 464 must be the" 
        across all models—each tokenizer might have a different mapping.

        It is the role of the token embedding table to map all tokens to vectors whose 
        dimensionality matches the model's internal hidden dimension. In the case of vlm2vec
        -full, the hidden dimension is 3072. In effect, all tokens will have an associated 
        3072-dimensional embedding vector. The fact that 3072 is quite large just reflects 
        the architecture's capacity. It allows the model to store richer contextual and 
        semantic information for each token than if the dimension were smaller.

        While each individual dimension isn't interpretable on its own, taken together, the 
        dimensions encode patterns of usage and context—syntactic, semantic, even morphological 
        traits—that help the model distinguish how tokens behave in language. Hence, although 
        the embedding matrix itself is just a big table indexed by token IDs, each row 
        (embedding) is a rich, learned representation that encodes diverse nuance about that 
        token's role in language.

        Each model has its on vocabulary length. The length of the vocabulary is determined by 
        the vocab_size in the transformer's model card config.json file. For example, for 
        vlm2vec-full, vocab_size is 32064. The shape of the matrix of the token embedding 
        table is [32064, 3072] for vlm2-vec full. Index i in this matrix corresponds to the 
        embedding for the i-th token in the vocabulary. The table does not store the literal 
        text “the” (or any other strings). Instead, each row in the table is just a vector of 
        learned numbers (floating-point parameters). Row index i (for example, 464 for “the”) 
        corresponds to a 3072-dimensional vector (like [0.12, -0.03, 0.78, ...]) that the model 
        learns to represent the concept of that token.
        """
        vocab_size = self.config.get('vocab_size', 0)
        return [vocab_size, self.vector_dimension_length]

    @property
    def positional_embeddings(self) -> Tuple[int, int]:
        """
        Transformers are permutation-invariant. They don't capture sequence ordering of inputs 
        by default. At the same time, the token embedding table only provides semantic structure 
        to individual tokens. It does not provide sequence ordering. Token embeddings answer the 
        question "what token is this?" but not "where does this token appear in the sentence?" 
        Sequence ordering captures its own meanings. For example, we can say "I love cats" or 
        "cats love me"? Both share the tokens "I," "love," "cats," but they have fundamentally 
        different meanings because the tokens appear in a different order.

        There are so many ways to arrange words in a sentence that express different meanings. 
        If we were to arrange words all the different ways possible, we will have an infinite 
        number of permutations. Indeed, there are infinitely many possible word permutations—
        but a Transformer (or any language model) does not store every permutation. Instead, 
        it learns generalizable patterns of how word order affects meaning. In fact, the 
        positional emebdding matrix of vlm2vec-full, for example, is [131072, 3072]. Thus, a 
        lot of features are captured to understand what it is like to be in position i in the 
        input sequence.
        """
        positional_size = self.config.get('max_position_embeddings') or self.max_sequence_length
        return [positional_size, self.vector_dimension_length]

    @property
    def attention_heads(self) -> int:
        """
        For each token in your input, the forward pass looks up the embedding vector from 
        the token table and the positional table, sums them, and sends that to the Transformer 
        blocks. The newly-represented embedding vector is what is passed into the first 
        transformer block. All tokens in the input sequence are converted to their vector 
        representations at the same time using the token and positioanl embedding tables and 
        all pass into the first transformer block in parallel.

        Multi-Head Self-Attention is the first processing of input vectors in the Forward Pass 
        of a transformer block. Multi-Head Self-Attention can be broken into two parts: (1) 
        computing “Scaled Dot-Product Attention” for one head, and (2) repeating this process 
        over multiple heads in parallel.

        Each token in your sequence starts with some embedding vector $x_i$. In an encoder or 
        decoder block, these embeddings could be:
        - The original token embedding + positional embedding (in the first layer)
        - The output of the previous layer (in deeper layers).

        For each token i, we produce three vectors:
        $$
        Q_i = x_i W_Q,\quad
        K_i = x_i W_K,\quad
        V_i = x_i W_V.
        $$

        $W_Q, W_K, W_V$ are learned parameter matrices. Intuition:
        - $Q_i$ (Query) represents "what information token i is seeking."
        - $K_j$ (Key) represents "the content token j holds."
        - $V_j$ (Value) is "the information token j will actually contribute if it's attended to."

        For each token i, we compare its query $Q_i$ to all keys in the sequence $K_1, K_2, \ldots, 
        K_n$ via dot products:
        $$
        \text{score}(i, j) = Q_i \cdot K_j
        $$
        These raw similarity scores tell us how relevant token j is to token i.

        When we compare a particular token's query vector $Q_i$ with the key vectors $K_j$ for all 
        tokens j in the sequence, we get a set of raw similarity scores ($Q_i ⋅ K_j$). The softmax 
        function is then applied to these scores to transform them into a set of nonnegative weights 
        that sum to 1. In other words, we are creating probabilties. After applying softmax, each 
        score $\alpha_{i,j}$ becomes a number in the range [0,1]. Sum to 1: $\sum_{j} \alpha_{i,j} 
        = 1$. You can interpret  $\alpha_{i,j}$ like a “probability” that token i attends to token j.**

        $$
        \alpha_{i,j} 
        = \mathrm{Softmax}\!\Bigl(
        \frac{Q_i \cdot K_j}{\sqrt{d_k}}
        \Bigr).
        $$
        - The Softmax ensures that $\sum_{j} \alpha_{i,j} = 1$
        - Each $\alpha_{i,j}$ is called the “attention weight” from token i to token j.

        Finally, we create the new representation for token i, denoted $z_i$, by weighting each 
        value $V_j$ by $\alpha_{i,j}$.
        $$
        z_i = \sum_{j=1}^{n} \alpha_{i,j} \, V_j.
        $$
        This $z_i$ is a contextually updated vector for token i. It blends information from all other 
        tokens based on their relevance to i.

        In practice, Transformers do Scaled Dot-Product Attention several times in parallel “heads,” 
        each with different projection matrices $W_Q^{(h)}, \quad W_K^{(h)}, \quad W_V^{(h)}.$

        Each head h produces its own set of $\{\, z_1^{(h)}, \dots, z_n^{(h)} \}$. The intuition: 
        Different heads can learn to focus on different types of relationships. For example, one head 
        might learn to track verb-object relations, another might focus on synonyms or long-range 
        dependencies, etc.

        After all heads compute their $z_i^{(h)}$ vectors, we concatenate the results for each token i:
        $$
        z_i^{\mathrm{multi}} = \mathrm{Concat}\bigl(z_i^{(1)},\, z_i^{(2)},\, \ldots,\, z_i^{(H)}\bigr).
        $$

        Then we pass this concatenated vector through another linear projection (often denoted $W_O$):
        $$
        z_i^{\mathrm{output}} = z_i^{\mathrm{multi}} \, W_O.
        $$
        That final $z_i^{\mathrm{output}}$ is what leaves the multi-head attention sublayer for token i. 
        It then goes on to the residual connection + layer normalization, and ultimately to the feed-
        forward sublayer in the Transformer block.
        """
        return self.config.get("num_attention_heads")

    @property
    def hidden_layers(self) -> int:
        """
        In the context of Transformers num_hidden_layers refers to the number of Transformer blocks 
        stacked in the model. Each Transformer block typically includes:
        - A (multi-head) self-attention sub-layer,
        - Residual (skip) connections, and
        - Layer normalization
        - A feed-forward sub-layer (sometimes called the MLP block),

        A residual connection (also called a skip connection) simply means we add the original input 
        to the sublayer's output, rather than relying on the sublayer's output alone. In the case of 
        multi-head attention, if:
        - $x_i$ is the input representation for token i (which might be either the original token 
        embedding + positional embedding in the very first layer, or the output of the previous 
        layer in deeper layers),
        - $z_i^{(\mathrm{multi\text{-}head})}$ is the result of the multi-head attention sublayer for 
        token i, then the residual connection is:
        $$
        \text{AttentionOutput}_i = z_i^{(\mathrm{multi\text{-}head})} + x_i.
        $$

        In addition to the Residual Connection, we have Layer Normalization: Normalize the result across 
        the hidden dimension to stabilize training and keep values in a consistent range:
        $$
        \text{NormedOutput} = \mathrm{LayerNorm}\bigl(\text{AttentionOutput}\bigr).
        $$

        In the attention sublayer, we do a lot of “mixing” across tokens (each token attends to every 
        other token). In contrast, the feed-forward sublayer is position-wise—meaning it applies the 
        same MLP to each token separately. “MLP” stands for Multi-Layer Perceptron, which is essentially 
        a feed-forward neural network composed of one or more fully connected (dense) layers. In an MLP:
        - Inputs are multiplied by a weight matrix (plus a bias) and then passed through a nonlinear 
        activation function (e.g., ReLU).
        - Multiple layers can be stacked, where each layer's output is the next layer's input.
        - No recurrence or convolution is used, so it's called "feed-forward"—information flows strictly 
        from the input layer to the output layer.
        """
        return self.config.get("num_hidden_layers")

    @property
    def key_value_heads(self) -> int:
        """
        Key Value Heads indicates how many attention heads are specifically used to 
        process keys and values (K/V) in the attention mechanism.
        """
        return self.config.get("num_key_value_heads")


    @property
    def sequence_length_forward_pass(self) -> int:
        """
        sequence length refers to the maximum number of tokens (or positions) 
        a model can handle in one forward pass of the neural network, where
        the input (sequence of tokens) flows through the input layer, then all
        the hidden layers, and finally the final output layer. The output being
        next-token preditions.
        """
        return self.local_config.get("max_model_len") or self.positional_embeddings[0]

    @property
    def max_batch_tokens_forward_pass(self) -> int:
        """
        max batch tokens refers to a per-batch cap on how many tokens in total can 
        be grouped in one forward-pass batch.

        For instance, if max-num-batched-tokens is 65536, and you have several shorter 
        requests pending (like four 16k-token requests in the queue), vLLM, for example,
        can batch them all together in a single forward pass (16k x 4 = 64k). But if you 
        tried to batch five such requests (which would total 80k tokens), it would exceed 
        max-num-batched-tokens, so vLLM, for example, would split them into multiple 
        forward passes.

        Note this same behavior will work on Hugging Face TGI and TEI and all supported
        platforms of the gwblue modules.
        """
        return self.local_config.get("max_batch_tokens") or self.sequence_length_forward_pass

    @property
    def max_new_tokens(self) -> Optional[int]:
        return self.local_config.get("max_new_tokens", None)

    def __repr__(self) -> str:
        f"""
        {get_tokenizer_by_prefix(self.name)}(
            name={self.name}
            vector_dimension_length={self.vector_dimension_length}
            token_embeddings={self.token_embeddings}
            positional_embeddings={self.positional_embeddings}
            attention_heads={self.attention_heads}
            hidden_layers={self.hidden_layers}
            key_value_heads={self.key_value_heads}
            sequence_length_forward_pass={self.sequence_length_forward_pass}
            max_batch_tokens_forward_pass={self.max_batch_tokens_forward_pass}
            max_new_tokens={self.max_new_tokens}
        )
        """

    def __str__(self) -> str:
        return "[{self.sequence_length_forward_pass}, {self.max_batch_tokens_forward_pass}]"
    
    def __len__(self) -> int:
        return self.sequence_length_forward_pass

class BaseChatTokenizer(BaseLocalTokenizer, ABC):
    def to_chat_template_ids(self, messages: List[Dict[str, Any]]) -> List[int]:
        """
        The "chat_template" syntax is specific to language models that support a 
        predefined chat or conversation format, known as an Instruct. For example,
        Llama 2/3 Chat variants are known as instruction-tuned models that explicitly
        include a chat template definitions in their tokenizer config. Hence, you
        have a model name like "Llama 3.1 70B Instruct".

        There is an important distinction to draw. Chat Templates are not baked into 
        the language model's internal weights, whether it is the token or positional
        embeddings tables, the weights of the forward pass, or other model weights.
        Those internal model weights are completely separate from the chat template
        instructions template. However, the model's weights (including token embeddings) 
        expect or are attuned to the special tokens and ordering used by its chat 
        template (remember in the end, the chat template will get broken down into tokens 
        when pass into the neural network layers). However, that does not mean the 
        template itself lives in the weights. Rather:
        - During training, the model sees text data that follows a particular format:
            - special tokens for roles (<|start_header_id|>user<|end_header_id|>) 
            - begin-of-sequence(BOS)/end-of-sequence (end of turn) (EOS) markers 
                - <|begin_of_text|> and <|eot_id|>
                - <s> and </s>
        - The embedding matrix ends up with learned representations for those special 
        tokens; the self-attention layers learn to recognize patterns of usage for them.
        - At inference time, to get the best behavior, you must replicate the same token 
        structure (with those special tokens and message order). That's exactly what the 
        "chat template" handles externally.

        In effect, the "chat_template" is not baked into the model's internal weights 
        or forward pass; rather, it's typically stored as metadata in the model's files
        —most often in tokenizer_config.json or a separate Python helper (like 
        apply_chat_template).

        The "chat template" handles the role-based formatting (system, user, assistant), 
        multi-turn conversation state, and tokenization for you. In other 
        words, instead of calling a generic "text-generation" pipeline and manually 
        stitching messages together, a chat template can:
        - Accept messages in a structured format (e.g., system and user messages).
        - Automate the insertion of special "chat" tokens (like role tags, BOS/EOS markers).
        - Keep track of conversation history across multiple turns so the model maintains 
        context.

        Note a base LLM (not fine-tuned for chat) typically has no dedicated "system"/"user"
        /"assistant" roles, so it will not include any "chat_template" in its config.

        Verify support of chat_template:
        - Check the Model's tokenizer_config.json
            - If the model's maintainers have included a chat template, you'll see an entry 
            like "chat_template": ... describing how roles and messages get wrapped in 
            special tokens.
        - Look for "chat" or "instruct" in the Model's Card
            - Models named "-chat-", "-instruct-", or "-dialog-" often come with an 
            official conversation format.
        - Use the Official Example
            - If the model card or README shows code like tokenizer.apply_chat_template(...), 
            that's a good sign it has built-in chat functionality.

        Utilization of "chat_template" ensures token counting reflects the true prompt length 
        used by the model. If you simply concatenate the raw message contents without the 
        template (or use a wrong template), the token count will not match the actual model 
        input length.
        """
        token_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True
        )
        return token_ids
    
    def multimodal_template(self, content: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": content},
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]