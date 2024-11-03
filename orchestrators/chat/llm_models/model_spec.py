from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_ARCHITECTURES = [('A10G', 'Ampere'), ('A100', 'Ampere'), ('H100', 'Hopper')]

class BytePrecision(Enum):
    BYTES_2 = 2
    BYTES_4 = 4

class ModelSpec(BaseModel):
    available_mem: float = Field(description='Post-initialization GPU memory')
    hidden_layers: int = Field(alias='num_hidden_layers', description='Number of transformer layers, e.g. self-attention and feed-forward (Do not conflate with supervised learning hidden layer in forward prop Neural Network like word2vec)', frozen=True)
    activation_dims: int = Field(alias='hidden_size', description='Dimensions of activation', frozen=True)
    max_sequence_len: int = Field(alias='max_position_embeddings', description='Maximum sequence length', frozen=True)
    precision: int = Field(alias='torch_dtype', description='Memory requirements per element', frozen=True)
    vocab_size: int = Field(description='Specifies the size of the model\'s vocabulary, i.e., the total number of unique tokens that the model can recognize and generate.', frozen=True)
    prefill_operation: Optional[float] = Field(default=None, description='Computed prefill operation value')
    num_heads: int = Field(alias='num_attention_heads', description='Number of attention heads')

    @field_validator(mode='before')
    @classmethod
    def set_precision(cls, v: str) -> int:
        return {
            'bfloat16': BytePrecision.BYTES_2.value,
            'float32': BytePrecision.BYTES_4.value,
        }.get(v, None)

    @model_validator(mode='before')
    @classmethod
    def check_positive_rationals(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        fields_to_check = ['post_memory', 'hidden_layers', 'activation_dims', 'max_sequence_len']
        for field in fields_to_check:
            if values[field] <= 0:
                raise ValueError(f'{field} must be a positive rational number')
        
        return values
    
    @staticmethod
    def self_attention_memory(seq_len, activation_dim, num_heads, precision_bytes) -> int:
        """
        Estimated memory in bytes for the self-attention layer for a single forward pass
        Formula:
        $$
        \text{QKV\_memory} = 3 \times (k \times d \times b)
        $$

        $$
        \text{Attention\_scores\_memory} = h \times (k \times k \times b)
        $$

        $$
        \text{Attention\_output\_memory} = k \times d \times b
        $$

        $$
        \text{Total\_self\_attention\_memory} = \text{QKV\_memory} + \text{Attention\_scores\_memory} + \text{Attention\_output\_memory}
        $$

        $$
        \text{Total\_self\_attention\_memory} = 3 \cdot (k \cdot d \cdot b) + h \cdot (k \cdot k \cdot b) + (k \cdot d \cdot b)
        $$

        $$
        \text{Total\_self\_attention\_memory} = (4 \cdot k \cdot d + h \cdot k^2) \cdot b
        $$
        """
        qkv_memory = 3 * (seq_len * activation_dim * precision_bytes)
        
        attention_scores_memory = num_heads * (seq_len * seq_len * precision_bytes)
        
        attention_output_memory = seq_len * activation_dim * precision_bytes
        
        total_memory = qkv_memory + attention_scores_memory + attention_output_memory
        
        return total_memory

    @staticmethod
    def feed_forward_memory(seq_len, activation_dim, precision_bytes, ffn_multiplier=4) -> int:
        """
        Estimated memory in bytes for the feed-forward layer in a transformer model during a single forward pass
        Formula:
        $$
        d_{\text{ffn}} = \text{activation\_dim} \times \text{ffn\_multiplier}
        $$

        $$
        \text{FFN\_layer1\_memory} = \text{seq\_len} \times d_{\text{ffn}} \times \text{precision\_bytes}
        $$

        $$
        \text{FFN\_activation\_memory} = \text{seq\_len} \times d_{\text{ffn}} \times \text{precision\_bytes}
        $$

        $$
        \text{FFN\_layer2\_memory} = \text{seq\_len} \times \text{activation\_dim} \times \text{precision\_bytes}
        $$

        $$
        \text{Total\_FFN\_memory} = \text{FFN\_layer1\_memory} + \text{FFN\_activation\_memory} + \text{FFN\_layer2\_memory}
        $$

        $$
        \text{Total\_FFN\_memory} = (\text{seq\_len} \times d_{\text{ffn}} \times \text{precision\_bytes}) + (\text{seq\_len} \times d_{\text{ffn}} \times \text{precision\_bytes}) + (\text{seq\_len} \times \text{activation\_dim} \times \text{precision\_bytes})
        $$

        $$
        \text{Total\_FFN\_memory} = (2 \times \text{seq\_len} \times d_{\text{ffn}} + \text{seq\_len} \times \text{activation\_dim}) \times \text{precision\_bytes}
        $$
        """
        d_ffn = activation_dim * ffn_multiplier
        
        ffn_layer1_memory = seq_len * d_ffn * precision_bytes
        
        ffn_activation_memory = seq_len * d_ffn * precision_bytes
        
        ffn_layer2_memory = seq_len * activation_dim * precision_bytes
        
        total_ffn_memory = ffn_layer1_memory + ffn_activation_memory + ffn_layer2_memory
        
        return total_ffn_memory
    
    def calculate_max_batch_total_tokens(self) -> int:
        """
        Formula:
        $$
        M = \frac{i \, \text{GB}}{\ell \cdot d \cdot k \cdot n \, \text{bytes}}
        $$
        """
        available_mem = self.available_mem
        hidden_layers = self.hidden_layers
        activation_dims = self.activation_dims
        max_sequence_len = self.max_sequence_len
        precision = self.precision
        num_heads = self.num_heads

        available_mem_bytes = available_mem * (1024 ** 3)

        estimated_self_attention_memory = self.self_attention_memory(max_sequence_len, activation_dims, num_heads, precision)
        estimated_feed_forward_memory = self.feed_forward_memory(max_sequence_len, activation_dims, precision)

        total_layer_memory = estimated_self_attention_memory + estimated_feed_forward_memory

        forward_prop_activation_bytes = hidden_layers * total_layer_memory

        M = available_mem_bytes / forward_prop_activation_bytes
        prefill_operation = M * 100

        return prefill_operation