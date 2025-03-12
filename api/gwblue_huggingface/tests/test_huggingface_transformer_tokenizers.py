from ..huggingface_transformer_tokenizers import (
    BgeLargePretrainedTokenizer,
    NomicPretrainedTokenizer,
    MetaLlama3170BInstructPretrainedTokenizer,
    Llama3290BVisionInstructPretrainedTokenizer,
)

def test_bge_large_pretrained_tokenizer():
    bge = BgeLargePretrainedTokenizer()
    assert bge.sequence_length > 0

def test_nomic_pretrained_tokenizer():
    nomic = NomicPretrainedTokenizer()
    assert nomic.sequence_length > 0

def test_llama_70b_instruct_pretrained_tokenizer():
    llama_70b = MetaLlama3170BInstructPretrainedTokenizer()
    assert llama_70b.sequence_length > 0

def test_llama_90b_vision_instruct_pretrained_tokenizer():
    llama_90b = Llama3290BVisionInstructPretrainedTokenizer()
    assert llama_90b.sequence_length > 0
