from ..huggingface_transformer_tokenizers import (
    BgeLargePretrainedTokenizer,
    NomicPretrainedTokenizer,
    MetaLlama3170BInstructPretrainedTokenizer,
    Llama90BVisionInstructPretrainedTokenizer
)

def test_bge_large_pretrained_tokenizer():
    bge = BgeLargePretrainedTokenizer()
    assert 'tokenizer' in bge.__dict__

def test_nomic_pretrained_tokenizer():
    nomic = NomicPretrainedTokenizer()
    assert 'tokenizer' in nomic.__dict__

def test_llama_70b_instruct_pretrained_tokenizer():
    llama_70b = MetaLlama3170BInstructPretrainedTokenizer()
    assert 'tokenizer' in llama_70b.__dict__

def test_llama_90b_vision_instruct_pretrained_tokenizer():
    llama_90b = Llama90BVisionInstructPretrainedTokenizer()
    assert 'tokenizer' in llama_90b.__dict__