from ..huggingface_transformer_tokenizers import get_tokenizer_class_by_prefix

def test_meta_llama_3_2_11B_vision_instruct_tokenizer():
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    LocalTokenizer = get_tokenizer_class_by_prefix(model_name)
    local_tokenizer = LocalTokenizer(model_name)
    assert local_tokenizer.sequence_length_forward_pass == 38334
    assert local_tokenizer.max_batch_tokens_forward_pass == 38334

def test_meta_llama_guard_3_8B_tokenizer():
    model_name = "meta-llama/Llama-Guard-3-8B"

    LocalTokenizer = get_tokenizer_class_by_prefix(model_name)
    local_tokenizer = LocalTokenizer(model_name)
    assert local_tokenizer.sequence_length_forward_pass == 38334
    assert local_tokenizer.max_batch_tokens_forward_pass == 38334

def test_tiger_lab_vlm2vec_full_tokenizer():
    model_name = "TIGER-Lab/VLM2Vec-Full"

    LocalTokenizer = get_tokenizer_class_by_prefix(model_name)
    local_tokenizer = LocalTokenizer(model_name)
    assert local_tokenizer.sequence_length_forward_pass == 2024
    assert local_tokenizer.max_batch_tokens_forward_pass == 8096

def test_baai_bge_large_en_1_5_tokenizer():
    model_name = "BAAI/bge-large-en-v1.5"

    LocalTokenizer = get_tokenizer_class_by_prefix(model_name)
    local_tokenizer = LocalTokenizer(model_name)
    assert local_tokenizer.sequence_length_forward_pass == 512
    assert local_tokenizer.max_batch_tokens_forward_pass == 32768