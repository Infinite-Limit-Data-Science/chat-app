The UI must have a configuration similar to the following:

MODELS='[
    {
        "name":"meta-llama/Meta-Llama-3.1-8B-Instruct", 
        "endpoints":[{"url": "http://0.0.0.0:8080/generate_stream"}, "type": "tgi"],
        "description": "The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.",
        "promptExamples": {"title": "Code a snake game", "prompt": "Code a basic snake game in python. Give explanations for each step."},
        "parameters": {"stop": ["<|eot_id|>","<|im_end|>"], "truncate": "", "max_new_tokens": 1024}

    }
]'