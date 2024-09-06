from langchain_huggingface import HuggingFaceEndpoint

def hf_endpoint(orig_func):
    def wrapper(self, data):
        endpoint_url = data['endpoints'][0]['url']
        task = "text-generation"
        max_new_tokens = data['parameters']['max_new_tokens']
        return orig_func(self, HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            task=task,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        ))
    return wrapper