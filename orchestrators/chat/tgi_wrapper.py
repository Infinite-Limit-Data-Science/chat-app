import requests
from orchestrators.chat.llm_wrapper import LLMWrapper 

class TGIWrapper(LLMWrapper):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def generate(self, template: str, values: dict, **kwargs):
        prompt = template.format(**values)
        response = requests.post(self.endpoint_url, json={"prompt": prompt, **kwargs})
        response.raise_for_status()
        return response.json()["generated_text"]
    
    def generate_stream():
        pass