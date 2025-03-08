from typing import Any, Dict, Optional, Union
from huggingface_hub.inference._common import _as_dict
from huggingface_hub.inference._providers._common import (
    TaskProviderHelper,
    filter_none,
)

class VLLMEmbeddingTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="gwblue-vllm", base_url="", task="embedding")

    def _prepare_mapped_model(self, model: Optional[str]) -> str:
        return model or ""

    def _prepare_route(self, mapped_model: str) -> str:
        return "/v1/embeddings"

    def _prepare_url(self, api_key: Optional[str], model: str) -> str:
        route = self._prepare_route(model)
        return model.rstrip("/") + route

    def _prepare_payload_as_dict(
        self, 
        inputs: Any, 
        parameters: Dict, 
        mapped_model: str
    ) -> Optional[Dict]:
        parameters = filter_none(parameters)

        payload = {
            "model": inputs.get("model", mapped_model),
            "encoding_format": "float",
            "mm_processor_kwargs": {},
        }

        if isinstance(inputs["text"], str):
            payload["input"] = inputs["text"]
            payload.update(parameters)
            return payload

        payload["messages"] = [
            {
                "role": "user",
                "content": []
            }
        ]

        if isinstance(inputs["text"], list) and len(inputs["text"]) > 0:
            message_item = inputs["text"][0]
            if isinstance(message_item, dict):
                if "image_url" in message_item:
                    payload["messages"][0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": message_item["image_url"]}
                    })

                if "text" in message_item:
                    user_text = message_item["text"]
                    if isinstance(user_text, list):
                        text_str = " ".join(user_text)
                    else:
                        text_str = user_text

                    payload["messages"][0]["content"].append({
                        "type": "text",
                        "text": text_str
                    })

        payload.update(parameters)
        return payload
    
    def get_response(self, response: Union[bytes, Dict]) -> Any:
        resp_dict = _as_dict(response)
        return resp_dict["data"][0]["embedding"]