import json
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
        self, inputs: Any, parameters: Dict, mapped_model: str
    ) -> Optional[Dict]:
        parameters = filter_none(parameters)

        payload = {
            "model": inputs.get("model", mapped_model),
            "encoding_format": "float",
            "mm_processor_kwargs": {},
        }

        text_val = inputs.get("text")

        if isinstance(text_val, list) and all(isinstance(t, str) for t in text_val):
            payload["input"] = text_val
            payload.update(parameters)
            return payload

        if isinstance(text_val, str):
            payload["input"] = [text_val]
            payload.update(parameters)
            return payload

        if (
            isinstance(text_val, list)
            and len(text_val) > 0
            and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in text_val)
        ):
            payload["messages"] = text_val
            payload.update(parameters)
            return payload

        payload["messages"] = [{"role": "user", "content": []}]

        content_list = text_val if isinstance(text_val, list) else [text_val]

        for item in content_list:
            if isinstance(item, dict):
                if "image_url" in item:
                    payload["messages"][0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]},
                    })
                if "text" in item:
                    payload["messages"][0]["content"].append({
                        "type": "text",
                        "text": item["text"],
                    })
            elif isinstance(item, str):
                payload["messages"][0]["content"].append({
                    "type": "text",
                    "text": item,
                })

        payload.update(parameters)
        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        resp_dict = _as_dict(response)
        all_embeddings = []
        for item in resp_dict["data"]:
            all_embeddings.append(item["embedding"])
        return all_embeddings
