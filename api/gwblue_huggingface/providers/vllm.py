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

        if isinstance(inputs.get("text"), list) and all(
            isinstance(t, str) for t in inputs["text"]
        ):
            payload["input"] = inputs["text"]
            payload.update(parameters)
            return payload

        payload["messages"] = [{"role": "user", "content": []}]

        content_list = (
            inputs["text"] if isinstance(inputs["text"], list) else [inputs["text"]]
        )

        for item in content_list:
            if isinstance(item, dict):
                if "image_url" in item:
                    payload["messages"][0]["content"].append(
                        {"type": "image_url", "image_url": {"url": item["image_url"]}}
                    )
                if "text" in item:
                    payload["messages"][0]["content"].append(
                        {"type": "text", "text": item["text"]}
                    )
            elif isinstance(item, str):

                payload["messages"][0]["content"].append({"type": "text", "text": item})

        payload.update(parameters)
        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        resp_dict = _as_dict(response)
        all_embeddings = []
        for item in resp_dict["data"]:
            all_embeddings.append(item["embedding"])
        return all_embeddings
