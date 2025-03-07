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
        
    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        parameters = filter_none(parameters)

        payload = {}
        if isinstance(inputs, dict):
            payload.update(inputs)
        if "model" not in payload:
            payload["model"] = mapped_model

        for k, v in parameters.items():
            payload[k] = v

        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        resp_dict = _as_dict(response)
        return resp_dict["data"][0]["embedding"]