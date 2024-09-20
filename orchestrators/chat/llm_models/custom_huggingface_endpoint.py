import logging
import json
from typing import Optional, List, Any, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_huggingface import HuggingFaceEndpoint as BaseHuggingFaceEndpoint, HuggingFacePipeline


from langchain_huggingface.chat_models import huggingface

class CustomHuggingFaceEndpoint(BaseHuggingFaceEndpoint):
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling text generation inference API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "return_full_text": False,
            "truncate": self.truncate,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
            "do_sample": self.do_sample,
            "watermark": self.watermark,
            **self.model_kwargs,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to HuggingFace Hub's inference endpoint."""
        logging.warning(f'WHAT IS CURRENT PROMPT {prompt}')
        logging.warning(f'WHAT IS THE CURRENT STOP {stop}')

        invocation_params = self._invocation_params(stop, **kwargs)
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **invocation_params):
                completion += chunk.text
            return completion
        else:
            invocation_params["stop"] = invocation_params[
                "stop_sequences"
            ]  # porting 'stop_sequences' into the 'stop' argument
            response = self.client.post(
                json={"inputs": prompt, "parameters": invocation_params},
                stream=False,
                task=self.task,
            )
            response_text = json.loads(response.decode())[0]["generated_text"]

            # Maybe the generation has stopped at one of the stop sequences:
            # then we remove this stop sequence from the end of the generated text
            for stop_seq in invocation_params["stop_sequences"]:
                if response_text[-len(stop_seq) :] == stop_seq:
                    response_text = response_text[: -len(stop_seq)]
            return response_text

    async def _acall(
        self,
        prompt: List[dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        invocation_params = self._invocation_params(stop, **kwargs)
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt, stop, run_manager, **invocation_params
            ):
                completion += chunk.text
            return completion
        else:
            invocation_params["stop"] = invocation_params["stop_sequences"]
            response = await self.async_client.post(
                json={"inputs": prompt, "parameters": invocation_params},
                stream=False,
                task=self.task,
            )
            response_text = json.loads(response.decode())[0]["generated_text"]

            # Maybe the generation has stopped at one of the stop sequences:
            # then remove this stop sequence from the end of the generated text
            for stop_seq in invocation_params["stop_sequences"]:
                if response_text[-len(stop_seq) :] == stop_seq:
                    response_text = response_text[: -len(stop_seq)]
            return response_text