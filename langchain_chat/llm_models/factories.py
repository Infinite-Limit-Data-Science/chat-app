from .hf_tgi_llm import HFTGI
from .hf_pipeline_llm import HFPipelineLLM

FACTORIES = {
    'tgi': HFTGI,
    'pipeline': HFPipelineLLM,
}