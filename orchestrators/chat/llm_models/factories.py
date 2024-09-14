from orchestrators.chat.llm_models.hf_tgi_llm import HFTGI
from orchestrators.chat.llm_models.hf_pipeline_llm import HFPipelineLLM

FACTORIES = {
    'tgi': HFTGI,
    'pipeline': HFPipelineLLM,
}