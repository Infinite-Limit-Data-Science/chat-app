
from orchestrators.chat.llm_decorator import hf_endpoint

# chat orchestrator must be used independently of api schemas
# therefore cannot pass in the api schemas like ModelConfigEndpointSchema
class HFTGI:
    schema: str = __model_config_endpoint_schema__

    @hf_endpoint
    def llm(self, data):
        return data