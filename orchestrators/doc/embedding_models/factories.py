from orchestrators.doc.embedding_models.hf_tei import HFTEI
from orchestrators.doc.embedding_models.hf_hub import HFHub

FACTORIES = {
    'tei': HFTEI,
    'hub': HFHub,
}