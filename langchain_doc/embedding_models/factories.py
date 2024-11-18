from .hf_tei import HFTEI
from .hf_hub import HFHub

FACTORIES = {
    'tei': HFTEI,
    'hub': HFHub,
}