from .base_loader import BaseLoader
from .pdf_loader import PyPDFImageLoader
from .power_point_loader import PowerPointLoader
from .word_loader import WordLoader

__all__ = [
    'BaseLoader', 
    'PyPDFImageLoader', 
    'PowerPointLoader', 
    'WordLoader',
]