from .lazy_pdf_ingestor import LazyPdfIngestor
from .lazy_word_ingestor import LazyWordIngestor
from .lazy_powerpoint_ingestor import LazyPowerPointIngestor
from .lazy_text_ingestor import LazyTextIngestor

FACTORIES = {
    'pdf': LazyPdfIngestor,
    'docx': LazyWordIngestor,
    'pptx': LazyPowerPointIngestor,
    'txt': LazyTextIngestor,
}