from orchestrators.doc.ingestors.lazy_pdf_ingestor import LazyPdfIngestor
from orchestrators.doc.ingestors.lazy_word_ingestor import LazyWordIngestor
from orchestrators.doc.ingestors.lazy_powerpoint_ingestor import LazyPowerPointIngestor
from orchestrators.doc.ingestors.lazy_text_ingestor import LazyTextIngestor

FACTORIES = {
    'pdf': LazyPdfIngestor,
    'docx': LazyWordIngestor,
    'pptx': LazyPowerPointIngestor,
    'txt': LazyTextIngestor,
}

__all__ = ['LazyPdfIngestor', 'LazyWordIngestor', 'LazyPowerPointIngestor', 'LazyTextIngestor', 'FACTORIES']