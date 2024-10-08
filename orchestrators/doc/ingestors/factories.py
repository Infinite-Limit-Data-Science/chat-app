from orchestrators.doc.ingestors.lazy_pdf_ingestor import LazyPdfIngestor
from orchestrators.doc.ingestors.lazy_word_ingestor import LazyWordIngestor
from orchestrators.doc.ingestors.lazy_powerpoint_ingestor import LazyPowerPointIngestor
from orchestrators.doc.ingestors.text_ingestor import TextIngestor

FACTORIES = {
    'pdf': LazyPdfIngestor,
    'docx': LazyWordIngestor,
    'pptx': LazyPowerPointIngestor,
    'txt': TextIngestor,
}

__all__ = ['LazyPdfIngestor', 'LazyWordIngestor', 'LazyPowerPointIngestor', 'TextIngestor', 'FACTORIES']