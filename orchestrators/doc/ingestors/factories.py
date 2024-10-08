from orchestrators.doc.ingestors.lazy_pdf_ingestor import LazyPdfIngestor
from orchestrators.doc.ingestors.lazy_word_ingestor import LazyWordIngestor
from orchestrators.doc.ingestors.powerpoint_ingestor import PowerPointIngestor
from orchestrators.doc.ingestors.text_ingestor import TextIngestor

FACTORIES = {
    'pdf': LazyPdfIngestor,
    'docx': LazyWordIngestor,
    'pptx': PowerPointIngestor,
    'txt': TextIngestor,
}

__all__ = ['LazyPdfIngestor', 'LazyWordIngestor', 'PowerPointIngestor', 'TextIngestor', 'FACTORIES']