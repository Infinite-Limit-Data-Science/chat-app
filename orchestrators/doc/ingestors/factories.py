from orchestrators.doc.ingestors.lazy_pdf_ingestor import LazyPdfIngestor
from orchestrators.doc.ingestors.word_ingestor import WordIngestor
from orchestrators.doc.ingestors.powerpoint_ingestor import PowerPointIngestor
from orchestrators.doc.ingestors.text_ingestor import TextIngestor

FACTORIES = {
    'pdf': LazyPdfIngestor,
    'pptx': PowerPointIngestor,
    'txt': TextIngestor,
    'docx': WordIngestor,
}

__all__ = ['LazyPdfIngestor', 'PowerPointIngestor', 'TextIngestor', 'WordIngestor', 'FACTORIES']