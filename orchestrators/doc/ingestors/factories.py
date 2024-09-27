from orchestrators.doc.ingestors.pdf_ingestor import PdfIngestor
from orchestrators.doc.ingestors.word_ingestor import WordIngestor
from orchestrators.doc.ingestors.powerpoint_ingestor import PowerPointIngestor
from orchestrators.doc.ingestors.text_ingestor import TextIngestor

FACTORIES = {
    'pdf': PdfIngestor,
    'ppt': PowerPointIngestor,
    'txt': TextIngestor,
    'docx': WordIngestor,
}

__all__ = ['PdfIngestor', 'PowerPointIngestor', 'TextIngestor', 'WordIngestor', 'FACTORIES']