from typing import List
import docx
from langchain_core.documents import Document
from orchestrators.doc.document_loaders.base_loader import BaseLoader

class WordLoader(BaseLoader):
    def load(self) -> List[Document]:
        doc = docx.Document(self._file_path)
        full_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return [
            Document(
                page_content=full_text, 
                metadata={'source': self._file_path}
            )
        ]