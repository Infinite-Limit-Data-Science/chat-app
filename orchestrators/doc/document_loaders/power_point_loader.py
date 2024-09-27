from pptx import Presentation
from pptx.shapes.shapetree import SlideShapes
from pptx.text.text import TextFrame
from pptx.table import Table
from pptx.enum.shapes import MSO_SHAPE_TYPE
from langchain_core.documents import Document
from typing import List

class PowerPointLoader:
    def __init__(self, file_path):
        self._file_path = file_path

    def load(self) -> List[Document]:
        prs = Presentation(self._file_path)
        text_parts = []
        for slide in prs.slides:
            text_parts.extend(self._extract_text_from_shapes(slide.shapes))

        full_text = "\n".join(text_parts)
        
        return [
            Document(
                page_content=full_text, 
                metadata={'source': self._file_path}
            )
        ]

    def _extract_text_from_shapes(self, shapes: SlideShapes):
        """Extract text from all shapes in a slide"""
        text_parts = []
        for shape in shapes:
            if shape.has_text_frame:
                text_parts.append(self._extract_text_from_text_frame(shape))
            if shape.has_table:
                text_parts.append(self._extract_text_from_table(shape.table))
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                text_parts.append(self._handle_picture(shape)) 
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                text_parts.extend(self._extract_text_from_shapes(shape.shapes))
        return text_parts

    def _extract_text_from_text_frame(self, shape: TextFrame):
        """Extract text from text frames (titles, text boxes, etc.)"""
        return shape.text

    def _extract_text_from_table(self, shape: Table):
        """Extract text from table cells"""
        table_text = []
        for row in shape:
            for cell in row.cells:
                table_text.append(cell.text)
        return "\n".join(table_text)

    def _handle_picture(self, shape):
        """Requires CNN"""
        return '[Image unavailable]'