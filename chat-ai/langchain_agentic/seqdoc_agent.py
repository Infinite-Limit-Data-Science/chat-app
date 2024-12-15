from langchain.tools import Tool

# the whole doc upload piece will need to be part of the Tool
def sequential_document_qa(prompt: str) -> str:
    pass

seqdoc_agent = Tool(
    name="get_answer_from_sequencial_doc",
    func=sequential_document_qa,
    description="Streams generated responses from sequential documents"
)