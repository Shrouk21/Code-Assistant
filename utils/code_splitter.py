import re
from langchain_core.documents import Document

def split_code_by_function(doc: Document):
    text = doc.page_content
    matches = list(re.finditer(r"^def\s+\w+\(.*?\):", text, re.MULTILINE))
    if not matches:
        return [doc]
    starts = [m.start() for m in matches] + [len(text)]
    return [
        Document(page_content=text[starts[i]:starts[i+1]].strip(), metadata=doc.metadata)
        for i in range(len(starts) - 1)
    ]
