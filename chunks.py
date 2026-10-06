from langchain_core.documents import Document
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunk:
  def __init__(self, file_name):
    self.file_name = file_name


  def pdf_reader(self):
    reader = PdfReader(self.file_name)
    docs = list()
    for index, page in enumerate(reader.pages):
      doc = Document(
        page_content=page.extract_text(),
        metadata={'source': self.file_name, 'page_number': index + 1}
      )
      docs.append(doc)
    return docs

  def chunk_generator(self):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=110,
      chunk_overlap=10
    )
    return splitter.split_documents(self.pdf_reader())
