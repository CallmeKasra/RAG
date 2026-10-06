from chunks import Chunk
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import hashlib


class Embedding():
  def __init__(self):
    self.docs = Chunk('symptoms.pdf').chunk_generator()
    self.ids = self.gen_id()
    self.model = HuggingFaceEmbeddings(
      model_name='sentence-transformers/all-MiniLM-L6-v2',
      encode_kwargs={'normalize_embeddings': True}
    )
    self.vectorStore = Chroma(
      collection_name='medical_data',
      persist_directory='./chroma',
      embedding_function=self.model,
      collection_metadata={'hnsw:space': 'cosine'}
    )
    self.embed_docs()

  def gen_id(self):
    return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in self.docs]


  def embed_docs(self):
    self.vectorStore.add_documents(self.docs, ids=self.ids)

  def similarity(self, query):
    return self.vectorStore.similarity_search_with_score(query, k=3)