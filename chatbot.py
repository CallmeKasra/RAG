from sympy.integrals.intpoly import strip

from embedding import Embedding
from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

class Chatbot:
  def __init__(self):
    self.client = OpenAI(
      api_key=os.getenv('Groq_API_Key'),
      base_url='https://api.groq.com/openai/v1'
    )
    self.prompt =f"""You are a Medical information assistant. Answer the user's question using ONLY
            the provided source documents below.

            Rules you must follow:
            
            1. Base your answer ONLY on the provided sources - do not use outside knowledge.
            2. If the sources don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."
            3. Cite sources using [Source N] notation whenever you use information from them.
            4. If sources contradict each other, mention the contradiction.
            5. Be concise - answer in 3-5 sentences unless the question requires more detail.
            6. Never speculate or make assumptions beyond what the sources say.

Sources:\n\n"""

  def chat(self):
    embedding = Embedding()
    while True:
      user_input = input('ask a question or press q to quit: ').strip().lower()
      if user_input == 'q':
        break
      retrieved_texts = embedding.similarity(user_input)
      parts = []
      for index, (doc, score) in enumerate(retrieved_texts):
        parts.append(f"Source {index + 1} | page {doc.metadata.get('page_number', '?')}\n {doc.page_content}")
      context = '\n\n'.join(parts)
      message = [
        {'role': 'system', 'content': self.prompt + context},
        {'role': 'user', 'content': user_input}
        ]
      response = self.client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages=message,
        temperature=0.5
      )
      print(response.choices[0].message.content)

if __name__ == "__main__":
  chatbot = Chatbot()
  chatbot.chat()