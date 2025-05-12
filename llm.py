from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import logging
import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger("llm")
logger.setLevel(logging.DEBUG)


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def embed_query(self, text):
        return self.model.encode(text)

embeddings = SentenceTransformerEmbeddings("nlpai-lab/KURE-v1")

loader = DirectoryLoader("data/", glob="*.md", loader_cls=TextLoader)
docs = loader.load()
vector_store = FAISS.from_documents(docs, embeddings)



genai.configure(api_key=os.getenv("GEMINI_API"))
model = genai.GenerativeModel('gemini-2.0-flash')

def answer_query(query):
    relevant_docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = model.generate_content(prompt, stream=True)
    
    return response


if __name__ == "__main__":
    answer_query("딥러닝프로그래밍에 대해 알려줘")