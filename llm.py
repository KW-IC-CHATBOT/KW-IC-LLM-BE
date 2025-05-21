from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
# from langchain.chat_models import init_chat_model
# from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv
from logger_config import ChatLogger
import logging

# 로거 설정
logger = ChatLogger("llm")

# .env 파일 로드
load_dotenv(override=True)

# API 키 확인
api_key = os.getenv("GEMINI_API")
if not api_key:
    logger.log_error("GEMINI_API key not found in .env file")
    raise ValueError("GEMINI_API key not found in .env file")

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        logger.log_chat(f"Initializing SentenceTransformer with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        logger.log_chat(f"Embedding {len(texts)} documents", level=logging.DEBUG)
        return self.model.encode(texts)
    
    def embed_query(self, text):
        logger.log_chat(f"Embedding query: {text[:100]}...", level=logging.DEBUG)
        return self.model.encode(text)

logger.log_chat("Initializing embeddings model")
embeddings = SentenceTransformerEmbeddings("nlpai-lab/KURE-v1")

logger.log_chat("Loading documents from data directory")
loader = DirectoryLoader("data/", glob="*.md", loader_cls=TextLoader)
docs = loader.load()
logger.log_chat(f"Loaded {len(docs)} documents")

logger.log_chat("Creating FAISS vector store")
vector_store = FAISS.from_documents(docs, embeddings)

logger.log_chat("Initializing Gemini model")
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.log_chat("Successfully initialized Gemini model")
except Exception as e:
    logger.log_error(f"Failed to initialize Gemini model: {str(e)}", exc_info=True)
    raise

def answer_query(query):
    logger.log_chat(f"Processing query: {query}")
    
    try:
        logger.log_chat("Searching for relevant documents", level=logging.DEBUG)
        relevant_docs = vector_store.similarity_search(query, k=3)
        logger.log_chat(f"Found {len(relevant_docs)} relevant documents", level=logging.DEBUG)
        
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        logger.log_chat("Generating response with Gemini model", level=logging.DEBUG)
        response = model.generate_content(prompt, stream=True)
        logger.log_chat("Response generation completed")
        
        return response
    except Exception as e:
        logger.log_error(f"Error in answer_query: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.log_chat("Testing answer_query function")
    answer_query("딥러닝프로그래밍에 대해 알려줘")