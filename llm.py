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
from prompt_security import PromptSecurity

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
    _instance = None
    _model = None

    def __new__(cls, model_name):
        if cls._instance is None:
            cls._instance = super(SentenceTransformerEmbeddings, cls).__new__(cls)
            cls._instance.model_name = model_name
        return cls._instance

    def __init__(self, model_name):
        if self._model is None:
            logger.log_system(f"Initializing SentenceTransformer with model: {model_name}")
            self._model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        logger.log_system(f"Embedding {len(texts)} documents", level=logging.DEBUG)
        return self._model.encode(texts)
    
    def embed_query(self, text):
        logger.log_system(f"Embedding query: {text[:100]}...", level=logging.DEBUG)
        return self._model.encode(text)

# Lazy loading for embeddings
embeddings = None
vector_store = None
model = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        logger.log_system("Initializing embeddings model", level=logging.DEBUG)
        embeddings = SentenceTransformerEmbeddings("jhgan/ko-sroberta-multitask")
    return embeddings

def get_vector_store():
    global vector_store
    if vector_store is None:
        logger.log_system("Loading documents from data directory", level=logging.DEBUG)
        loader = DirectoryLoader("data/", glob="*.md", loader_cls=TextLoader)
        docs = loader.load()
        logger.log_system(f"Loaded {len(docs)} documents", level=logging.DEBUG)
        
        logger.log_system("Creating FAISS vector store", level=logging.DEBUG)
        vector_store = FAISS.from_documents(docs, get_embeddings())
    return vector_store

def get_model():
    global model
    if model is None:
        logger.log_system("Initializing Gemini model", level=logging.DEBUG)
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            logger.log_system("Successfully initialized Gemini model", level=logging.DEBUG)
        except Exception as e:
            logger.log_system(f"Failed to initialize Gemini model: {str(e)}", level=logging.ERROR, exc_info=True)
            raise
    return model
