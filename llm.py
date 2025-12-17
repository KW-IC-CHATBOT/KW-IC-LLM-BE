from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
# from langchain.chat_models import init_chat_model
# from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.embeddings.base import Embeddings
import google.generativeai as genai
import os
import re
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        logger.log_system("Initializing Reranker model (BAAI/bge-reranker-v2-m3)...", level=logging.DEBUG)
        try:
            reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device='cpu')
            logger.log_system("Successfully initialized Reranker model", level=logging.DEBUG)
        except Exception as e:
            logger.log_error(f"Failed to initialize Reranker: {e}")
            raise
    return reranker

def get_embeddings():
    global embeddings
    if embeddings is None:
        logger.log_system("Initializing embeddings model", level=logging.DEBUG)
        embeddings = SentenceTransformerEmbeddings("jhgan/ko-sroberta-multitask")
    return embeddings

def get_vector_store():
    global vector_store
    if vector_store is None:
        logger.log_system("Loading documents from data directory with Custom Splitter", level=logging.DEBUG)
        
        docs = []
        
        # 1. Standard Splitter (일반 문서용)
        basic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        data_path = "data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        md_files = glob.glob(os.path.join(data_path, "*.md"))
        
        for filepath in md_files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # ---------------- [최종 수정된 정규식 분할 로직] ----------------
                # 파일명에 [Regulation] 태그가 있으면 무조건 정규식 스플리터 적용
                if "[Regulation]" in filename:
                    logger.log_system(f"Applying Regulation Regex Splitter for: {filename}", level=logging.DEBUG)
                    
                    # [부정형 전방탐색 적용]
                    # 1. 줄바꿈(\n) 뒤에 '제N조' 또는 '제N장'이 와야 함
                    # 2. 단, 그 바로 뒤에 조사(의,에,를,는,가,은)가 오면 안 됨 (참조 문구 제외)
                    pattern = r'(?=\n\s*제\s*\d+\s*(?:조|장)(?![의에를는가은]))'
                    
                    splits = re.split(pattern, content)
                    
                    for chunk in splits:
                        if chunk.strip():
                             # 메타데이터에 category 추가
                             docs.append(Document(
                                 page_content=chunk.strip(), 
                                 metadata={"source": filepath, "category": "regulation"}
                             ))
                # -----------------------------------------------------------
                else:
                    logger.log_system(f"Applying Standard Splitter for: {filename}", level=logging.DEBUG)
                    raw_doc = Document(page_content=content, metadata={"source": filepath, "category": "general"})
                    splits = basic_splitter.split_documents([raw_doc])
                    docs.extend(splits)
                    
            except Exception as e:
                logger.log_error(f"Error processing file {filepath}: {e}")

        logger.log_system(f"Total Loaded Chunks: {len(docs)}", level=logging.DEBUG)
        
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

def preload_models():
    """Start-up time model loading to prevent first-request latency."""
    logger.log_system("Starting model preloading... (This may take a while)", level=logging.INFO)
    try:
        get_embeddings()
        get_vector_store()
        get_reranker()
        get_model()
        logger.log_system("All models preloaded successfully.", level=logging.INFO)
    except Exception as e:
        logger.log_error(f"Model preloading failed: {e}")