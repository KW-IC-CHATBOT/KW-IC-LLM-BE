import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 로그 디렉토리 생성
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 로그 파일명 설정 (날짜별)
current_date = datetime.now().strftime("%Y-%m-%d")
CHAT_LOG_FILE = os.path.join(LOG_DIR, f"chat_{current_date}.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, f"error_{current_date}.log")

# 로그 포맷 설정
CHAT_LOG_FORMAT = "%(asctime)s - %(levelname)s - [User: %(user_id)s] - %(message)s"
ERROR_LOG_FORMAT = "%(asctime)s - %(levelname)s - [User: %(user_id)s] - %(message)s\n%(pathname)s:%(lineno)d"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class ChatLogger:
    def __init__(self, user_id="anonymous"):
        self.user_id = user_id
        self.logger = logging.getLogger(f"chat_{user_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # 채팅 로그 핸들러
        chat_handler = RotatingFileHandler(
            CHAT_LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        chat_handler.setLevel(logging.INFO)
        chat_formatter = logging.Formatter(CHAT_LOG_FORMAT, DATE_FORMAT)
        chat_handler.setFormatter(chat_formatter)
        
        # 에러 로그 핸들러
        error_handler = RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(ERROR_LOG_FORMAT, DATE_FORMAT)
        error_handler.setFormatter(error_formatter)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(CHAT_LOG_FORMAT, DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        
        # 핸들러 추가
        self.logger.addHandler(chat_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def log_chat(self, message, level=logging.INFO):
        """채팅 메시지 로깅"""
        extra = {'user_id': self.user_id}
        if level == logging.INFO:
            self.logger.info(message, extra=extra)
        elif level == logging.DEBUG:
            self.logger.debug(message, extra=extra)
        elif level == logging.WARNING:
            self.logger.warning(message, extra=extra)
    
    def log_error(self, message, exc_info=None):
        """에러 로깅"""
        extra = {'user_id': self.user_id}
        self.logger.error(message, exc_info=exc_info, extra=extra)
    
    def log_query(self, query):
        """사용자 쿼리 로깅"""
        self.log_chat(f"User Query: {query}")
    
    def log_response(self, response):
        """봇 응답 로깅"""
        self.log_chat(f"Bot Response: {response}")
    
    def log_context(self, context):
        """컨텍스트 정보 로깅"""
        self.log_chat(f"Context Used: {context}", level=logging.DEBUG) 