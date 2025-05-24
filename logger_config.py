import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import psycopg2 # SQLite 대신 psycopg2 임포트
# from dotenv import load_dotenv # .env 파일 로드는 main.py나 llm.py에서 한다고 가정
from urllib.parse import urlparse # URL 파싱을 위해 추가

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
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.logger = logging.getLogger(f"chat_logger_{user_id}")
        self.logger.setLevel(logging.INFO)
        
        # 로거가 이미 핸들러를 가지고 있다면 중복 추가하지 않음
        if not self.logger.handlers:
            # 파일 핸들러 설정
            file_handler = logging.FileHandler(f'logs/chat_{user_id}.log')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # 콘솔 핸들러 설정
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.db = None
        try:
            self.connect_db()
        except Exception as e:
            self.logger.warning(f"Database connection failed: {str(e)}. Continuing without database logging.")

    def connect_db(self):
        try:
            self.db = psycopg2.connect(
                host=os.getenv('PG_HOST', 'localhost'),
                port=os.getenv('PG_PORT', '5432'),
                dbname=os.getenv('PG_DBNAME', 'postgres'),
                user=os.getenv('PG_USER', 'postgres'),
                password=os.getenv('PG_PASSWORD', 'whffu18')
            )
            self.logger.info("PostgreSQL database connection established")
            self._init_db()
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {str(e)}", exc_info=True)
            raise

    def close_db(self):
        if self.db:
            try:
                self.db.close()
                self.logger.info("PostgreSQL database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {str(e)}", exc_info=True)

    def __del__(self):
        try:
            self.close_db()
        except Exception as e:
            print(f"Error in ChatLogger destructor: {str(e)}")

    def _init_db(self):
        """데이터베이스 테이블 생성 (테이블이 없을 경우)"""
        if not self.db:
            return

        try:
            cursor = self.db.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    user_id VARCHAR(255),
                    log_level VARCHAR(50),
                    log_type VARCHAR(50),
                    message TEXT,
                    details TEXT
                )
            ''')
            self.db.commit()
            self.logger.info("chat_logs table checked/created")
        except Exception as e:
            self.logger.error(f"Error initializing database table: {e}", exc_info=True)

    def _log_to_db(self, level, log_type, message, details=None):
        """데이터베이스에 로그 저장"""
        if not self.db:
            return

        try:
            cursor = self.db.cursor()
            timestamp = datetime.now()
            cursor.execute('''
                INSERT INTO chat_logs (timestamp, user_id, log_level, log_type, message, details)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (timestamp, self.user_id, level, log_type, message, details))
            self.db.commit()
        except Exception as e:
            self.logger.error(f"Error logging to database: {e}", exc_info=True)
            self.close_db()

    # 기존 로깅 메서드 수정 (데이터베이스 로깅 추가)
    def _log_chat(self, message, type="system", level=logging.INFO, details=None):
        """채팅 메시지 로깅"""
        extra = {'user_id': self.user_id}
        level_name = logging.getLevelName(level)
        # 파일 로깅
        if level == logging.INFO:
            self.logger.info(message, extra=extra)
        elif level == logging.DEBUG:
            self.logger.debug(message, extra=extra)
        elif level == logging.WARNING:
            self.logger.warning(message, extra=extra)
        elif level == logging.ERROR:
            self.logger.error(message, extra=extra)
        # DB 로깅
        self._log_to_db(level_name, type, message, details)

    def log_system(self, message, type="system", level=logging.INFO, exc_info=None, details=None):
        """시스템 로깅        """
        # DB 로깅
        details = None
        if exc_info:
             # exc_info 정보를 문자열로 변환하여 details에 저장
             import traceback
             details = traceback.format_exc()
        
        self.log_chat(message=message, type=type, level=level, details=details)

    def log_chat(self, type="chat", level=logging.INFO, message=None, details=None):
        """사용자 쿼리 로깅
        type: chat, query, response, context
        level: INFO, WARNING
        message: 로깅할 메시지
        """
        self._log_chat(message=message, type=type, level=level, details=details)

    # 애플리케이션 종료 시 데이터베이스 연결을 안전하게 닫기 위한 메서드 추가
    def __del__(self):
        self.close_db() 