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
    def __init__(self, user_id="anonymous"):
        # DATABASE_URL 환경 변수 읽기 및 파싱
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            self.db_config = None
            print("DATABASE_URL environment variable not set.") # 초기 에러 메시지 (로거 사용 전)
        else:
            try:
                result = urlparse(db_url)
                self.db_config = {
                    "dbname": result.path[1:], # 경로에서 '/' 제거
                    "user": result.username,
                    "password": result.password,
                    "host": result.hostname,
                    "port": result.port or 5432 # 포트가 없으면 기본값 5432
                }
            except Exception as e:
                self.db_config = None
                print(f"Error parsing DATABASE_URL: {e}") # 초기 에러 메시지

        # 데이터베이스 연결 객체
        self.conn = None
        self._connect_db() # 초기 연결 시도

        # DB 연결 성공 시에만 테이블 초기화 시도
        if self.conn:
            self._init_db() # 테이블 초기화

        self.user_id = user_id
        self.logger = logging.getLogger(f"chat_{user_id}")
        self.logger.setLevel(logging.DEBUG)

        # 파일 로깅 핸들러 (기존 코드 유지)
        chat_handler = RotatingFileHandler(
            CHAT_LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        chat_handler.setLevel(logging.INFO)
        chat_formatter = logging.Formatter(CHAT_LOG_FORMAT, DATE_FORMAT)
        chat_handler.setFormatter(chat_formatter)

        error_handler = RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(ERROR_LOG_FORMAT, DATE_FORMAT)
        error_handler.setFormatter(error_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(CHAT_LOG_FORMAT, DATE_FORMAT)
        console_handler.setFormatter(console_formatter)

        # 핸들러 추가
        self.logger.addHandler(chat_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)

    def _connect_db(self):
        """데이터베이스 연결"""
        if self.conn is not None and not self.conn.closed:
             return # 이미 연결되어 있으면 다시 연결하지 않음

        if not self.db_config:
             # DATABASE_URL 설정 문제로 db_config가 없는 경우
             return

        try:
            # 환경 변수가 모두 설정되었는지 확인 (파싱 결과 확인)
            if not all(self.db_config.values()):
                 self.logger.error("PostgreSQL connection details incomplete from DATABASE_URL.")
                 self.conn = None
                 return

            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True # 자동 커밋 설정
            self.logger.info("PostgreSQL database connected")
        except psycopg2.OperationalError as e:
            self.logger.error(f"Error connecting to PostgreSQL database: {e}")
            self.conn = None
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to database: {e}")
            self.conn = None

    def close_db(self):
        """데이터베이스 연결 종료"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.logger.info("PostgreSQL database connection closed")
            self.conn = None

    def _init_db(self):
        """데이터베이스 테이블 생성 (테이블이 없을 경우)"""
        self._connect_db() # 연결 확인
        if not self.conn:
            return

        try:
            cursor = self.conn.cursor()
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
            self.logger.info("chat_logs table checked/created")
        except Exception as e:
            self.logger.error(f"Error initializing database table: {e}", exc_info=True)

    def _log_to_db(self, level, log_type, message, details=None):
        """데이터베이스에 로그 저장"""
        self._connect_db() # 연결 확인
        if not self.conn:
            self.logger.error("Cannot log to DB: Database connection not available.")
            return

        try:
            cursor = self.conn.cursor()
            # TIMESTAMP WITH TIME ZONE 형식에 맞게 현재 시간 생성
            timestamp = datetime.now()
            cursor.execute('''
                INSERT INTO chat_logs (timestamp, user_id, log_level, log_type, message, details)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (timestamp, self.user_id, level, log_type, message, details))
            # autocommit = True 설정으로 commit 생략 가능
            # self.conn.commit()
            # self.logger.debug(f"Logged to DB: {log_type} - {message[:50]}...") # 디버깅용
        except Exception as e:
            self.logger.error(f"Error logging to database: {e}", exc_info=True)
            # 데이터베이스 에러 발생 시 연결 끊고 다음 시도 시 재연결하도록 함
            self.close_db()

    # 기존 로깅 메서드 수정 (데이터베이스 로깅 추가)
    def log_chat(self, message, level=logging.INFO):
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
        # DB 로깅
        self._log_to_db(level_name, "chat", message)

    def log_error(self, message, exc_info=None):
        """에러 로깅"""
        extra = {'user_id': self.user_id}
        # 파일 로깅
        self.logger.error(message, exc_info=exc_info, extra=extra)
        # DB 로깅
        details = None
        if exc_info:
             # exc_info 정보를 문자열로 변환하여 details에 저장
             import traceback
             details = traceback.format_exc()
        self._log_to_db("ERROR", "error", message, details)

    def log_query(self, query):
        """사용자 쿼리 로깅"""
        # print(f"[DEBUG] log_query called: {query}") # 디버깅용 출력 (기존 코드)
        # 파일 로깅
        self.log_chat(f"User Query: {query}")
        # DB 로깅
        self._log_to_db("INFO", "query", query)

    def log_response(self, response):
        """봇 응답 로깅"""
        # 파일 로깅
        self.log_chat(f"Bot Response: {response}")
        # DB 로깅
        self._log_to_db("INFO", "response", response)

    def log_context(self, context):
        """컨텍스트 정보 로깅"""
        # 파일 로깅
        self.log_chat(f"Context Used: {context}", level=logging.DEBUG)
        # DB 로깅
        self._log_to_db("DEBUG", "context", f"Context Used: {context}", context)

    # 애플리케이션 종료 시 데이터베이스 연결을 안전하게 닫기 위한 메서드 추가
    def __del__(self):
        self.close_db() 