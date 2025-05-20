from .database import engine
from .models import Base

def init_db():
    """데이터베이스 테이블을 생성합니다."""
    Base.metadata.create_all(bind=engine)

def drop_db():
    """데이터베이스 테이블을 삭제합니다."""
    Base.metadata.drop_all(bind=engine)

if __name__ == "__main__":
    print("데이터베이스 테이블을 생성합니다...")
    init_db()
    print("데이터베이스 테이블이 생성되었습니다.") 