import os
import psycopg2
from dotenv import load_dotenv
import sys

def check_env_variables():
    required_vars = [
        "GEMINI_API",
        "PG_HOST",
        "PG_PORT",
        "PG_DBNAME",
        "PG_USER",
        "PG_PASSWORD"
    ]
    
    load_dotenv(override=True)
    
    # 환경 변수 기본값 설정
    if not os.getenv("PG_HOST"):
        os.environ["PG_HOST"] = "localhost"
    if not os.getenv("PG_PORT"):
        os.environ["PG_PORT"] = "5418"
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    print("All required environment variables are set.")
    print(f"Database connection settings:")
    print(f"Host: {os.getenv('PG_HOST')}")
    print(f"Port: {os.getenv('PG_PORT')}")
    print(f"Database: {os.getenv('PG_DBNAME')}")
    print(f"User: {os.getenv('PG_USER')}")

def check_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            dbname=os.getenv("PG_DBNAME"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD")
        )
        
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"Successfully connected to PostgreSQL. Version: {version[0]}")
        
        # 테이블 존재 여부 확인
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()
        print(f"Available tables: {[table[0] for table in tables]}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("Checking environment variables...")
    check_env_variables()
    
    print("\nChecking database connection...")
    check_db_connection() 