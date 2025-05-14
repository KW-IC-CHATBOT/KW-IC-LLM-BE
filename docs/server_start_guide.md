# 서버 실행 가이드 (로컬)

이 문서는 서버를 완전히 종료한 뒤, 처음부터 환경설정 및 실행까지의 전체 과정을 안내합니다.

---

## 1. 서버 프로세스 종료

이미 실행 중인 서버가 있다면 종료합니다.

```bash
# 실행 중인 python main.py 프로세스 확인
tasklist | grep python   # (Windows)
ps -ef | grep "python main.py" | grep -v grep   # (Mac/Linux)

# 프로세스 종료 (예: PID가 12345인 경우)
kill 12345
```

---

## 2. 환경 변수(.env) 설정

프로젝트 루트에 `.env` 파일이 존재해야 하며, 다음과 같이 작성합니다.

```
GEMINI_API=여기에_본인_API_KEY_입력
```

---

## 3. 의존성 설치

필요한 패키지가 모두 설치되어 있는지 확인합니다.

```bash
pip install -r requirements.txt
pip install sentence-transformers faiss-cpu google-generativeai python-dotenv
```

---

## 4. 로그 디렉토리 생성 (최초 1회)

```bash
mkdir -p logs
```

---

## 5. 서버 실행

```bash
python main.py
```

---

## 6. 정상 동작 확인

- 터미널에 `Starting server on port 35504` 메시지가 보이면 정상 실행
- 브라우저에서 [http://localhost:35504/](http://localhost:35504/) 접속 시 `good!` 메시지 확인
- 클라이언트(https://kw-ic.vercel.app/)에서 챗봇 대화 시도
- 로그 파일(`logs/chat_YYYY-MM-DD.log`)에 대화 내용 기록되는지 확인

---

## 7. 서버 중지

실행 중인 터미널에서 `Ctrl + C` 입력

---

## 8. 참고

- .env 파일은 git에 올리지 않습니다.
- 로그 파일은 `logs/` 디렉토리에서 확인할 수 있습니다.
- 에러 발생 시 `logs/error_YYYY-MM-DD.log` 파일을 확인하세요.
