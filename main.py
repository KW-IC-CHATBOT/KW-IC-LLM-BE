from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
from llm import answer_query
from logger_config import ChatLogger
import logging 
from sqlalchemy.orm import Session
from models.database import get_db
from models.models import ChatMessage as ChatMessageModel
from models.schemas import ChatLogRequest, ChatLogResponse
from datetime import datetime
import json

app = FastAPI()

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.chat_histories: dict = {}  # Store chat histories per connection
        self.logger = ChatLogger("system")
        self.logger.log_chat("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.chat_histories[id(websocket)] = []  # Initialize empty chat history
        self.logger.log_chat(f"New client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if id(websocket) in self.chat_histories:
            del self.chat_histories[id(websocket)]  # Clean up chat history
        self.logger.log_chat(f"Client disconnected. Remaining connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        self.logger.log_chat(f"Message sent: {message[:100]}...", level=logging.DEBUG)

    def add_to_history(self, websocket: WebSocket, query: str, response: str):
        if id(websocket) not in self.chat_histories:
            self.chat_histories[id(websocket)] = []
        self.chat_histories[id(websocket)].append({
            "query": query,
            "response": response
        })

    def get_chat_history(self, websocket: WebSocket):
        return self.chat_histories.get(id(websocket), [])

manager = ConnectionManager()

@app.get("/")
async def root():
    return "good!"

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections)
    }

@app.post("/log/chat", response_model=ChatLogResponse)
async def log_chat(chat_log: ChatLogRequest, db: Session = Depends(get_db)):
    try:
        db_chat = ChatMessageModel(
            session_id=chat_log.session_id,
            message=chat_log.message,
            response=chat_log.response,
            tokens_used=chat_log.tokens_used,
            processing_time=chat_log.processing_time,
            model_used=chat_log.model_used,
            error_occurred=chat_log.error_occurred,
            error_message=chat_log.error_message
        )
        db.add(db_chat)
        db.commit()
        db.refresh(db_chat)
        return db_chat
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 각 연결마다 고유한 로거 생성
    logger = ChatLogger(f"user_{id(websocket)}")
    await manager.connect(websocket)
    
    try:
        while True:
            raw_data = await websocket.receive_text()
            
            try:
                if raw_data.startswith('{') and raw_data.endswith('}'):
                    data = json.loads(raw_data)
                    query = data.get('query', raw_data)
                else:
                    query = raw_data
            except json.JSONDecodeError:
                query = raw_data
            
            logger.log_query(type="query", query=query)
            
            try:
                chat_history = manager.get_chat_history(websocket)
                response = answer_query(query, chat_history)
                full_response = ""
                
                for chunk in response:
                    full_response += chunk.text
                    await manager.send_personal_message(chunk.text, websocket)
                
                manager.add_to_history(websocket, query, full_response)
                logger.log_response(type="response", response=full_response)
                
                await manager.send_personal_message("[EOS]", websocket)
                logger.log_info("Response streaming completed")
                
            except Exception as e:
                logger.log_error(message=f"Error processing query: {str(e)}", exc_info=True)
                await manager.send_personal_message("죄송합니다. 오류가 발생했습니다.", websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.log_info("WebSocket connection disconnected", exc_info=True)
    except Exception as e:
        logger.log_error(f"Unexpected error: {str(e)}", exc_info=True)
        manager.disconnect(websocket)

if __name__ == "__main__":
    logger = ChatLogger("system")
    logger.log_info("Starting server on port 35504")
    uvicorn.run(app, host="0.0.0.0", port=35504)