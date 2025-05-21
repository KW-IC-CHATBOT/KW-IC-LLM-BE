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
        self.logger = ChatLogger("system")
        self.logger.log_chat("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.log_chat(f"New client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.logger.log_chat(f"Client disconnected. Remaining connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        self.logger.log_chat(f"Message sent: {message[:100]}...", level=logging.DEBUG)

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
            query = await websocket.receive_text()
            logger.log_query(query)
            
            try:
                response = answer_query(query)
                full_response = ""
                for chunk in response:
                    full_response += chunk.text
                    logger.log_response(chunk.text)
                    await manager.send_personal_message(chunk.text, websocket)
                
                await manager.send_personal_message("[EOS]", websocket)
                logger.log_chat("Response streaming completed")
                
            except Exception as e:
                logger.log_error(f"Error processing query: {str(e)}", exc_info=True)
                await manager.send_personal_message("죄송합니다. 오류가 발생했습니다.", websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.log_chat("WebSocket connection disconnected", level=logging.WARNING)
    except Exception as e:
        logger.log_error(f"Unexpected error: {str(e)}", exc_info=True)
        manager.disconnect(websocket)

if __name__ == "__main__":
    logger = ChatLogger("system")
    logger.log_chat("Starting server on port 35504")
    uvicorn.run(app, host="0.0.0.0", port=35504)