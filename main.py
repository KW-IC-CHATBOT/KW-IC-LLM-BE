from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
from llm import answer_query
from logger_config import ChatLogger
import logging

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
                for chunk in response:
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