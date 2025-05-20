from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatMessage(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    session_id: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
    status: str = "error"

class ChatLogRequest(BaseModel):
    message: str
    response: str
    session_id: str
    user_id: str
    tokens_used: Optional[int] = 0
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    error_occurred: Optional[bool] = False
    error_message: Optional[str] = None

class ChatLogResponse(BaseModel):
    id: int
    session_id: str
    message: str
    response: str
    created_at: datetime
    tokens_used: Optional[int]
    processing_time: Optional[float]
    model_used: Optional[str]
    error_occurred: Optional[bool]
    error_message: Optional[str]