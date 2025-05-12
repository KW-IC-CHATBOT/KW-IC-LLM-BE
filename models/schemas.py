from pydantic import BaseModel
from typing import Optional

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