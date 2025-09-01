from pydantic import BaseModel, RootModel, Field
from typing import List, Union, Optional, Dict, Any
from enum import Enum


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question or message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    language: str = Field("English", description="Response language")

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]
    total_messages: int

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str = "1.0.0"

class ResponseType(str, Enum):
    """Types of responses the AI can return."""
    TEXT = "text"           # Regular chat response
    FORM = "form"           # JSON form to be rendered
    ACTION = "action"       # Action buttons/links
    TABLE = "table"         # Structured data table
    ERROR = "error"         # Error response


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant's response")
    response_type: ResponseType = Field(ResponseType.TEXT, description="Type of response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for response")
    session_id: str = Field(..., description="Session ID used")

class StreamingChatResponse(BaseModel):
    """Structure for each streaming chunk."""
    content: str = Field(..., description="Content chunk")
    response_type: ResponseType = Field(ResponseType.TEXT, description="Type of response") 
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    is_complete: bool = Field(False, description="Whether this is the final chunk")

# Form structure for violations
class ViolationForm(BaseModel):
    form_id: str = Field(..., description="Unique form identifier")
    title: str = Field(..., description="Form title")
    fields: List[Dict[str, Any]] = Field(..., description="Form fields definition")
    submit_endpoint: str = Field(..., description="Where to submit the form")



class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"