import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directories to path for custom imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger.custom_logger import CustomLogger
from exception.custom_exception import CommonException
from src.data_retrieval.retrieval import ConversationalRAG
from models.models import ChatRequest, ChatResponse, SessionHistoryResponse, HealthResponse

# Load environment variables
load_dotenv()

# Initialize custom logger
logger = CustomLogger().get_logger(__file__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG AI Assistant API",
    description="Conversational RAG system with streaming support",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance (initialized on startup)
rag_system = None


async def get_rag_system() -> ConversationalRAG:
    """Dependency to get RAG system instance."""
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_system
    try:
        logger.info("Starting RAG API service initialization")
        
        # Get configuration from environment
        SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
        SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        INDEX_NAME = os.getenv("INDEX_NAME", "documents-index")
        
        # Validate required environment variables
        required_vars = {
            "AZURE_SEARCH_ENDPOINT": SEARCH_ENDPOINT,
            "AZURE_SEARCH_KEY": SEARCH_KEY,
            "OPENAI_API_KEY": OPENAI_API_KEY
        }
        
        for var_name, var_value in required_vars.items():
            if not var_value:
                raise ValueError(f"{var_name} environment variable is required")
        
        # Initialize RAG system without session (global instance)
        rag_system = ConversationalRAG()
        
        # Load retriever
        rag_system.load_retriever(
            search_endpoint=SEARCH_ENDPOINT,
            search_key=SEARCH_KEY,
            index_name=INDEX_NAME,
            openai_api_key=OPENAI_API_KEY
        )
        
        logger.info("RAG API service initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize RAG service", error=str(e))
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="healthy",
            message="RAG AI Assistant API is running"
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, rag: ConversationalRAG = Depends(get_rag_system)):
    """Chat endpoint for single response."""
    try:
        logger.info("Processing chat request", 
                   message=request.message[:100],
                   session_id=request.session_id,
                   language=request.language)
        
        # Create session-specific RAG instance
        session_rag = ConversationalRAG(
            session_id=request.session_id,
            retriever=rag.retriever
        )
        
        # Get response
        response = session_rag.invoke(
            user_input=request.message,
            language=request.language
        )
        
        logger.info("Chat response generated", 
                   session_id=session_rag.session_id,
                   response_length=len(response))
        
        return ChatResponse(
            response=response,
            session_id=session_rag.session_id
        )
        
    except Exception as e:
        logger.error("Chat request failed", 
                    message=request.message,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, rag: ConversationalRAG = Depends(get_rag_system)):
    """Streaming chat endpoint - returns chunks as they're generated."""
    try:
        logger.info("Processing streaming chat request", 
                   message=request.message[:100],
                   session_id=request.session_id,
                   language=request.language)
        
        # Create session-specific RAG instance
        session_rag = ConversationalRAG(
            session_id=request.session_id,
            retriever=rag.retriever
        )
        
        async def generate_response():
            """Generator for streaming response."""
            try:
                # Stream the response chunk by chunk
                for chunk in session_rag.stream(
                    user_input=request.message,
                    language=request.language
                ):
                    # Format as server-sent events for frontend
                    yield f"data: {chunk}\n\n"
                
                # Send end signal
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                logger.error("Streaming generation failed", error=str(e))
                yield f"data: Error: {str(e)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error("Streaming chat request failed", 
                    message=request.message,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Streaming chat failed: {str(e)}")


@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """Get chat history for a specific session."""
    try:
        # Create temporary RAG instance to access session history
        temp_rag = ConversationalRAG(session_id=session_id)
        chat_history = temp_rag.get_chat_history()
        
        # Format history for response
        formatted_history = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                formatted_history.append({
                    "human": chat_history[i].content,
                    "ai": chat_history[i + 1].content
                })
        
        logger.info("Retrieved session history", 
                   session_id=session_id,
                   total_messages=len(chat_history))
        
        return SessionHistoryResponse(
            session_id=session_id,
            history=formatted_history,
            total_messages=len(chat_history)
        )
        
    except Exception as e:
        logger.error("Failed to get session history", 
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")


@app.post("/session/{session_id}/clear")
async def clear_session_history(session_id: str):
    """Clear chat history for a specific session."""
    try:
        temp_rag = ConversationalRAG(session_id=session_id)
        temp_rag.clear_chat_history()
        
        logger.info("Session history cleared", session_id=session_id)
        
        return {"message": f"Session {session_id} history cleared successfully"}
        
    except Exception as e:
        logger.error("Failed to clear session history", 
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "streaming_chat": "/chat/stream",
            "health": "/health",
            "session_history": "/session/{session_id}/history",
            "clear_session": "/session/{session_id}/clear"
        }
    }


# Run the API server
if __name__ == "__main__":
    try:
        logger.info("Starting RAG API server")
        
        # Get port from environment or use default
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=True,  # Auto-reload on code changes
            log_level="info"
        )
        
    except Exception as e:
        app_exc = CommonException(f"Failed to start API server: {e}", sys)
        logger.error(str(app_exc))
        raise app_exc