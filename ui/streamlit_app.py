# ui/streamlit_app.py

import streamlit as st
import requests
import json
import time
import uuid
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="ADEK AI Portal", 
    page_icon="🏢",
    layout="wide"
)

# Minimal CSS - only for background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .chat-message-user {
        background-color: #1e40af;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .chat-message-assistant {
        background-color: #f0f2f6;
        color: #333;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 3px solid #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def stream_chat_response(message, session_id, language="English"):
    """Stream chat response from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json={
                "message": message,
                "session_id": session_id,
                "language": language
            },
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            yield f"API Error: {response.status_code}"
            return
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                chunk = line[6:]
                if chunk == '[DONE]':
                    break
                if chunk.strip():
                    yield chunk
                    
    except Exception as e:
        yield f"Error: {str(e)}"

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create layout
col1, col2 = st.columns([2, 1])

# Left column - Website content
with col1:
    # Main content container
    with st.container():
        st.markdown("""
        <div style="background: white; padding: 30px; border-radius: 15px; margin: 20px 0;">
            <div style="background: #1e40af; color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <h1>Welcome to ADEK AI</h1>
                <p>Advanced AI Assistant for all your needs</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Service cards using Streamlit components
        st.markdown("## Our Services")
        
        col1_inner, col2_inner = st.columns(2)
        
        with col1_inner:
            st.info("📄 **Document Processing**\n\nAdvanced AI-powered document analysis and processing capabilities.")
            st.success("🔗 **API Integration**\n\nSeamless integration with existing systems through robust APIs.")
        
        with col2_inner:
            st.warning("⏰ **24/7 Support**\n\nOur AI assistant is available round the clock to help with queries.")
            st.error("🌐 **Multi-language Support**\n\nCommunicate in multiple languages with intelligent responses.")

# Right column - Chat widget
with col2:
    # Chat widget with separate window styling
    st.markdown("""
    <div style="
        background: white; 
        border-radius: 15px; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid #4c6ef5;
        overflow: hidden;
        position: sticky;
        top: 20px;
    ">
    """, unsafe_allow_html=True)
    
    # Chat header
    api_status = check_api_health()
    status_color = "#10b981" if api_status else "#ef4444"
    status_text = "Online" if api_status else "Offline"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #4c6ef5, #339af0);
        color: white;
        padding: 15px 20px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <span>🤖 ADEK AI Assistant</span>
        <span style="font-size: 12px; display: flex; align-items: center;">
            <span style="width: 8px; height: 8px; background: {status_color}; border-radius: 50%; margin-right: 5px;"></span>
            {status_text}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat messages area
    st.markdown("""
    <div style="
        height: 300px;
        overflow-y: auto;
        padding: 15px;
        background: #f8fafc;
    ">
    """, unsafe_allow_html=True)
    
    # Welcome message or chat history
    if not st.session_state.messages:
        st.markdown("""
        <div style="
            text-align: center; 
            color: #64748b; 
            font-style: italic; 
            padding: 20px;
            background: #e0f2fe;
            border-radius: 10px;
            margin: 10px 0;
        ">
            👋 Hello! I'm your AI assistant.<br>
            How can I help you today?
        </div>
        """, unsafe_allow_html=True)
    
    # Display messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="
                background: #4c6ef5;
                color: white;
                padding: 10px 15px;
                border-radius: 15px 15px 5px 15px;
                margin: 8px 0 8px auto;
                max-width: 80%;
                text-align: right;
                word-wrap: break-word;
                font-size: 14px;
            ">
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: #e2e8f0;
                color: #1e293b;
                padding: 10px 15px;
                border-radius: 15px 15px 15px 5px;
                margin: 8px auto 8px 0;
                max-width: 80%;
                border-left: 3px solid #4c6ef5;
                word-wrap: break-word;
                font-size: 14px;
            ">
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close messages area
    
    # Chat input area
    st.markdown("""
    <div style="
        padding: 15px;
        background: white;
        border-top: 1px solid #e2e8f0;
    ">
    """, unsafe_allow_html=True)
    
   
    # Text input with proper clearing
    if "temp_message" in st.session_state:
        current_value = st.session_state.temp_message
        del st.session_state.temp_message
    else:
        current_value = ""
    
    user_input = st.text_input(
        "Type your message:",
        value=current_value,
        placeholder="Type your message here...",
        disabled=not api_status,
        key="chat_input_field"
    )
    
    # Send button
    if st.button("Send Message", type="primary", disabled=not api_status or not user_input.strip(), key="send_message"):
        if user_input.strip():
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            st.session_state.text_input = ""
            # Show thinking indicator
            with st.spinner('AI is thinking...'):
                try:
                    # Stream response
                    full_response = ""
                    response_placeholder = st.empty()
                    
                    for chunk in stream_chat_response(user_input, st.session_state.session_id):
                        full_response += chunk
                        with response_placeholder.container():
                            st.markdown(f"""
                            <div style="
                                background: #fef3c7;
                                color: #92400e;
                                padding: 10px 15px;
                                border-radius: 15px;
                                margin: 8px 0;
                                border-left: 3px solid #f59e0b;
                                font-size: 14px;
                            ">
                                {full_response}▌
                            </div>
                            """, unsafe_allow_html=True)
                        time.sleep(0.02)
                    
                    response_placeholder.empty()
                    
                    # Add final response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                    # Clear the input by rerunning
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
          
    st.markdown('</div>', unsafe_allow_html=True)  # Close input area
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat widget

# Sidebar for session management
with st.sidebar:
    st.title("🤖 ADEK AI")
    st.markdown("---")
    
    st.markdown("### Session Info")
    st.text(f"Session: {st.session_state.session_id}")
    st.text(f"Messages: {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### API Status")
    if api_status:
        st.success("✅ Connected")
    else:
        st.error("❌ Disconnected")
        st.code("uv run python api/main.py")