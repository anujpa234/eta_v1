import os
import sys
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path for custom imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from logger.custom_logger import CustomLogger
from exception.custom_exception import CommonException
from prompts.prompt_library import PROMPT_REGISTRY, DEFAULT_LANGUAGE


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.utils import Input, Output
from operator import itemgetter


from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential


load_dotenv()
logger = CustomLogger().get_logger(__file__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Config loaded successfully", config_path=config_path)
        return config
    except Exception as e:
        raise CommonException(f"Failed to load config file from {config_path}: {e}", sys)

class AzureSearchRetriever:
    """Simple retriever for Azure AI Search with vector and text search."""
    
    def __init__(self, search_endpoint: str, search_key: str, index_name: str, openai_api_key: str, config: dict = None):
        """Initialize the retriever."""
        try:
            self.search_endpoint = search_endpoint
            self.search_key = search_key
            self.index_name = index_name
            self.credential = AzureKeyCredential(search_key)
            
            # Load config
            if config is None:
                config = load_config()
            
            # Initialize search client
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=index_name,
                credential=self.credential
            )
            
            # Initialize embeddings for vector search from config
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for vector search")
            
            embedding_model_name = config["embedding_model"]["openai"]["model_name"]
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model=embedding_model_name
            )
            
            logger.info("AzureSearchRetriever initialized successfully", 
                       endpoint=search_endpoint,
                       index_name=index_name)
            
        except Exception as e:
            raise CommonException(f"Failed to initialize AzureSearchRetriever: {e}", sys)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search (combines text + vector)."""
        try:
            logger.info("Performing hybrid search", query=query, top_k=top_k)
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Perform hybrid search (text + vector)
            results = self.search_client.search(
                search_text=query,  # Text search
                vector_queries=[vector_query],  # Vector search
                top=top_k,
                select=["id", "content", "file_path", "file_type", "chunk_index"]
            )
            
            # Convert results to list of documents
            documents = []
            for result in results:
                documents.append({
                    "page_content": result.get("content", ""),
                    "metadata": {
                        "id": result.get("id", ""),
                        "file_path": result.get("file_path", ""),
                        "file_type": result.get("file_type", ""),
                        "chunk_index": result.get("chunk_index", 0),
                        "score": result.get("@search.score", 0.0)
                    }
                })
            
            logger.info("Hybrid search completed", 
                       query=query,
                       results_found=len(documents))
            
            return documents
            
        except Exception as e:
            raise CommonException(f"Hybrid search failed: {e}", sys)



class ConversationalRAG:
    """Conversational RAG system with session management and LCEL chains."""
    
    def __init__(self, session_id: Optional[str] = None, retriever=None, config: dict = None):
        """Initialize the Conversational RAG system."""
        try:
            # Load config
            if config is None:
                config = load_config()
            
            self.session_id = session_id or f"session_{os.urandom(8).hex()}"
            self.retriever = retriever
            
            # Session management - store chat histories
            if not hasattr(ConversationalRAG, '_session_store'):
                ConversationalRAG._session_store = {}
            
            # Initialize session if it doesn't exist
            if self.session_id not in ConversationalRAG._session_store:
                ConversationalRAG._session_store[self.session_id] = []
           
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            model_name = config["llm"]["openai"]["model_name"]
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=openai_api_key
            )
            
            # 3. Load prompts from prompt library
            self.contextualize_prompt = PROMPT_REGISTRY["contextualize_question"]
            self.qa_prompt = PROMPT_REGISTRY["context_qa"]
            
            # Initialize chain to None
            self.chain = None
            
            logger.info("ConversationalRAG initialized", 
                       session_id=self.session_id,
                       has_retriever=self.retriever is not None)
            
            # 4. Build LCEL chain if retriever is provided
            if self.retriever is not None:
                self._build_lcel_chain()
            
        except Exception as e:
            raise CommonException(f"Failed to initialize ConversationalRAG: {e}", sys)
    
    def load_retriever(self, search_endpoint: str, search_key: str, index_name: str, openai_api_key: str, config: dict = None):
        """Load Azure Search retriever and build LCEL chain."""
        try:
            if config is None:
                config = load_config()
                
            logger.info("Loading Azure Search retriever", 
                       search_endpoint=search_endpoint,
                       index_name=index_name)
            
            # Create retriever with config
            self.retriever = AzureSearchRetriever(
                search_endpoint=search_endpoint,
                search_key=search_key,
                index_name=index_name,
                openai_api_key=openai_api_key,
                config=config
            )
            
            # Build the LCEL chain
            self._build_lcel_chain()
            
            logger.info("Retriever loaded and chain built successfully")
            
        except Exception as e:
            raise CommonException(f"Failed to load retriever: {e}", sys)
    
    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a single string."""
        try:
            formatted_docs = []
            for doc in docs:
                content = doc.get("page_content", "")
                if content.strip():
                    formatted_docs.append(content.strip())
            
            return "\n\n".join(formatted_docs)
            
        except Exception as e:
            logger.error("Error formatting documents", error=str(e))
            return ""
    
    def _build_lcel_chain(self):
        """Build the complete LCEL chain for conversational RAG."""
        try:
            if not self.retriever:
                raise ValueError("Retriever must be loaded before building chain")
            
            logger.info("Building LCEL chain")
            
            # 1. Question rewriter based on chat history
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                    "language": itemgetter("language")
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 2. Retrieve docs for rewritten question
            def retrieve_docs(rewritten_question: str) -> str:
                docs = self.retriever.search(rewritten_question, top_k=5)
                return self._format_docs(docs)
            
            retrieve_docs_chain = question_rewriter | RunnableLambda(retrieve_docs)
            
            # 3. Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs_chain,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                    "language": itemgetter("language"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("LCEL chain built successfully")
            
        except Exception as e:
            raise CommonException(f"Failed to build LCEL chain: {e}", sys)
    
    def get_chat_history(self) -> List[BaseMessage]:
        """Get chat history for current session."""
        return ConversationalRAG._session_store.get(self.session_id, [])
    
    def add_to_chat_history(self, human_message: str, ai_message: str):
        """Add messages to chat history."""
        try:
            if self.session_id not in ConversationalRAG._session_store:
                ConversationalRAG._session_store[self.session_id] = []
            
            ConversationalRAG._session_store[self.session_id].extend([
                HumanMessage(content=human_message),
                AIMessage(content=ai_message)
            ])
            
            logger.info("Added messages to chat history", 
                       session_id=self.session_id,
                       history_length=len(ConversationalRAG._session_store[self.session_id]))
            
        except Exception as e:
            logger.error("Failed to add to chat history", error=str(e))
    
    def clear_chat_history(self):
        """Clear chat history for current session."""
        try:
            ConversationalRAG._session_store[self.session_id] = []
            logger.info("Chat history cleared", session_id=self.session_id)
        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
    
    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None, language: str = DEFAULT_LANGUAGE) -> str:
        """Main method to process user input and generate response."""
        try:
            if not self.chain:
                raise ValueError("Chain not built. Please load a retriever first using load_retriever()")
            
            # Use provided chat history or get from session
            if chat_history is None:
                chat_history = self.get_chat_history()
            
            logger.info("Processing user query", 
                       session_id=self.session_id,
                       user_input=user_input[:100],  # Log first 100 chars
                       chat_history_length=len(chat_history),
                       language=language)
            
            # Prepare input for the chain
            chain_input = {
                "input": user_input,
                "chat_history": chat_history,
                "language": language
            }
            
            # Invoke the chain
            response = self.chain.invoke(chain_input)
            
            # Add to chat history
            self.add_to_chat_history(user_input, response)
            
            logger.info("Response generated successfully", 
                       session_id=self.session_id,
                       response_length=len(response))
            
            return response
            
        except Exception as e:
            raise CommonException(f"Failed to process user input: {e}", sys)
    
    def stream(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None, language: str = DEFAULT_LANGUAGE):
        """Stream response for user input."""
        try:
            if not self.chain:
                raise ValueError("Chain not built. Please load a retriever first using load_retriever()")
            
            # Use provided chat history or get from session
            if chat_history is None:
                chat_history = self.get_chat_history()
            
            # Prepare input for the chain
            chain_input = {
                "input": user_input,
                "chat_history": chat_history,
                "language": language
            }
            
            # Stream the response
            full_response = ""
            for chunk in self.chain.stream(chain_input):
                full_response += chunk
                yield chunk
            
            # Add to chat history after streaming is complete
            self.add_to_chat_history(user_input, full_response)
            
        except Exception as e:
            raise CommonException(f"Failed to stream response: {e}", sys)


# Example usage and testing
def test_conversational_rag():
    """Test the conversational RAG system."""
    try:
        # Get configuration from environment
        SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
        SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        INDEX_NAME = "documents-index"
        
        # Validate environment variables
        required_vars = {
            "AZURE_SEARCH_ENDPOINT": SEARCH_ENDPOINT,
            "AZURE_SEARCH_KEY": SEARCH_KEY,
            "OPENAI_API_KEY": OPENAI_API_KEY
        }
        
        for var_name, var_value in required_vars.items():
            if not var_value:
                raise ValueError(f"{var_name} environment variable is required")
        
        print("Testing Conversational RAG System...")
        
        # Load config
        config = load_config()
        
        # Test 1: Initialize without retriever
        print("\nTesting initialization without retriever...")
        rag = ConversationalRAG(session_id="test_session_1", config=config)
        print(f"RAG initialized with session: {rag.session_id}")
        
        # Test 2: Load retriever
        print("\nLoading retriever...")
        rag.load_retriever(
            search_endpoint=SEARCH_ENDPOINT,
            search_key=SEARCH_KEY,
            index_name=INDEX_NAME,
            openai_api_key=OPENAI_API_KEY,
            config=config
        )
        print("Retriever loaded and chain built")
        
        # Test 3: Single query
        print("\nTesting single query...")
        response = rag.invoke("What is best guidelines for APIs?", language="English")
        print(f"Query: What is best guidelines for APIs?")
        print(f"Response: {response}")
    
        # Test 4: Different language
        # print("\nTesting different language...")
        # response3 = rag.invoke("What is API design principles?", language="French")
        # print(f"Query: What is API design principles? (in French)")
        # print(f"Response: {response3}")
        
        # Test 6: Chat history inspection
        # print("\nChat history:")
        # chat_history = rag.get_chat_history()
        # print(f"Total messages in history: {len(chat_history)}")
        # for i, msg in enumerate(chat_history[-4:]):  # Show last 4 messages
        #     msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        #     print(f"  {msg_type}: {msg.content[:100]}...")
        
        # Test 7: New session
        # print("\nTesting new session...")
        # rag2 = ConversationalRAG(session_id="test_session_2", config=config)
        # rag2.load_retriever(SEARCH_ENDPOINT, SEARCH_KEY, INDEX_NAME, OPENAI_API_KEY, config)
        
        # response4 = rag2.invoke("What is security guidelines?", language="English")
        # print(f"New session query: What is security guidelines?")
        # print(f"Response: {response4}")
        # print(f"New session history length: {len(rag2.get_chat_history())}")
        
        # print("\n🎉 All tests completed successfully!")
        
        # # Interactive mode
        # print("\n" + "="*60)
        # print("🎯 INTERACTIVE MODE")
        # print("Chat with your documents! (type 'quit' to exit)")
        # print("Commands: 'clear' to clear history, 'history' to see chat history")
        
        # while True:
        #     try:
        #         user_query = input(f"\n[{rag.session_id[:8]}] You: ").strip()
                
        #         if user_query.lower() == 'quit':
        #             break
        #         elif user_query.lower() == 'clear':
        #             rag.clear_chat_history()
        #             print("Chat history cleared!")
        #             continue
        #         elif user_query.lower() == 'history':
        #             history = rag.get_chat_history()
        #             print(f"📚 Chat History ({len(history)} messages):")
        #             for i, msg in enumerate(history[-6:]):  # Show last 6
        #                 msg_type = "You" if isinstance(msg, HumanMessage) else "AI"
        #                 print(f"  {msg_type}: {msg.content[:150]}...")
        #             continue
        #         elif not user_query:
        #             continue
                
        #         # Get AI response
        #         response = rag.invoke(user_query, language="English")
        #         print(f"AI: {response}")
                
        #     except KeyboardInterrupt:
        #         break
        #     except Exception as e:
        #         print(f"Error: {e}")
        
        # print("\nThanks for testing the Conversational RAG system!")
        
    except Exception as e:
        app_exc = CommonException(f"Conversational RAG test failed: {e}", sys)
        logger.error(str(app_exc))
        raise app_exc


if __name__ == "__main__":
    test_conversational_rag()