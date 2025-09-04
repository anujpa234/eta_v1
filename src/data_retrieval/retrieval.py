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
        """Perform hybrid search with CORRECTED relevance thresholds."""
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
            
            # Perform hybrid search
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
                select=["id", "content", "file_path", "file_type", "chunk_index"]
            )
            
            # Convert results to list of documents
            documents = []
            all_scores = []
            
            for result in results:
                search_score = result.get("@search.score", 0.0)
                all_scores.append(search_score)
                
                documents.append({
                    "page_content": result.get("content", ""),
                    "metadata": {
                        "id": result.get("id", ""),
                        "file_path": result.get("file_path", ""),
                        "file_type": result.get("file_type", ""),
                        "chunk_index": result.get("chunk_index", 0),
                        "score": search_score
                    }
                })
            
            if not documents:
                logger.info("No search results returned", query=query)
                return []
            
            # OPTION 1: Use much lower threshold (RECOMMENDED)
            RELEVANCE_THRESHOLD = 0.01  # Much more realistic for Azure Search
            
            relevant_docs = []
            for doc in documents:
                score = doc["metadata"]["score"]
                if score >= RELEVANCE_THRESHOLD:
                    relevant_docs.append(doc)
          
            # Debug output
            print(f" SEARCH DEBUG:")
            print(f"  Query: {query}")
            print(f"  Total results: {len(documents)}")
            if all_scores:
                print(f"  Score range: {min(all_scores):.4f} - {max(all_scores):.4f}")
                print(f"  Threshold used: {RELEVANCE_THRESHOLD}")
            print(f"  Relevant docs: {len(relevant_docs)}")
            
            if relevant_docs:
                logger.info("Relevant documents found", 
                        query=query,
                        total_results=len(documents),
                        relevant_results=len(relevant_docs),
                        best_score=max([doc["metadata"]["score"] for doc in relevant_docs]))
            else:
                logger.info("No relevant documents found", query=query)
                
                # Emergency fallback: if nothing passes threshold, keep the best result
                if documents:
                    best_doc = max(documents, key=lambda x: x["metadata"]["score"])
                    print(f" FALLBACK: Keeping best result with score {best_doc['metadata']['score']:.4f}")
                    return [best_doc]
            
            return relevant_docs
        
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
                if not docs:
                    return ""
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



# Add this method to your ConversationalRAG class (inside the class definition)

def debug_full_pipeline(self, user_input: str, chat_history: List[BaseMessage] = None, language: str = "English"):
    """Debug the entire RAG pipeline step by step."""
    
    print(f"\n{'='*80}")
    print(f"🔍 DEBUGGING FULL RAG PIPELINE")
    print(f"{'='*80}")
    print(f"Input Query: '{user_input}'")
    print(f"Language: {language}")
    
    if chat_history is None:
        chat_history = self.get_chat_history()
    print(f"Chat History Length: {len(chat_history)}")
    
    try:
        # STEP 1: Question Contextualization
        print(f"\n📝 STEP 1: Question Contextualization")
        print("-" * 50)
        
        contextualize_input = {
            "input": user_input,
            "chat_history": chat_history,
            "language": language
        }
        
        contextualized_question = (
            self.contextualize_prompt | self.llm | StrOutputParser()
        ).invoke(contextualize_input)
        
        print(f"Original Question: {user_input}")
        print(f"Contextualized Question: {contextualized_question}")
        
        # STEP 2: Document Retrieval  
        print(f"\n🔍 STEP 2: Document Retrieval")
        print("-" * 50)
        
        if not self.retriever:
            print("❌ ERROR: No retriever found!")
            return
            
        # Test direct retrieval
        docs = self.retriever.search(contextualized_question, top_k=5)
        
        print(f"Documents Retrieved: {len(docs)}")
        
        if not docs:
            print("❌ NO DOCUMENTS RETRIEVED!")
            print("Testing raw Azure Search...")
            
            # Try raw search
            try:
                query_embedding = self.retriever.embeddings.embed_query(contextualized_question)
                vector_query = VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=5,
                    fields="content_vector"
                )
                
                raw_results = self.retriever.search_client.search(
                    search_text=contextualized_question,
                    vector_queries=[vector_query],
                    top=5,
                    select=["id", "content", "file_path", "file_type", "chunk_index"]
                )
                
                raw_docs = list(raw_results)
                print(f"Raw search returned: {len(raw_docs)} documents")
                
                if raw_docs:
                    print("Raw results with scores:")
                    for i, result in enumerate(raw_docs[:3]):
                        score = result.get("@search.score", 0.0)
                        content = result.get("content", "")[:100]
                        print(f"  {i+1}. Score: {score:.4f} | Content: {content}...")
                else:
                    print("❌ Even raw search returned no results - INDEX IS EMPTY!")
            except Exception as e:
                print(f"❌ Raw search failed: {e}")
            
            return
        
        # Show retrieved documents
        for i, doc in enumerate(docs):
            score = doc["metadata"].get("score", 0.0)
            content = doc["page_content"][:150]
            file_path = doc["metadata"].get("file_path", "Unknown")
            
            print(f"Doc {i+1}:")
            print(f"  Score: {score:.4f}")
            print(f"  Source: {file_path}")
            print(f"  Content: {content}...")
            print()
        
        # STEP 3: Context Formatting
        print(f"📄 STEP 3: Context Formatting")
        print("-" * 50)
        
        formatted_context = self._format_docs(docs)
        print(f"Formatted Context Length: {len(formatted_context)} characters")
        print(f"Context Preview (first 300 chars):")
        print(f"'{formatted_context[:300]}...'")
        
        if not formatted_context.strip():
            print("❌ FORMATTED CONTEXT IS EMPTY!")
            return
        
        # STEP 4: Answer Generation
        print(f"\n🤖 STEP 4: Answer Generation")
        print("-" * 50)
        
        qa_input = {
            "context": formatted_context,
            "input": user_input,
            "chat_history": chat_history,
            "language": language
        }
        
        answer = (self.qa_prompt | self.llm | StrOutputParser()).invoke(qa_input)
        
        print(f"Generated Answer: '{answer}'")
        
        is_no_answer = "don't have enough relevant information" in answer.lower()
        print(f"Is 'I don't know' response: {is_no_answer}")
        
    except Exception as e:
        print(f"❌ ERROR during debugging: {e}")
        import traceback
        traceback.print_exc()
def debug_rag_pipeline():
    """Standalone debug function - run this immediately."""
    
    print("🔍 STANDALONE RAG PIPELINE DEBUG")
    print("=" * 60)
    
    try:
        # Initialize your RAG system
        print("1. Initializing RAG system...")
        rag = ConversationalRAG(session_id="debug_session")
        
        print("2. Loading retriever...")
        rag.load_retriever(
            search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            search_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name="documents-index",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        query = "What are API design best practices?"
        print(f"3. Testing query: '{query}'")
        
        # TEST 1: Check if Azure Search has any data
        print("\n🔍 TEST 1: Checking Azure Search Index")
        print("-" * 40)
        
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="documents-index",
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
        
        # Check total documents
        all_docs = list(search_client.search("*", top=5))
        print(f"Total docs in index: {len(all_docs)}")
        
        if all_docs:
            print("✅ Index has data!")
            sample_doc = all_docs[0]
            print(f"Sample document keys: {list(sample_doc.keys())}")
            content = sample_doc.get('content', 'No content field')
            print(f"Sample content: {str(content)[:100]}...")
        else:
            print("❌ INDEX IS EMPTY! This is your main problem.")
            print("You need to run your data ingestion pipeline first.")
            return
        
        # TEST 2: Test direct search
        print("\n🔍 TEST 2: Testing Direct Search")
        print("-" * 40)
        
        search_results = list(search_client.search(query, top=3))
        print(f"Direct search results: {len(search_results)}")
        
        if search_results:
            for i, result in enumerate(search_results):
                score = result.get("@search.score", 0.0)
                content = str(result.get("content", ""))[:100]
                print(f"Result {i+1}: Score={score:.4f}, Content={content}...")
        
        # TEST 3: Test your retriever
        print("\n🔍 TEST 3: Testing Your Retriever")
        print("-" * 40)
        
        retrieved_docs = rag.retriever.search(query, top_k=3)
        print(f"Your retriever returned: {len(retrieved_docs)} documents")
        
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                score = doc["metadata"].get("score", 0.0)
                content = doc["page_content"][:100]
                print(f"Doc {i+1}: Score={score:.4f}, Content={content}...")
        else:
            print("❌ YOUR RETRIEVER RETURNED NOTHING!")
            print("This means your relevance filtering is too strict.")
        
        # TEST 4: Test context formatting
        print("\n📄 TEST 4: Testing Context Formatting")
        print("-" * 40)
        
        if retrieved_docs:
            context = rag._format_docs(retrieved_docs)
            print(f"Formatted context length: {len(context)}")
            print(f"Context preview: {context[:200]}...")
            
            if not context.strip():
                print("❌ FORMATTED CONTEXT IS EMPTY!")
            else:
                print("✅ Context formatted successfully")
        
        # TEST 5: Test full pipeline
        print("\n🤖 TEST 5: Testing Full Pipeline")
        print("-" * 40)
        
        try:
            response = rag.invoke(query)
            print(f"Final response: {response}")
            
            is_no_answer = "don't have enough relevant information" in response.lower()
            print(f"Is 'I don't know' response: {is_no_answer}")
            
            if is_no_answer and retrieved_docs:
                print("❌ PROBLEM: Retrieved docs but still saying 'I don't know'")
                print("This suggests your QA prompt is too strict.")
            elif not is_no_answer:
                print("✅ SUCCESS: Generated actual answer!")
                
        except Exception as e:
            print(f"❌ Full pipeline failed: {e}")
        
        # SUMMARY
        print(f"\n📊 DEBUG SUMMARY")
        print("=" * 40)
        print(f"Index has data: {'✅' if all_docs else '❌'}")
        print(f"Direct search works: {'✅' if search_results else '❌'}")
        print(f"Retriever returns docs: {'✅' if retrieved_docs else '❌'}")
        print(f"Context formatting works: {'✅' if retrieved_docs and context.strip() else '❌'}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

# Run the debug
if __name__ == "__main__":
    debug_rag_pipeline()

# Or add this to your test function:
def test_conversational_rag():
    """Modified test function with debugging."""
    try:
        # Run debug first
        debug_rag_pipeline()
        
        # Then your regular tests...
        
    except Exception as e:
        print(f"Test failed: {e}")
            
# Run the debug
if __name__ == "__main__":
    debug_rag_pipeline()

# Or add this to your test function:
def test_conversational_rag():
    """Modified test function with debugging."""
    try:
        # Run debug first
        debug_rag_pipeline()
        
        # Then your regular tests...
        
    except Exception as e:
        print(f"Test failed: {e}")