import os
import sys
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path for custom imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from logger.custom_logger import CustomLogger
from exception.custom_exception import CommonException

# Document processing
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Azure AI Search
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration
)
from azure.core.credentials import AzureKeyCredential

# Initialize custom logger
logger = CustomLogger().get_logger(__file__)


class AISearchManager:
    """Simple Azure AI Search manager for vector operations."""
    
    def __init__(self, search_endpoint: str, search_key: str, index_name: str = "documents-index"):
        """Initialize Azure AI Search connection."""
        try:
            self.search_endpoint = search_endpoint
            self.search_key = search_key
            self.index_name = index_name
            self.credential = AzureKeyCredential(search_key)
            
            # Initialize clients
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=index_name,
                credential=self.credential
            )
            self.index_client = SearchIndexClient(
                endpoint=search_endpoint,
                credential=self.credential
            )
            
            logger.info("AISearchManager initialized", 
                       endpoint=search_endpoint, 
                       index_name=index_name)
            
        except Exception as e:
            raise CommonException(f"Failed to initialize AISearchManager: {e}", sys)
    
    def index_exists(self) -> bool:
        """Check if search index exists."""
        try:
            self.index_client.get_index(self.index_name)
            logger.info("Search index exists", index_name=self.index_name)
            return True
        except Exception as e:
            logger.info("Search index does not exist", 
                       index_name=self.index_name, 
                       error=str(e))
            return False
    
    def create_index(self, vector_dimensions: int = 3072):  # Changed from 1536 to 3072
        """Create search index with vector support."""
        try:
            # Define the search index schema
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True, sortable=True),
                SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="file_path", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="file_type", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="content_vector", 
                           type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                           searchable=True, 
                           vector_search_dimensions=vector_dimensions,
                           vector_search_profile_name="myHnswProfile")
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="myHnsw"
                    )
                ]
            )
            
            # Create the search index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self.index_client.create_or_update_index(index)
            logger.info("Created search index successfully", 
                       index_name=self.index_name,
                       vector_dimensions=vector_dimensions)
            
        except Exception as e:
            raise CommonException(f"Failed to create search index: {e}", sys)
    
    def get_fingerprint(self, file_path: str) -> str:
        """Generate fingerprint for file to detect changes."""
        try:
            stat = os.stat(file_path)
            file_info = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            fingerprint = hashlib.md5(file_info.encode()).hexdigest()
            
            logger.info("Generated file fingerprint", 
                       file_path=file_path,
                       fingerprint=fingerprint[:8])  # Only log first 8 chars
            return fingerprint
            
        except Exception as e:
            raise CommonException(f"Failed to generate fingerprint for {file_path}: {e}", sys)
    
    def save_metadata(self, metadata: Dict[str, Any], metadata_file: str = "ingestion_metadata.json"):
        """Save ingestion metadata to track processed files."""
        try:
            os.makedirs(os.path.dirname(metadata_file) if os.path.dirname(metadata_file) else ".", exist_ok=True)
            
            # Load existing metadata
            existing_metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        existing_metadata = json.load(f)
                except Exception as e:
                    logger.warning("Could not load existing metadata", 
                                 metadata_file=metadata_file,
                                 error=str(e))
            
            # Update with new metadata
            existing_metadata.update(metadata)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
            
            logger.info("Saved metadata", 
                       metadata_file=metadata_file,
                       files_tracked=len(existing_metadata))
            
        except Exception as e:
            raise CommonException(f"Failed to save metadata: {e}", sys)
    
    def load_metadata(self, metadata_file: str = "ingestion_metadata.json") -> Dict[str, Any]:
        """Load ingestion metadata."""
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    logger.info("Loaded metadata", 
                               metadata_file=metadata_file,
                               files_tracked=len(metadata))
                    return metadata
            else:
                logger.info("No metadata file found, starting fresh", 
                           metadata_file=metadata_file)
                return {}
                
        except Exception as e:
            logger.warning("Could not load metadata, starting fresh", 
                          metadata_file=metadata_file,
                          error=str(e))
            return {}
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to search index."""
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        try:
            result = self.search_client.upload_documents(documents)
            
            # Count successes and failures
            successful = sum(1 for item in result if item.succeeded)
            failed = len(documents) - successful
            
            logger.info("Document upload completed", 
                       total_docs=len(documents),
                       successful=successful,
                       failed=failed)
            
            # Log any failures
            for item in result:
                if not item.succeeded:
                    logger.error("Document upload failed", 
                               document_key=item.key,
                               error=item.error_message)
                    
        except Exception as e:
            raise CommonException(f"Error uploading documents to search index: {e}", sys)
    
    def search(self, query: str, top: int = 5) -> List[Dict[str, Any]]:
        """Simple text search."""
        try:
            results = self.search_client.search(search_text=query, top=top)
            search_results = [dict(result) for result in results]
            
            logger.info("Search completed", 
                       query=query,
                       results_count=len(search_results))
            
            return search_results
            
        except Exception as e:
            raise CommonException(f"Search error: {e}", sys)


class ChatIngestor:
    """Document ingestion and processing for chat applications."""
    
    def __init__(self, ai_search_manager: AISearchManager, 
                 openai_api_key: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the chat ingestor."""
        try:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required")
                
            self.ai_search = ai_search_manager
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Text splitter configuration
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            
            # Supported file extensions
            self.supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
            
            logger.info("ChatIngestor initialized", 
                       chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap,
                       supported_extensions=list(self.supported_extensions))
            
        except Exception as e:
            raise CommonException(f"Failed to initialize ChatIngestor: {e}", sys)
    
    def load_document(self, file_path: str) -> List[Any]:
        """Load document based on file extension."""
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension == '.txt':
                loader = TextLoader(str(file_path))
            elif extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif extension == '.docx':
                loader = Docx2txtLoader(str(file_path))
            elif extension == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
            
            documents = loader.load()
            
            logger.info("Document loaded successfully", 
                       file_path=str(file_path),
                       pages_loaded=len(documents),
                       file_type=extension)
            
            return documents
            
        except Exception as e:
            raise CommonException(f"Error loading document {file_path}: {e}", sys)
    
    def process_files(self, file_paths: List[str], force_reprocess: bool = False) -> int:
        """Process multiple files and add to search index."""
        try:
            metadata = self.ai_search.load_metadata()
            total_chunks = 0
            processed_files = 0
            skipped_files = 0
            
            logger.info("Starting file processing", 
                       total_files=len(file_paths),
                       force_reprocess=force_reprocess)
            
            for file_path in file_paths:
                try:
                    # Check if file needs processing
                    current_fingerprint = self.ai_search.get_fingerprint(file_path)
                    stored_fingerprint = metadata.get(file_path, {}).get('fingerprint')
                    
                    if not force_reprocess and current_fingerprint == stored_fingerprint:
                        logger.info("Skipping file - already processed", file_path=file_path)
                        skipped_files += 1
                        continue
                    
                    # Load and process document
                    documents = self.load_document(file_path)
                    chunks = self.text_splitter.split_documents(documents)
                    
                    if not chunks:
                        logger.warning("No content extracted from file", file_path=file_path)
                        continue
                    
                    # Create search documents
                    search_documents = []
                    for i, chunk in enumerate(chunks):
                        # Generate embedding
                        embedding = self.embeddings.embed_query(chunk.page_content)
                        
                        # Create document for search index
                        doc_id = f"{Path(file_path).stem}_{i}"
                        search_doc = {
                            "id": doc_id,
                            "content": chunk.page_content,
                            "file_path": str(file_path),
                            "file_type": Path(file_path).suffix.lower(),
                            "chunk_index": i,
                            "content_vector": embedding
                        }
                        search_documents.append(search_doc)
                    
                    # Upload to search index
                    if search_documents:
                        self.ai_search.add_documents(search_documents)
                        total_chunks += len(search_documents)
                        processed_files += 1
                        
                        # Update metadata
                        metadata[file_path] = {
                            'fingerprint': current_fingerprint,
                            'processed_at': str(Path(file_path).stat().st_mtime),
                            'chunks_count': len(search_documents)
                        }
                        self.ai_search.save_metadata(metadata)
                        
                        logger.info("File processed successfully", 
                                   file_path=file_path,
                                   chunks_created=len(search_documents))
                    
                except Exception as e:
                    logger.error("Failed to process individual file", 
                               file_path=file_path,
                               error=str(e))
                    continue
            
            logger.info("File processing completed", 
                       total_files=len(file_paths),
                       processed_files=processed_files,
                       skipped_files=skipped_files,
                       total_chunks=total_chunks)
            
            return total_chunks
            
        except Exception as e:
            raise CommonException(f"Error during file processing: {e}", sys)
    
    def build_retriever(self, data_folder: str) -> SearchClient:
        """Build retriever by processing all supported files in folder."""
        try:
            # Create index if it doesn't exist
            if not self.ai_search.index_exists():
                logger.info("Creating new search index")
                self.ai_search.create_index()
            
            # Find all supported files
            data_path = Path(data_folder)
            if not data_path.exists():
                raise ValueError(f"Data folder does not exist: {data_folder}")
            
            file_paths = []
            for extension in self.supported_extensions:
                found_files = list(data_path.glob(f"**/*{extension}"))
                file_paths.extend(found_files)
                logger.info("Found files by extension", 
                           extension=extension,
                           count=len(found_files))
            
            if not file_paths:
                logger.warning("No supported files found", data_folder=data_folder)
                return self.ai_search.search_client
            
            # Process files
            file_paths_str = [str(fp) for fp in file_paths]
            total_chunks = self.process_files(file_paths_str)
            
            logger.info("Retriever build completed", 
                       data_folder=data_folder,
                       total_files=len(file_paths),
                       total_chunks=total_chunks)
            
            # Return search client for retrieval
            return self.ai_search.search_client
            
        except Exception as e:
            raise CommonException(f"Failed to build retriever: {e}", sys)


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Example usage
if __name__ == "__main__":
    try:
        # Configuration from environment variables
        SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
        SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY") 
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        INDEX_NAME = "documents-index"
        DATA_FOLDER = "data"  # Folder containing your documents
        
        # Validate required environment variables
        if not SEARCH_ENDPOINT:
            raise ValueError("AZURE_SEARCH_ENDPOINT environment variable is required")
        if not SEARCH_KEY:
            raise ValueError("AZURE_SEARCH_KEY environment variable is required")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize components
        ai_search_manager = AISearchManager(
            search_endpoint=SEARCH_ENDPOINT,
            search_key=SEARCH_KEY,
            index_name=INDEX_NAME
        )
        
        chat_ingestor = ChatIngestor(
            ai_search_manager=ai_search_manager,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Build retriever (this processes all files and creates vector store)
        search_client = chat_ingestor.build_retriever(DATA_FOLDER)
        
        logger.info("Data ingestion pipeline completed successfully")
        print("Data ingestion complete!")
        print(f"Search client ready: {search_client}")
        
    except Exception as e:
        app_exc = CommonException(f"Pipeline execution failed: {e}", sys)
        logger.error(str(app_exc))
        raise app_exc