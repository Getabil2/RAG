# import os
# import json
# import time
# import logging
# import random
# from datetime import datetime, timezone
# from typing import List, Dict, Optional
# import io
# import concurrent.futures
# from dotenv import load_dotenv
# # Azure imports
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient
# from azure.storage.blob import BlobServiceClient
# from openai import AzureOpenAI

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler("data_indexing.log")
#     ]
# )
# logger = logging.getLogger(__name__)

# class DataIndexer:
#     def __init__(self):
#         """Initialize the data indexer with optimized configurations."""
#         self._initialize_services()
#         self._setup_configurations()
#         self._validate_configurations()
        
#     def _initialize_services(self):
#         """Initialize all Azure services with optimized configurations."""
#         try:
#             # Blob Storage with optimized settings
#             self.blob_service = BlobServiceClient.from_connection_string(
#                 os.getenv("STORAGE_CONNECTION_STRING"),
#                 retry_total=3,
#                 retry_backoff_factor=0.5,
#                 max_single_get_size=4*1024*1024,
#                 connection_timeout=30
#             )
            
#             # Search client with optimized settings
#             self.search_client = SearchClient(
#                 endpoint=os.getenv("SEARCH_ENDPOINT"),
#                 index_name=os.getenv("INDEX_NAME", "vehicle-manuals"),
#                 credential=AzureKeyCredential(os.getenv("SEARCH_KEY")),
#                 timeout=60
#             )
            
#             # Azure OpenAI with optimized settings
#             self.openai_client = AzureOpenAI(
#                 api_key=os.getenv("AZURE_OPENAI_KEY"),
#                 api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
#                 azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#                 max_retries=3,
#                 timeout=30
#             )
            
#         except Exception as e:
#             logger.error(f"Service initialization failed: {str(e)}")
#             raise

#     def _setup_configurations(self):
#         """Set up optimized system configurations."""
#         self.container_name = os.getenv("CONTAINER_NAME", "vehicle-manuals")
#         self.processed_prefix = os.getenv("PROCESSED_PREFIX", "processed_data/")
#         self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
#         self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        
#         # Performance tuning parameters
#         self.max_workers = min(
#             int(os.getenv("MAX_WORKERS", min(8, (os.cpu_count() or 4)))),
#             (os.cpu_count() or 4) * 2
#         )
#         self.upload_batch_size = int(os.getenv("UPLOAD_BATCH_SIZE", 20))
#         self.max_upload_retries = int(os.getenv("MAX_UPLOAD_RETRIES", 3))
#         self.max_retry_delay = int(os.getenv("MAX_RETRY_DELAY", 60))
#         self.max_concurrent_embeddings = min(4, self.max_workers)
#         self.api_call_delay = float(os.getenv("API_CALL_DELAY", 0.1))
#         self.embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT", 30))

#     def _validate_configurations(self):
#         """Validate all configurations and environment variables."""
#         required_vars = [
#             "STORAGE_CONNECTION_STRING",
#             "SEARCH_ENDPOINT",
#             "SEARCH_KEY",
#             "AZURE_OPENAI_ENDPOINT",
#             "AZURE_OPENAI_KEY"
#         ]
        
#         missing_vars = [var for var in required_vars if not os.getenv(var)]
#         if missing_vars:
#             raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

#     def index_documents(self) -> int:
#         """Main document indexing pipeline with parallel processing."""
#         logger.info("Starting document indexing pipeline")
#         start_time = time.time()
        
#         try:
#             # Load processed documents from blob storage
#             processed_docs = self._load_processed_documents()
#             if not processed_docs:
#                 logger.warning("No processed documents found to index")
#                 return 0
            
#             # Index documents with optimized batching
#             indexed_count = self._index_documents_optimized(processed_docs)
            
#             # Final statistics
#             total_time = time.time() - start_time
#             logger.info(
#                 f"Indexing completed in {total_time:.2f} seconds\n"
#                 f"Documents indexed: {indexed_count}/{len(processed_docs)}"
#             )
            
#             return indexed_count
            
#         except Exception as e:
#             logger.error(f"Indexing pipeline failed: {str(e)}")
#             raise

#     def _load_processed_documents(self) -> List[Dict]:
#         """Load processed documents from blob storage."""
#         processed_docs = []
#         container_client = self.blob_service.get_container_client(self.container_name)
        
#         try:
#             # List and load all processed documents
#             blobs = container_client.list_blobs(name_starts_with=self.processed_prefix)
#             for blob in blobs:
#                 if blob.name.endswith('.json'):
#                     try:
#                         blob_client = container_client.get_blob_client(blob.name)
#                         data = blob_client.download_blob().readall()
#                         document = json.loads(data.decode('utf-8'))
                        
#                         # Transform document to match index schema
#                         document = self._transform_document_structure(document)
                        
#                         if not self._validate_document_structure(document):
#                             logger.error(f"Invalid document structure in {blob.name}")
#                             continue
                            
#                         processed_docs.append(document)
#                         logger.info(f"Loaded processed document: {blob.name}")
#                     except Exception as e:
#                         logger.error(f"Failed to load {blob.name}: {str(e)}")
#                         continue
                        
#             return processed_docs
            
#         except Exception as e:
#             logger.error(f"Failed to load processed documents: {str(e)}")
#             raise

#     def _transform_document_structure(self, document: Dict) -> Dict:
#         """Transform document structure to match the index schema."""
#         try:
#             # Ensure all chunks have required fields
#             for chunk in document.get("content_chunks", []):
#                 if "id" not in chunk:
#                     chunk["id"] = f"{document['id']}-{chunk.get('chunk_index', 0)}"
                
#                 # Ensure all required fields exist with defaults
#                 chunk.setdefault("text", "")
#                 chunk.setdefault("chunk_index", 0)
#                 chunk.setdefault("page_number", 1)
#                 chunk.setdefault("word_count", 0)
            
#             # Transform tables
#             for table in document.get("tables", []):
#                 if "id" not in table:
#                     table["id"] = f"table-{document['id']}-{table.get('page', 1)}-{table.get('table_index', 0)}"
#                 table.setdefault("caption", "")
#                 table.setdefault("content_markdown", "")
#                 table.setdefault("content", "")
#                 table.setdefault("row_count", 0)
#                 table.setdefault("column_count", 0)
#                 table.setdefault("image_url", "")
#                 table.setdefault("bounding_box", "")
#                 table.setdefault("chunk_references", [])
#                 table.setdefault("related_content", {
#                     "chunk_ids": [],
#                     "image_ids": [],
#                     "warning_ids": []
#                 })
            
#             # Transform images
#             for image in document.get("images", []):
#                 if "id" not in image:
#                     image["id"] = f"img-{document['id']}-{image.get('page', 1)}-{image.get('image_index', 0)}"
#                 image.setdefault("caption", "")
#                 image.setdefault("dimensions", "0x0")
#                 image.setdefault("size_kb", 0)
#                 image.setdefault("bounding_box", "")
#                 image.setdefault("chunk_references", [])
#                 image.setdefault("related_content", {
#                     "chunk_ids": [],
#                     "table_ids": [],
#                     "warning_ids": []
#                 })
#                 image.setdefault("analysis", {
#                     "tags": [],
#                     "description": ""
#                 })
            
#             # Transform warnings
#             for warning in document.get("warnings", []):
#                 if "id" not in warning:
#                     warning["id"] = f"warn-{document['id']}-{warning.get('page', 1)}-{warning.get('warning_index', 0)}"
#                 warning.setdefault("text", "")
#                 warning.setdefault("severity", "medium")
#                 warning.setdefault("context", "")
#                 warning.setdefault("chunk_references", [])
            
#             # Transform relationships
#             for rel in document.get("relationships", []):
#                 rel.setdefault("content_type", "reference")
#                 rel.setdefault("content_id", "")
#                 rel.setdefault("content_summary", "")
#                 rel.setdefault("target_page", 0)
#                 rel.setdefault("relationship_type", "reference")
#                 rel.setdefault("confidence_score", 0.0)
            
#             # Ensure metadata has all required fields
#             document.setdefault("metadata", {})
#             document["metadata"].setdefault("processing_date", datetime.now(timezone.utc).isoformat())
#             document["metadata"].setdefault("page_count", 1)
#             document["metadata"].setdefault("word_count", 0)
#             document["metadata"].setdefault("document_type", "manual")
#             document["metadata"].setdefault("total_chunks", len(document.get("content_chunks", [])))
#             document["metadata"].setdefault("table_count", len(document.get("tables", [])))
#             document["metadata"].setdefault("image_count", len(document.get("images", [])))
            
#             return document
            
#         except Exception as e:
#             logger.error(f"Error transforming document {document.get('id', 'unknown')}: {str(e)}")
#             raise

#     def _validate_document_structure(self, document: Dict) -> bool:
#         """Validate the document structure has required fields."""
#         required_fields = [
#             'id', 'title', 'source_file', 'content_chunks',
#             'metadata', 'tables', 'images', 'warnings'
#         ]
        
#         for field in required_fields:
#             if field not in document:
#                 logger.error(f"Missing required field: {field}")
#                 return False
                
#             if field == 'content_chunks' and not isinstance(document[field], list):
#                 logger.error("content_chunks must be a list")
#                 return False
                
#             if field == 'metadata' and 'processing_date' not in document[field]:
#                 logger.error("metadata missing processing_date")
#                 return False
                
#         # Validate each chunk has required fields
#         for i, chunk in enumerate(document.get('content_chunks', [])):
#             if not all(k in chunk for k in ['id', 'text', 'chunk_index', 'page_number', 'word_count']):
#                 logger.error(f"Chunk {i} missing required fields in document {document.get('id', 'unknown')}")
#                 return False
                
#         return True

#     def _index_documents_optimized(self, documents: List[Dict]) -> int:
#         """Optimized document indexing with parallel preparation and bulk operations."""
#         logger.info(f"Starting optimized indexing of {len(documents)} documents")
#         start_time = time.time()
#         total_indexed = 0
        
#         # Prepare all search documents in parallel
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             search_docs = []
#             futures = []
            
#             for doc in documents:
#                 futures.append(executor.submit(
#                     self._prepare_search_documents,
#                     doc
#                 ))
                
#                 # Control memory usage
#                 if len(futures) >= self.max_workers * 2:
#                     for future in concurrent.futures.as_completed(futures):
#                         try:
#                             search_docs.extend(future.result())
#                         except Exception as e:
#                             logger.error(f"Document preparation failed: {str(e)}")
#                     futures = []
            
#             # Process remaining futures
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     search_docs.extend(future.result())
#                 except Exception as e:
#                     logger.error(f"Document preparation failed: {str(e)}")
        
#         # Index in optimized batches with dynamic throttling
#         batch_size = self.upload_batch_size
#         for i in range(0, len(search_docs), batch_size):
#             batch = search_docs[i:i + batch_size]
            
#             # Index batch with retry
#             indexed = self._index_batch_with_retry(batch)
#             total_indexed += indexed
            
#             # Dynamic throttling based on success rate
#             success_rate = indexed / len(batch) if len(batch) > 0 else 1.0
#             if success_rate < 0.8:
#                 delay = min(5, 2 ** (1 - success_rate))  # Exponential backoff
#                 logger.warning(f"Batch success rate {success_rate:.1%}, delaying {delay:.1f}s")
#                 time.sleep(delay)
#             elif self.api_call_delay > 0:
#                 time.sleep(self.api_call_delay)
        
#         logger.info(f"Indexing completed. {total_indexed} documents indexed in {time.time()-start_time:.2f}s")
#         return total_indexed

#     def _prepare_search_documents(self, document: Dict) -> List[Dict]:
#         """Prepare document chunks for indexing with parallel embedding generation."""
#         search_docs = []
        
#         # Generate embeddings in parallel with controlled concurrency
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_embeddings) as executor:
#             future_to_chunk = {}
            
#             for chunk in document.get("content_chunks", []):
#                 if not all(k in chunk for k in ['id', 'text', 'chunk_index', 'page_number', 'word_count']):
#                     logger.warning(f"Skipping invalid chunk in document {document['id']}")
#                     continue
                    
#                 future = executor.submit(
#                     self._generate_embeddings_with_retry, 
#                     chunk["text"]
#                 )
#                 future_to_chunk[future] = chunk
            
#             for future in concurrent.futures.as_completed(future_to_chunk):
#                 chunk = future_to_chunk[future]
#                 try:
#                     embeddings = future.result()
                    
#                     # Build page_elements structure
#                     page_elements = {
#                         "page_number": chunk["page_number"],
#                         "chunks": [{
#                             "id": ch["id"],
#                             "text": ch["text"]
#                         } for ch in document.get("content_chunks", []) 
#                         if ch.get("page_number") == chunk["page_number"]],
#                         "tables": [{
#                             "id": t["id"],
#                             "image_url": t.get("image_url")
#                         } for t in document.get("tables", []) 
#                         if t.get("page") == chunk["page_number"]],
#                         "images": [{
#                             "id": i["id"],
#                             "url": i.get("url")
#                         } for i in document.get("images", []) 
#                         if i.get("page") == chunk["page_number"]],
#                         "warnings": [{
#                             "id": w["id"],
#                             "text": w.get("text", "")
#                         } for w in document.get("warnings", []) 
#                         if w.get("page") == chunk["page_number"]]
#                     }
                    
#                     # Create the search document
#                     search_doc = {
#                         "id": chunk["id"],
#                         "title": document["title"],
#                         "content": chunk["text"],
#                         "source_file": document["source_file"],
#                         "content_vector": embeddings,
#                         "metadata": {
#                             "processing_date": document["metadata"]["processing_date"],
#                             "page_count": document["metadata"]["page_count"],
#                             "word_count": chunk["word_count"],
#                             "document_type": document["metadata"].get("document_type", "manual"),
#                             "chunk_index": chunk["chunk_index"],
#                             "total_chunks": document["metadata"].get("total_chunks", len(document["content_chunks"])),
#                             "page_number": chunk["page_number"],
#                             "table_count": document["metadata"].get("table_count", 0),
#                             "image_count": document["metadata"].get("image_count", 0)
#                         },
#                         "warnings": [{
#                             "text": w.get("text", ""),
#                             "severity": w.get("severity", "medium"),
#                             "context": w.get("context", ""),
#                             "page": w["page"],
#                             "chunk_references": [chunk["id"]]
#                         } for w in document.get("warnings", []) 
#                         if w.get("page") == chunk["page_number"]],
#                         "tables": [{
#                             "caption": t.get("caption", f"Table on page {t['page']}"),
#                             "content_markdown": t.get("content_markdown", ""),
#                             "page": t["page"],
#                             "content": t.get("content", ""),
#                             "row_count": t.get("row_count", 0),
#                             "column_count": t.get("column_count", 0),
#                             "image_url": t.get("image_url"),
#                             "bounding_box": t.get("bounding_box", ""),
#                             "chunk_references": [chunk["id"]],
#                             "related_content": {
#                                 "chunk_ids": [ch["id"] for ch in document.get("content_chunks", []) 
#                                 if ch.get("page_number") == t["page"]],
#                                 "image_ids": [img["id"] for img in document.get("images", []) 
#                                 if img.get("page") == t["page"]],
#                                 "warning_ids": [w["id"] for w in document.get("warnings", []) 
#                                 if w.get("page") == t["page"]]
#                             }
#                         } for t in document.get("tables", []) 
#                         if t.get("page") == chunk["page_number"]],
#                         "images": [{
#                             "url": i.get("url", ""),
#                             "caption": i.get("caption", f"Image on page {i['page']}"),
#                             "page": i["page"],
#                             "dimensions": i.get("dimensions", "0x0"),
#                             "size_kb": i.get("size_kb", 0),
#                             "bounding_box": i.get("bounding_box", ""),
#                             "chunk_references": [chunk["id"]],
#                             "related_content": {
#                                 "chunk_ids": [ch["id"] for ch in document.get("content_chunks", []) 
#                                 if ch.get("page_number") == i["page"]],
#                                 "table_ids": [t["id"] for t in document.get("tables", []) 
#                                 if t.get("page") == i["page"]],
#                                 "warning_ids": [w["id"] for w in document.get("warnings", []) 
#                                 if w.get("page") == i["page"]]
#                             },
#                             "analysis": {
#                                 "tags": i.get("tags", []),
#                                 "description": i.get("caption", "")
#                             }
#                         } for i in document.get("images", []) 
#                         if i.get("page") == chunk["page_number"]],
#                         "relationships": [{
#                             "content_type": rel.get("content_type", "reference"),
#                             "content_id": rel.get("content_id", ""),
#                             "content_summary": rel.get("content_summary", ""),
#                             "source_chunk_index": rel.get("source_chunk_index", 0),
#                             "target_page": rel.get("target_page", 0),
#                             "relationship_type": rel.get("relationship_type", "reference"),
#                             "confidence_score": rel.get("confidence_score", 0.0)
#                         } for rel in document.get("relationships", []) 
#                         if rel.get("source_chunk_index") == chunk["chunk_index"]],
#                         "page_elements": page_elements
#                     }
                    
#                     search_docs.append(search_doc)
                    
#                 except Exception as e:
#                     logger.warning(f"Failed to prepare chunk {chunk.get('chunk_index', 'unknown')}: {str(e)}")
#                     continue
        
#         return search_docs

#     def _generate_embeddings_with_retry(self, text: str) -> List[float]:
#         """Generate embeddings with retry logic and validation."""
#         if not text.strip():
#             return [0.0] * self.embedding_dimensions
            
#         for attempt in range(3):
#             try:
#                 start_time = time.time()
#                 response = self.openai_client.embeddings.create(
#                     input=text,
#                     model=self.embedding_deployment,
#                     timeout=self.embedding_timeout
#                 )
#                 embeddings = response.data[0].embedding
                
#                 if len(embeddings) != self.embedding_dimensions:
#                     raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimensions}, got {len(embeddings)}")
                
#                 logger.debug(f"Generated embeddings in {time.time()-start_time:.2f}s")
#                 return embeddings
                
#             except Exception as e:
#                 if attempt == 2:
#                     logger.error(f"Failed to generate embeddings after 3 attempts: {str(e)}")
#                     return [0.0] * self.embedding_dimensions
                
#                 wait_time = min((2 ** attempt) + random.random(), self.max_retry_delay)
#                 logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {str(e)}")
#                 time.sleep(wait_time)

#     def _index_batch_with_retry(self, batch: List[Dict]) -> int:
#         """Index a batch of documents with exponential backoff retry logic."""
#         if not batch:
#             return 0
            
#         for attempt in range(self.max_upload_retries):
#             try:
#                 result = self.search_client.upload_documents(batch)
#                 succeeded = sum(1 for r in result if r.succeeded)
                
#                 if succeeded < len(batch):
#                     failed = [r.key for r in result if not r.succeeded]
#                     logger.warning(f"Batch partial success: {succeeded}/{len(batch)}. Failed: {failed}")
#                 return succeeded
#             except Exception as e:
#                 if attempt == self.max_upload_retries - 1:
#                     logger.error(f"Final attempt failed for batch: {str(e)}")
#                     return 0
                
#                 wait_time = min((2 ** attempt) + random.random(), self.max_retry_delay)
#                 logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {str(e)}")
#                 time.sleep(wait_time)
        
#         return 0

# if __name__ == "__main__":
#     try:
#         indexer = DataIndexer()
#         indexed_count = indexer.index_documents()
#         logger.info(f"Indexed {indexed_count} documents successfully")
#     except Exception as e:
#         logger.error(f"Data indexing failed: {str(e)}")
#         raise


import os
import json
import time
import logging
import random
from datetime import datetime, timezone
from typing import List, Dict, Optional
import io
import concurrent.futures
from dotenv import load_dotenv

# Azure imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_indexing.log")
    ]
)
logger = logging.getLogger(__name__)

class DataIndexer:
    def __init__(self):
        """Initialize the data indexer with optimized configurations."""
        self._initialize_services()
        self._setup_configurations()
        self._validate_configurations()
        
    def _initialize_services(self):
        """Initialize all Azure services with optimized configurations."""
        try:
            # Blob Storage with optimized settings
            self.blob_service = BlobServiceClient.from_connection_string(
                os.getenv("STORAGE_CONNECTION_STRING"),
                retry_total=3,
                retry_backoff_factor=0.5,
                max_single_get_size=4*1024*1024,
                connection_timeout=30
            )
            
            # Search client with optimized settings
            self.search_client = SearchClient(
                endpoint=os.getenv("SEARCH_ENDPOINT"),
                index_name=os.getenv("INDEX_NAME", "vehicle-manuals"),
                credential=AzureKeyCredential(os.getenv("SEARCH_KEY")),
                timeout=60
            )
            
            # Azure OpenAI with optimized settings
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                max_retries=3,
                timeout=30
            )
            
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            raise

    def _setup_configurations(self):
        """Set up optimized system configurations."""
        self.container_name = os.getenv("CONTAINER_NAME", "vehicle-manuals")
        self.processed_prefix = os.getenv("PROCESSED_PREFIX", "processed_data/")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        
        # Performance tuning parameters
        self.max_workers = min(
            int(os.getenv("MAX_WORKERS", min(8, (os.cpu_count() or 4)))),
            (os.cpu_count() or 4) * 2
        )
        self.upload_batch_size = int(os.getenv("UPLOAD_BATCH_SIZE", 20))
        self.max_upload_retries = int(os.getenv("MAX_UPLOAD_RETRIES", 3))
        self.max_retry_delay = int(os.getenv("MAX_RETRY_DELAY", 60))
        self.max_concurrent_embeddings = min(4, self.max_workers)
        self.api_call_delay = float(os.getenv("API_CALL_DELAY", 0.1))
        self.embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT", 30))

    def _validate_configurations(self):
        """Validate all configurations and environment variables."""
        required_vars = [
            "STORAGE_CONNECTION_STRING",
            "SEARCH_ENDPOINT",
            "SEARCH_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def index_documents(self) -> int:
        """Main document indexing pipeline with parallel processing."""
        logger.info("Starting document indexing pipeline")
        start_time = time.time()
        
        try:
            # Load processed documents from blob storage
            processed_docs = self._load_processed_documents()
            if not processed_docs:
                logger.warning("No processed documents found to index")
                return 0
            
            # Index documents with optimized batching
            indexed_count = self._index_documents_optimized(processed_docs)
            
            # Final statistics
            total_time = time.time() - start_time
            logger.info(
                f"Indexing completed in {total_time:.2f} seconds\n"
                f"Documents indexed: {indexed_count}/{len(processed_docs)}"
            )
            
            return indexed_count
            
        except Exception as e:
            logger.error(f"Indexing pipeline failed: {str(e)}")
            raise

    def _load_processed_documents(self) -> List[Dict]:
        """Load processed documents from blob storage."""
        processed_docs = []
        container_client = self.blob_service.get_container_client(self.container_name)
        
        try:
            # List and load all processed documents
            blobs = container_client.list_blobs(name_starts_with=self.processed_prefix)
            for blob in blobs:
                if blob.name.endswith('.json'):
                    try:
                        blob_client = container_client.get_blob_client(blob.name)
                        data = blob_client.download_blob().readall()
                        document = json.loads(data.decode('utf-8'))
                        
                        # Validate document structure including page_elements
                        if not self._validate_document_structure(document):
                            logger.error(f"Invalid document structure in {blob.name}")
                            continue
                            
                        processed_docs.append(document)
                        logger.info(f"Loaded processed document: {blob.name}")
                    except Exception as e:
                        logger.error(f"Failed to load {blob.name}: {str(e)}")
                        continue
                        
            return processed_docs
            
        except Exception as e:
            logger.error(f"Failed to load processed documents: {str(e)}")
            raise

    def _validate_document_structure(self, document: Dict) -> bool:
        """Validate the document structure has required fields."""
        required_fields = [
            'id', 'title', 'source_file', 'content_chunks',
            'metadata', 'tables', 'images', 'warnings'
        ]
        
        for field in required_fields:
            if field not in document:
                logger.error(f"Missing required field: {field}")
                return False
                
            if field == 'content_chunks' and not isinstance(document[field], list):
                logger.error("content_chunks must be a list")
                return False
                
            if field == 'metadata' and 'processing_date' not in document[field]:
                logger.error("metadata missing processing_date")
                return False
                
        # Additional validation for content_chunks structure
        for chunk in document.get('content_chunks', []):
            if not all(k in chunk for k in ['text', 'chunk_index', 'page_number', 'word_count']):
                logger.error("Invalid chunk structure - missing required fields")
                return False
                
        return True

    def _index_documents_optimized(self, documents: List[Dict]) -> int:
        """Optimized document indexing with parallel preparation and bulk operations."""
        logger.info(f"Starting optimized indexing of {len(documents)} documents")
        start_time = time.time()
        total_indexed = 0
        
        # Prepare all search documents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            search_docs = []
            futures = []
            
            for doc in documents:
                futures.append(executor.submit(
                    self._prepare_search_documents,
                    doc
                ))
                
                # Control memory usage
                if len(futures) >= self.max_workers * 2:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            prepared_docs = future.result()
                            # Validate prepared documents before adding to batch
                            valid_docs = [d for d in prepared_docs if self._validate_search_document(d)]
                            search_docs.extend(valid_docs)
                        except Exception as e:
                            logger.error(f"Document preparation failed: {str(e)}")
                    futures = []
            
            # Process remaining futures
            for future in concurrent.futures.as_completed(futures):
                try:
                    prepared_docs = future.result()
                    valid_docs = [d for d in prepared_docs if self._validate_search_document(d)]
                    search_docs.extend(valid_docs)
                except Exception as e:
                    logger.error(f"Document preparation failed: {str(e)}")
        
        # Index in optimized batches with dynamic throttling
        batch_size = self.upload_batch_size
        for i in range(0, len(search_docs), batch_size):
            batch = search_docs[i:i + batch_size]
            
            # Index batch with retry
            indexed = self._index_batch_with_retry(batch)
            total_indexed += indexed
            
            # Dynamic throttling based on success rate
            success_rate = indexed / len(batch) if len(batch) > 0 else 1.0
            if success_rate < 0.8:
                delay = min(5, 2 ** (1 - success_rate))  # Exponential backoff
                logger.warning(f"Batch success rate {success_rate:.1%}, delaying {delay:.1f}s")
                time.sleep(delay)
            elif self.api_call_delay > 0:
                time.sleep(self.api_call_delay)
        
        logger.info(f"Indexing completed. {total_indexed} documents indexed in {time.time()-start_time:.2f}s")
        return total_indexed

    def _validate_search_document(self, document: Dict) -> bool:
        """Validate a search document before indexing."""
        required_fields = ['id', 'title', 'content', 'source_file', 'content_vector', 'metadata', 'page_elements']
        
        for field in required_fields:
            if field not in document:
                logger.error(f"Search document missing required field: {field}")
                return False
                
        # Ensure page_elements is not null and has required structure
        if not isinstance(document['page_elements'], list):
            logger.error("page_elements must be a non-null list")
            return False
        if not document['page_elements']:
            logger.error("page_elements must not be empty")
            return False
        required_page_elements = ['page_number', 'chunks']
        for elem in document['page_elements']:
            for field in required_page_elements:
                if field not in elem:
                    logger.error(f"page_elements element missing required field: {field}")
                    return False

    def _prepare_search_documents(self, document: Dict) -> List[Dict]:
        """Prepare document chunks for indexing with parallel embedding generation."""
        search_docs = []

        # Generate embeddings in parallel with controlled concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_embeddings) as executor:
            future_to_chunk = {}

            # First validate all chunks have required fields
            for chunk in document.get("content_chunks", []):
                if not all(k in chunk for k in ['text', 'chunk_index', 'page_number', 'word_count']):
                    logger.warning(f"Skipping invalid chunk in document {document['id']}")
                    continue

                future = executor.submit(
                    self._generate_embeddings_with_retry,
                    chunk["text"]
                )
                future_to_chunk[future] = chunk

            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    embeddings = future.result()

                    # Always build page_elements as a list of dicts
                    page_elements = [{
                        "page_number": chunk["page_number"],
                        "chunks": [{
                            "id": f"{document['id']}-{ch['chunk_index']}",
                            "text": ch["text"]
                        } for ch in document.get("content_chunks", [])
                        if ch.get("page_number") == chunk["page_number"]],
                        "tables": [{
                            "id": t["id"],
                            "image_url": t.get("image_url", "")
                        } for t in document.get("tables", [])
                        if t.get("page") == chunk["page_number"]],
                        "images": [{
                            "id": i["id"],
                            "url": i.get("url", "")
                        } for i in document.get("images", [])
                        if i.get("page") == chunk["page_number"]],
                        "warnings": [{
                            "id": w["id"],
                            "text": w.get("text", "")
                        } for w in document.get("warnings", [])
                        if w.get("page") == chunk["page_number"]]
                    }]

                    search_doc = {
                        "id": f"{document['id']}-{chunk['chunk_index']}",
                        "title": document["title"],
                        "content": chunk["text"],
                        "source_file": document["source_file"],
                        "content_vector": embeddings,
                        "metadata": {
                            "processing_date": document["metadata"]["processing_date"],
                            "page_count": document["metadata"]["page_count"],
                            "word_count": chunk["word_count"],
                            "document_type": document["metadata"].get("document_type", "manual"),
                            "chunk_index": chunk["chunk_index"],
                            "total_chunks": document["metadata"].get("total_chunks", len(document["content_chunks"])),
                            "page_number": chunk["page_number"],
                            "table_count": document["metadata"].get("table_count", 0),
                            "image_count": document["metadata"].get("image_count", 0)
                        },
                        "warnings": [{
                            "text": w.get("text", ""),
                            "severity": w.get("severity", "medium"),
                            "context": w.get("context", ""),
                            "page": w["page"],
                            "chunk_references": [f"{document['id']}-{chunk['chunk_index']}"]
                        } for w in document.get("warnings", [])
                        if w.get("page") == chunk["page_number"]],
                        "tables": [{
                            "caption": t.get("caption", f"Table on page {t['page']}"),
                            "content_markdown": t.get("content_markdown", ""),
                            "page": t["page"],
                            "content": t.get("content", ""),
                            "row_count": t.get("row_count", 0),
                            "column_count": t.get("column_count", 0),
                            "image_url": t.get("image_url", ""),
                            "bounding_box": t.get("bounding_box", []),
                            "chunk_references": [f"{document['id']}-{chunk['chunk_index']}"],
                            "related_content": {
                                "chunk_ids": [f"{document['id']}-{ch['chunk_index']}"
                                for ch in document.get("content_chunks", [])
                                if ch.get("page_number") == t["page"]],
                                "image_ids": [img["id"] for img in document.get("images", [])
                                if img.get("page") == t["page"]],
                                "warning_ids": [w["id"] for w in document.get("warnings", [])
                                if w.get("page") == t["page"]]
                            }
                        } for t in document.get("tables", [])
                        if t.get("page") == chunk["page_number"]],
                        "images": [{
                            "url": i.get("url", ""),
                            "caption": i.get("caption", f"Image on page {i['page']}"),
                            "page": i["page"],
                            "dimensions": i.get("dimensions", "0x0"),
                            "size_kb": i.get("size_kb", 0),
                            "bounding_box": i.get("bounding_box", []),
                            "chunk_references": [f"{document['id']}-{chunk['chunk_index']}"],
                            "related_content": {
                                "chunk_ids": [f"{document['id']}-{ch['chunk_index']}"
                                for ch in document.get("content_chunks", [])
                                if ch.get("page_number") == i["page"]],
                                "table_ids": [t["id"] for t in document.get("tables", [])
                                if t.get("page") == i["page"]],
                                "warning_ids": [w["id"] for w in document.get("warnings", [])
                                if w.get("page") == i["page"]]
                            },
                            "analysis": {
                                "tags": i.get("tags", []),
                                "description": i.get("caption", "")
                            }
                        } for i in document.get("images", [])
                        if i.get("page") == chunk["page_number"]],
                        "relationships": [{
                            "content_type": rel.get("content_type", "reference"),
                            "content_id": rel["content_id"],
                            "content_summary": rel.get("content_summary", ""),
                            "source_chunk_index": rel["source_chunk_index"],
                            "target_page": rel.get("target_page", 0),
                            "relationship_type": rel.get("relationship_type", "reference"),
                            "confidence_score": rel.get("confidence_score", 0.0)
                        } for rel in document.get("relationships", [])
                        if rel.get("source_chunk_index") == chunk["chunk_index"]],
                        "page_elements": page_elements  # Always a list of dict(s)
                    }

                    search_docs.append(search_doc)

                except Exception as e:
                    logger.warning(f"Failed to prepare chunk {chunk.get('chunk_index', 'unknown')}: {str(e)}")
                    continue

        return search_docs

    def _generate_embeddings_with_retry(self, text: str) -> List[float]:
        """Generate embeddings with retry logic and validation."""
        if not text.strip():
            return [0.0] * self.embedding_dimensions
            
        for attempt in range(3):
            try:
                start_time = time.time()
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.embedding_deployment,
                    timeout=self.embedding_timeout
                )
                embeddings = response.data[0].embedding
                
                if len(embeddings) != self.embedding_dimensions:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimensions}, got {len(embeddings)}")
                
                logger.debug(f"Generated embeddings in {time.time()-start_time:.2f}s")
                return embeddings
                
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to generate embeddings after 3 attempts: {str(e)}")
                    return [0.0] * self.embedding_dimensions
                
                wait_time = min((2 ** attempt) + random.random(), self.max_retry_delay)
                logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {str(e)}")
                time.sleep(wait_time)

    def _index_batch_with_retry(self, batch: List[Dict]) -> int:
        """Index a batch of documents with exponential backoff retry logic."""
        if not batch:
            return 0
            
        for attempt in range(self.max_upload_retries):
            try:
                # Final validation before sending to search
                valid_batch = [doc for doc in batch if self._validate_search_document(doc)]
                if len(valid_batch) != len(batch):
                    logger.warning(f"Filtered out {len(batch) - len(valid_batch)} invalid documents before indexing")
                
                result = self.search_client.upload_documents(valid_batch)
                succeeded = sum(1 for r in result if r.succeeded)
                
                if succeeded < len(valid_batch):
                    failed = [r.key for r in result if not r.succeeded]
                    logger.warning(f"Batch partial success: {succeeded}/{len(valid_batch)}. Failed: {failed}")
                
                return succeeded
                
            except Exception as e:
                if attempt == self.max_upload_retries - 1:
                    logger.error(f"Final attempt failed for batch: {str(e)}")
                    return 0
                
                wait_time = min((2 ** attempt) + random.random(), self.max_retry_delay)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {str(e)}")
                time.sleep(wait_time)
        
        return 0

if __name__ == "__main__":
    try:
        indexer = DataIndexer()
        indexed_count = indexer.index_documents()
        logger.info(f"Indexed {indexed_count} documents successfully")
    except Exception as e:
        logger.error(f"Data indexing failed: {str(e)}")
        raise