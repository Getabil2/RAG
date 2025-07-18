from typing import List, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from models.documents import DocumentChunk
from models.responses import ProcessedContent
import logging
from openai import AzureOpenAI
import os

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, endpoint: str, key: str, index_name: str):
        self.endpoint = endpoint
        self.key = key
        self.index_name = index_name
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(key)
        )
        
        # Initialize OpenAI client for embeddings
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    async def retrieve(self, query: str) -> List[dict]:
        """Retrieve relevant documents from Azure Cognitive Search"""
        try:
            # Generate embedding for the query
            embedding = self._generate_embedding(query)
            
            # Perform vector search
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=5,
                fields="content_vector"
            )
            
            results = self.client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "content", "metadata", "source_file", "tables", "images", "warnings"]
            )
            
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            embeddings = response.data[0].embedding
            
            if len(embeddings) != self.embedding_dimensions:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimensions}, got {len(embeddings)}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise