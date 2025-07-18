import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import json

# Azure imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SimpleField, SearchableField, ComplexField,
    VectorSearch, HnswAlgorithmConfiguration, HnswParameters, VectorSearchProfile,
    SemanticConfiguration, SemanticField, SemanticPrioritizedFields, SemanticSearch
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("index_management.log")
    ]
)
logger = logging.getLogger(__name__)

class IndexManager:
    def __init__(self):
        """Initialize the index manager with Azure Search client."""
        self._validate_configurations()
        self.index_client = SearchIndexClient(
            endpoint=os.getenv("SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("SEARCH_KEY")),
            timeout=60
        )
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    def _validate_configurations(self):
        """Validate required configurations."""
        required_vars = [
            "SEARCH_ENDPOINT",
            "SEARCH_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def create_or_update_index(self) -> bool:
        """Create or update the search index with optimized configuration."""
        index_name = os.getenv("INDEX_NAME", "vehicle-manuals")
        try:
            # Delete existing index if it exists
            try:
                self.index_client.delete_index(index_name)
                logger.info(f"Deleted existing index: {index_name}")
            except Exception as e:
                logger.info(f"No existing index to delete: {str(e)}")
            
            # Create new index
            fields = self._get_index_fields()
            vector_search = self._get_vector_search_config()
            semantic_config = self._get_semantic_config()
            
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=SemanticSearch(configurations=[semantic_config]))
            
            self.index_client.create_or_update_index(index)
            logger.info("Search index created/updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/update index: {str(e)}")
            raise

    def _get_index_fields(self) -> List[SearchField]:
        """Return optimized index field configurations with enhanced table and image support."""
        return [
            # Core document fields
            SimpleField(name="id", type="Edm.String", key=True, filterable=True),
            SearchableField(name="title", type="Edm.String", analyzer="en.microsoft", searchable=True),
            SearchableField(name="content", type="Edm.String", analyzer="en.microsoft", searchable=True),
            SearchableField(name="source_file", type="Edm.String", filterable=True),
            
            # Vector field for embeddings
            SearchField(
                name="content_vector",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=self.embedding_dimensions,
                vector_search_profile_name="vector-profile"
            ),
            
            # Document metadata
            ComplexField(name="metadata", fields=[
                SimpleField(name="processing_date", type="Edm.DateTimeOffset", filterable=True),
                SimpleField(name="page_count", type="Edm.Int32", filterable=True),
                SimpleField(name="word_count", type="Edm.Int32", filterable=True),
                SimpleField(name="document_type", type="Edm.String", filterable=True),
                SimpleField(name="chunk_index", type="Edm.Int32", filterable=True),
                SimpleField(name="total_chunks", type="Edm.Int32", filterable=True),
                SimpleField(name="page_number", type="Edm.Int32", filterable=True),
                SimpleField(name="table_count", type="Edm.Int32", filterable=True),
                SimpleField(name="image_count", type="Edm.Int32", filterable=True)
            ]),
            
            # Warnings and safety information
            ComplexField(name="warnings", collection=True, fields=[
                SearchableField(name="text", type="Edm.String", searchable=True),
                SimpleField(name="severity", type="Edm.String", filterable=True),
                SearchableField(name="context", type="Edm.String", searchable=True),
                SimpleField(name="page", type="Edm.Int32", filterable=True),
                SimpleField(name="chunk_references", type="Collection(Edm.String)")
            ]),
            
            # Enhanced tables with image support
            ComplexField(name="tables", collection=True, fields=[
                SearchableField(name="caption", type="Edm.String", searchable=True),
                SearchableField(name="content_markdown", type="Edm.String", searchable=True),
                SimpleField(name="page", type="Edm.Int32", filterable=True),
                SimpleField(name="content", type="Edm.String"),
                SimpleField(name="row_count", type="Edm.Int32", filterable=True),
                SimpleField(name="column_count", type="Edm.Int32", filterable=True),
                SimpleField(name="image_url", type="Edm.String"),
                SimpleField(name="bounding_box", type="Edm.String"),
                SimpleField(name="chunk_references", type="Collection(Edm.String)"),
                ComplexField(name="related_content", fields=[
                    SimpleField(name="chunk_ids", type="Collection(Edm.String)"),
                    SimpleField(name="image_ids", type="Collection(Edm.String)"),
                    SimpleField(name="warning_ids", type="Collection(Edm.String)")
                ])
            ]),
            
            # Enhanced images with content relationships
            ComplexField(name="images", collection=True, fields=[
                SimpleField(name="url", type="Edm.String"),
                SearchableField(name="caption", type="Edm.String", searchable=True),
                SimpleField(name="page", type="Edm.Int32", filterable=True),
                SimpleField(name="dimensions", type="Edm.String"),
                SimpleField(name="size_kb", type="Edm.Double", filterable=True),
                SimpleField(name="bounding_box", type="Edm.String"),
                SimpleField(name="chunk_references", type="Collection(Edm.String)"),
                ComplexField(name="related_content", fields=[
                    SimpleField(name="chunk_ids", type="Collection(Edm.String)"),
                    SimpleField(name="table_ids", type="Collection(Edm.String)"),
                    SimpleField(name="warning_ids", type="Collection(Edm.String)")
                ]),
                ComplexField(name="analysis", fields=[
                    SearchableField(name="tags", type="Collection(Edm.String)", searchable=True),
                    SearchableField(name="description", type="Edm.String", searchable=True)
                ])
            ]),
            
            # Enhanced relationships for RAG
            ComplexField(name="relationships", collection=True, fields=[
                SimpleField(name="content_type", type="Edm.String", filterable=True),
                SimpleField(name="content_id", type="Edm.String"),
                SearchableField(name="content_summary", type="Edm.String", searchable=True),
                SimpleField(name="source_chunk_index", type="Edm.Int32", filterable=True),
                SimpleField(name="target_page", type="Edm.Int32", filterable=True),
                SimpleField(name="relationship_type", type="Edm.String", filterable=True),
                SimpleField(name="confidence_score", type="Edm.Double", filterable=True)
            ]),
            
            # Page-level content grouping
            ComplexField(name="page_elements", collection=True, fields=[
                SimpleField(name="page_number", type="Edm.Int32", filterable=True),
                ComplexField(name="chunks", collection=True, fields=[
                    SimpleField(name="id", type="Edm.String"),
                    SearchableField(name="text", type="Edm.String", searchable=True)
                ]),
                ComplexField(name="tables", collection=True, fields=[
                    SimpleField(name="id", type="Edm.String"),
                    SimpleField(name="image_url", type="Edm.String")
                ]),
                ComplexField(name="images", collection=True, fields=[
                    SimpleField(name="id", type="Edm.String"),
                    SimpleField(name="url", type="Edm.String")
                ]),
                ComplexField(name="warnings", collection=True, fields=[
                    SimpleField(name="id", type="Edm.String"),
                    SearchableField(name="text", type="Edm.String", searchable=True)
                ])
            ])
        ]

    def _get_vector_search_config(self) -> VectorSearch:
        """Return optimized vector search configuration."""
        return VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )

    def _get_semantic_config(self) -> SemanticConfiguration:
        """Return optimized semantic search configuration with only valid string fields."""
        return SemanticConfiguration(
    name="vehicle-manuals-semantic",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        content_fields=[
            SemanticField(field_name="content"),
        ],
        keywords_fields=[
            SemanticField(field_name="metadata/document_type"),
        ]
    )
)

        

    def delete_index(self) -> bool:
        """Delete the search index."""
        index_name = os.getenv("INDEX_NAME", "vehicle-manuals")
        try:
            self.index_client.delete_index(index_name)
            logger.info(f"Successfully deleted index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index: {str(e)}")
            return False

    def _field_to_dict(self, field: SearchField) -> Dict[str, Any]:
        """Convert a SearchField object to a serializable dictionary."""
        field_dict = {
            "name": field.name,
            "type": field.type,
            "searchable": getattr(field, "searchable", None),
            "filterable": getattr(field, "filterable", None),
            "key": getattr(field, "key", None),
            "analyzer": getattr(field, "analyzer", None),
            "collection": getattr(field, "collection", None)
        }
        
        # Handle nested fields if they exist
        if hasattr(field, "fields") and field.fields is not None:
            field_dict["fields"] = [self._field_to_dict(f) for f in field.fields]
        
        # Handle vector search properties
        if hasattr(field, "vector_search_dimensions"):
            field_dict["vector_search_dimensions"] = field.vector_search_dimensions
        if hasattr(field, "vector_search_profile_name"):
            field_dict["vector_search_profile_name"] = field.vector_search_profile_name
            
        return {k: v for k, v in field_dict.items() if v is not None}

    def _vector_search_to_dict(self, config: VectorSearch) -> Dict[str, Any]:
        """Convert VectorSearch configuration to dictionary."""
        return {
            "algorithms": [{
                "name": alg.name,
                "parameters": {
                    "m": alg.parameters.m,
                    "ef_construction": alg.parameters.ef_construction,
                    "ef_search": alg.parameters.ef_search,
                    "metric": alg.parameters.metric
                }
            } for alg in config.algorithms],
            "profiles": [{
                "name": profile.name,
                "algorithm_configuration_name": profile.algorithm_configuration_name
            } for profile in config.profiles]
        }

    def _semantic_config_to_dict(self, config: SemanticConfiguration) -> Dict[str, Any]:
        """Convert SemanticConfiguration to dictionary."""
        return {
            "name": config.name,
            "prioritized_fields": {
                "title_field": {"field_name": config.prioritized_fields.title_field.field_name},
                "content_fields": [{"field_name": f.field_name} for f in config.prioritized_fields.content_fields],
                "keywords_fields": [{"field_name": f.field_name} for f in config.prioritized_fields.keywords_fields]
            }
        }

    def get_index_schema(self) -> Dict[str, Any]:
        """Return the complete index schema as a serializable dictionary."""
        return {
            "fields": [self._field_to_dict(field) for field in self._get_index_fields()],
            "vector_search": self._vector_search_to_dict(self._get_vector_search_config()),
            "semantic_config": self._semantic_config_to_dict(self._get_semantic_config())
        }

if __name__ == "__main__":
    try:
        manager = IndexManager()
        
        # Verify schema before creation
        logger.info("Index schema:")
        logger.info(json.dumps(manager.get_index_schema(), indent=2))
        
        # Create/update index
        success = manager.create_or_update_index()
        if success:
            logger.info("Index management completed successfully")
        else:
            logger.error("Index management failed")
    except Exception as e:
        logger.error(f"Index management failed: {str(e)}")
        raise