from typing import List, Optional
from models.responses import RAGResponse, ProcessedContent
from models.documents import DocumentImage
import logging
from services.search_service import SearchService
from services.content_processor import ContentProcessor
from services.relationship_builder import RelationshipBuilder
logger = logging.getLogger(__name__)

class MultimodalRAGAgent:
    def __init__(self, search_service, llm_service, storage_service):
        self.search = search_service
        self.llm = llm_service
        self.storage = storage_service

    async def process_query(self, query: str, images: List[bytes] = None) -> RAGResponse:
        """Orchestrate the full RAG pipeline"""
        try:
            # 1. Retrieve relevant documents
            search_results = await self.search.retrieve(query)
            if not search_results:
                return RAGResponse(
                    answer="No relevant documents found.",
                    sources=[],
                    relationships=[],
                    images=[],
                    tables=[],
                    warnings=[],
                    not_found=True,
                     chunks=[],

                )
            
            # 2. Extract and process all content types
            content_processor = ContentProcessor(self.storage)
            processed_content = content_processor.process(search_results)
            
            # 3. Download image bytes for multimodal processing
            image_bytes = []
            for img in processed_content.images:
                if img.blob_name:
                    bytes_data = await self.storage.download_blob_to_bytes(img.blob_name)
                    if bytes_data:
                        image_bytes.append(bytes_data)
            
            # 4. Generate multimodal response
            response = await self.llm.generate_response(
                query=query,
                context=processed_content.text_context,
                images=image_bytes + (images or [])
            )
            
            # 5. Build relationships
            relationship_builder = RelationshipBuilder()
            relationships = relationship_builder.build(
                chunks=processed_content.chunks,
                tables=processed_content.tables,
                images=processed_content.images,
                warnings=processed_content.warnings
            )
            
            return RAGResponse(
                answer=response,
                sources=processed_content.sources,
                relationships=relationships,
                images=processed_content.images,
                tables=processed_content.tables,
                warnings=processed_content.warnings,
                chunks=processed_content.chunks,  # Include chunks

                not_found=False
            )
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {str(e)}")
            raise