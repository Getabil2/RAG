from typing import List, Dict
from collections import defaultdict
from models.documents import *
from models.responses import Relationship
import logging

logger = logging.getLogger(__name__)

class RelationshipBuilder:
    def build(self, chunks: List[DocumentChunk], tables: List[DocumentTable], 
              images: List[DocumentImage], warnings: List[DocumentWarning]) -> List[Relationship]:
        """Build relationships between different content types"""
        relationships = []
        
        # Create page to chunk mapping
        page_to_chunks = defaultdict(list)
        for chunk in chunks:
            page_to_chunks[chunk.page_number].append(chunk)
        
        # Link tables to chunks
        for table in tables:
            for chunk in page_to_chunks.get(table.page_number, []):
                relationships.append(self._create_relationship(
                    source_type="chunk",
                    source_id=f"chunk-{chunk.chunk_index}",
                    target_type="table",
                    target_id=table.content_id,
                    rel_type="references"
                ))
        
        # Link images to chunks
        for image in images:
            for chunk in page_to_chunks.get(image.page_number, []):
                relationships.append(self._create_relationship(
                    source_type="chunk",
                    source_id=f"chunk-{chunk.chunk_index}",
                    target_type="image",
                    target_id=image.content_id,
                    rel_type="references"
                ))
        
        # Link warnings to chunks
        for warning in warnings:
            for chunk in page_to_chunks.get(warning.page_number, []):
                relationships.append(self._create_relationship(
                    source_type="chunk",
                    source_id=f"chunk-{chunk.chunk_index}",
                    target_type="warning",
                    target_id=warning.content_id,
                    rel_type="references"
                ))
        
        return relationships

    def _create_relationship(self, source_type: str, source_id: str, 
                           target_type: str, target_id: str, rel_type: str) -> Relationship:
        """Create a relationship object"""
        return Relationship(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relationship_type=rel_type
        )