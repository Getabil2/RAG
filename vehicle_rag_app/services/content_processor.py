from typing import List, Dict
from models.documents import *
from models.responses import ProcessedContent
import re
import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ContentProcessor:
    def __init__(self, storage_service):
        self.storage = storage_service

    def process(self, search_results: List[Dict]) -> ProcessedContent:
        """Process search results into structured content"""
        processed = ProcessedContent()
        
        for doc in search_results:
            try:
                metadata = doc.get("metadata", {})
                source_file = doc.get("source_file", "")
                
                # Process source document
                source = self._process_source(doc, metadata, source_file)
                processed.sources.append(source)
                
                # Process chunks
                if "content" in doc:
                    chunk = DocumentChunk(
                        text=doc["content"],
                        page_number=metadata.get("page_number", 0),
                        chunk_index=metadata.get("chunk_index", 0),
                        word_count=len(doc["content"].split()),
                        source_file=source_file
                    )
                    processed.chunks.append(chunk)
                    processed.text_context += f"\n\n{doc['content']}"
                
                # Process images
                for img in doc.get("images", []):
                    processed.images.append(self._process_image(img, source))
                
                # Process tables
                for table in doc.get("tables", []):
                    processed.tables.append(self._process_table(table, source))
                
                # Process warnings
                for warning in doc.get("warnings", []):
                    processed.warnings.append(self._process_warning(warning, source))
                    
            except Exception as e:
                logger.warning(f"Failed to process document: {str(e)}")
                continue
                
        return processed

    def _process_source(self, doc: Dict, metadata: Dict, source_file: str) -> DocumentSource:
        """Process document source information"""
        pdf_blob_name = (
            source_file
            .replace('processed_data/', 'raw_data/')
            .replace('.json', '.pdf')
            if source_file.endswith('.json') 
            else source_file
        )
        
        return DocumentSource(
            id=doc.get("id", ""),
            title=metadata.get("title", os.path.basename(pdf_blob_name)),
            content=doc.get("content", "")[:500] + ("..." if len(doc.get("content", "")) > 500 else ""),
            source_url=self.storage.get_blob_url_with_sas(pdf_blob_name),
            page_number=str(metadata.get("page_number", "")),
            document_type=metadata.get("document_type", "manual"),
            processing_date=metadata.get("processing_date", datetime.utcnow())
        )

    def _process_image(self, img: Dict, source: DocumentSource) -> DocumentImage:
        """Process image information"""
        return DocumentImage(
            url=img.get("url", ""),
            caption=img.get("caption", f"Image from page {img.get('page', '')}"),
            page_number=img.get("page", 0),
            dimensions=img.get("dimensions", ""),
            size_kb=img.get("size_kb", 0),
            content_id=f"image-{img.get('caption', str(img.get('page', '')))}",
            blob_name=img.get("url", "").split(f"/{self.storage.container_name}/")[-1].split("?")[0]
        )

    def _process_table(self, table: Dict, source: DocumentSource) -> DocumentTable:
        """Process table information"""
        return DocumentTable(
            caption=table.get("caption", f"Table from page {table.get('page', '')}"),
            content=table.get("content", ""),
            content_markdown=table.get("content_markdown", ""),
            page_number=table.get("page", 0),
            row_count=table.get("row_count", 0),
            column_count=table.get("column_count", 0),
            content_id=f"table-{table.get('caption', str(table.get('page', '')))}"
        )

    def _process_warning(self, warning: Dict, source: DocumentSource) -> DocumentWarning:
        """Process warning information"""
        return DocumentWarning(
            text=warning.get("text", ""),
            severity=warning.get("severity", "medium"),
            page_number=warning.get("page", 0),
            context=warning.get("context", ""),
            content_id=f"warning-{warning.get('severity', '')}-{warning.get('page', '')}"
        )