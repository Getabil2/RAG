from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentChunk:
    text: str
    page_number: int
    chunk_index: int
    word_count: int
    source_file: str

@dataclass
class DocumentImage:
    url: str
    caption: str
    page_number: int
    dimensions: str
    size_kb: float
    content_id: str
    blob_name: str

@dataclass
class DocumentTable:
    caption: str
    content: str
    content_markdown: str
    page_number: int
    row_count: int
    column_count: int
    content_id: str

@dataclass
class DocumentWarning:
    text: str
    severity: str  # "critical", "high", "medium", "low"
    page_number: int
    context: str
    content_id: str

@dataclass
class DocumentSource:
    id: str
    title: str
    content: str
    source_url: str
    page_number: str
    document_type: str
    processing_date: datetime