from typing import List, Dict
from dataclasses import dataclass
from models.documents import *

@dataclass
class Relationship:
    source_type: str  # "chunk", "table", "image", "warning"
    source_id: str
    target_type: str
    target_id: str
    relationship_type: str  # "references", "related_to", "supports"

@dataclass
class ProcessedContent:
    text_context: str = ""
    sources: List[DocumentSource] = None
    chunks: List[DocumentChunk] = None
    images: List[DocumentImage] = None
    tables: List[DocumentTable] = None
    warnings: List[DocumentWarning] = None

    def __post_init__(self):
        self.sources = self.sources or []
        self.chunks = self.chunks or []
        self.images = self.images or []
        self.tables = self.tables or []
        self.warnings = self.warnings or []

@dataclass
class RAGResponse:
    answer: str
    sources: List[DocumentSource]
    relationships: List[Relationship]
    images: List[DocumentImage]
    tables: List[DocumentTable]
    warnings: List[DocumentWarning]
    not_found: bool = False
    chunks: List[DocumentChunk] = None  # Add this line
