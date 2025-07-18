import os
import json
import time
import logging
import re
import random
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import io
import concurrent.futures
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# Azure imports
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.ai.formrecognizer import DocumentAnalysisClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("document_extraction.log")
    ]
)
logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self):
        self._initialize_services()
        self._setup_configurations()
        self._validate_configurations()
        
    def _initialize_services(self):
        try:
            self.blob_service = BlobServiceClient.from_connection_string(
                os.getenv("STORAGE_CONNECTION_STRING"),
                retry_total=3,
                retry_backoff_factor=0.5,
                max_single_get_size=4*1024*1024,
                connection_timeout=30
            )
            self.form_recognizer = DocumentAnalysisClient(
                endpoint=os.getenv("DOC_INTEL_ENDPOINT"),
                credential=AzureKeyCredential(os.getenv("DOC_INTEL_KEY")),
                headers={"x-ms-useragent": "DocExtractor/2.0"},
                timeout=60
            )
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            raise

    def _setup_configurations(self):
        self.container_name = os.getenv("CONTAINER_NAME", "vehicle-manuals")
        self.processed_prefix = os.getenv("PROCESSED_PREFIX", "processed_data/")
        self.image_prefix = os.getenv("IMAGE_PREFIX", "processed_data/images/")
        self.table_prefix = os.getenv("TABLE_PREFIX", "processed_data/tables/")
        default_workers = min(8, (os.cpu_count() or 4))
        self.max_workers = min(
            int(os.getenv("MAX_WORKERS", default_workers)),
            default_workers * 2
        )
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", 4 * 1024 * 1024))
        self.chunk_size_words = int(os.getenv("CHUNK_SIZE_WORDS", 500))
        self.table_processing_timeout = int(os.getenv("TABLE_PROCESSING_TIMEOUT", 30))
        self.max_retry_delay = int(os.getenv("MAX_RETRY_DELAY", 60))
        self.warning_pattern = re.compile(
            r'(?i)(warning|caution|danger|note|important)[\s:]+(.+?)(?=\n|$)',
            re.MULTILINE
        )

    def _validate_configurations(self):
        required_vars = [
            "STORAGE_CONNECTION_STRING",
            "DOC_INTEL_ENDPOINT",
            "DOC_INTEL_KEY"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def process_all_documents(self):
        logger.info("Starting document processing pipeline")
        start_time = time.time()
        try:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_list = [blob.name for blob in container_client.list_blobs(name_starts_with="raw_data/") 
                        if blob.name.endswith('.pdf')]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_single_document, blob_name): blob_name 
                          for blob_name in blob_list}
                for future in concurrent.futures.as_completed(futures):
                    blob_name = futures[future]
                    try:
                        doc = future.result()
                        logger.info(f"Successfully processed: {blob_name}")
                    except Exception as e:
                        logger.error(f"Failed to process {blob_name}: {str(e)}")
            logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

    def process_single_document(self, blob_name: str) -> Dict:
        logger.info(f"Processing document: {blob_name}")
        start_time = time.time()
        try:
            pdf_bytes = self._download_document_with_retry(blob_name)
            if not pdf_bytes:
                raise ValueError(f"Failed to download document: {blob_name}")
            analysis_result = self._analyze_document_with_retry(pdf_bytes)
            doc_id = self._generate_document_id(blob_name)
            chunks = self._extract_content_chunks(analysis_result, doc_id)
            tables = self._extract_tables(pdf_bytes, analysis_result, doc_id)
            images = self._extract_images(pdf_bytes, doc_id)
            warnings = self._extract_warnings(analysis_result, doc_id)
            document = {
                "id": doc_id,
                "source_file": blob_name,
                "title": self._generate_document_title(blob_name),
                "content_chunks": chunks,
                "tables": tables,
                "images": images,
                "warnings": warnings,
                "metadata": {
                    "processing_date": datetime.now(timezone.utc).isoformat(),
                    "page_count": len(analysis_result.pages),
                    "document_type": "manual",
                    "total_chunks": len(chunks),
                    "table_count": len(tables),
                    "image_count": len(images),
                    "warning_count": len(warnings)
                }
            }
            self._save_processed_document(document)
            logger.info(
                f"Processed {blob_name} in {time.time() - start_time:.2f}s - "
                f"Pages: {len(analysis_result.pages)}, "
                f"Chunks: {len(chunks)}, "
                f"Images: {len(images)}, "
                f"Tables: {len(tables)}, "
                f"Warnings: {len(warnings)}"
            )
            return document
        except Exception as e:
            logger.error(f"Error processing {blob_name}: {str(e)}")
            raise

    def _extract_tables(self, pdf_bytes: bytes, analysis_result, doc_id: str) -> List[Dict]:
        tables = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for table_idx, table in enumerate(analysis_result.tables, start=1):
                try:
                    page_num = table.bounding_regions[0].page_number if table.bounding_regions else 1
                    table_id = f"table_{doc_id}_p{page_num}_{table_idx}"
                    table_data = {
                        "id": table_id,
                        "content_id": f"content_{table_id}",
                        "caption": f"Table {table_idx} on page {page_num}",
                        "page": page_num,
                        "content": "\n".join(c.content for c in table.cells),
                        "content_markdown": self._convert_table_to_markdown(table),
                        "row_count": max(c.row_index for c in table.cells) + 1,
                        "column_count": max(c.column_index for c in table.cells) + 1,
                        "bounding_box": self._get_bounding_box(table.bounding_regions[0].polygon)
                    }
                    table_image = self._extract_table_image(doc, page_num - 1, table.bounding_regions[0].polygon)
                    if table_image:
                        image_url = self._upload_table_image(table_image, doc_id, page_num, table_idx)
                        table_data["image_url"] = image_url
                    tables.append(table_data)
                except Exception as e:
                    logger.warning(f"Skipping table {table_idx}: {str(e)}")
                    continue
        return tables

    def _extract_content_chunks(self, analysis_result, doc_id: str) -> List[Dict]:
        chunks = []
        chunk_idx = 0
        for page_num, page in enumerate(analysis_result.pages, start=1):
            page_paragraphs = [p for p in analysis_result.paragraphs 
                            if p.bounding_regions and p.bounding_regions[0].page_number == page_num]
            for para in page_paragraphs:
                chunk_id = f"chunk_{doc_id}_p{page_num}_{chunk_idx}"
                chunks.append({
                    "id": chunk_id,
                    "content_id": f"content_{chunk_id}",
                    "text": para.content,
                    "page_number": page_num,
                    "word_count": len(para.content.split()),
                    "bounding_box": self._get_bounding_box(para.bounding_regions[0].polygon),
                    "chunk_index": chunk_idx
                })
                chunk_idx += 1
        return chunks

    def _extract_images(self, pdf_bytes: bytes, doc_id: str) -> List[Dict]:
        images = []
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    for img_num, img in enumerate(page.get_images(full=True), start=1):
                        try:
                            xref = img[0]
                            img_info = doc.extract_image(xref)
                            if not (2048 <= len(img_info["image"]) <= self.max_image_size):
                                continue
                            img_id = f"img_{doc_id}_p{page_num+1}_{img_num}"
                            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                            ext = img_info["ext"] if img_info["ext"] in ["png", "jpg", "jpeg"] else "png"
                            blob_name = f"{self.image_prefix}{doc_id}/page_{page_num+1}_img_{img_num}_{timestamp}.{ext}"
                            blob_client = self.blob_service.get_blob_client(
                                container=self.container_name,
                                blob=blob_name)
                            blob_client.upload_blob(
                                img_info["image"],
                                overwrite=True,
                                content_settings=ContentSettings(content_type=f"image/{ext}"),
                                max_concurrency=2
                            )
                            images.append({
                                "id": img_id,
                                "content_id": f"content_{img_id}",
                                "url": blob_client.url,
                                "page": page_num + 1,
                                "caption": f"Page {page_num+1} Image {img_num}",
                                "dimensions": f"{img_info['width']}x{img_info['height']}",
                                "size_kb": len(img_info["image"]) / 1024,
                                "format": ext,
                                "bounding_box": self._get_image_bbox(page, img)
                            })
                        except Exception as e:
                            logger.warning(f"Skipping image on page {page_num+1}: {str(e)}")
                            continue
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
        return images

    def _extract_warnings(self, analysis_result, doc_id: str) -> List[Dict]:
        warnings = []
        warning_idx = 0
        for page_num, page in enumerate(analysis_result.pages, start=1):
            page_text = "\n".join(
                p.content for p in analysis_result.paragraphs 
                if p.bounding_regions and p.bounding_regions[0].page_number == page_num
            )
            for match in self.warning_pattern.finditer(page_text):
                severity = ("critical" if "danger" in match.group(1).lower() else
                          "high" if "warning" in match.group(1).lower() else
                          "medium" if "caution" in match.group(1).lower() else "low")
                warning_id = f"warning_{doc_id}_p{page_num}_{warning_idx}"
                warnings.append({
                    "id": warning_id,
                    "content_id": f"content_{warning_id}",
                    "text": match.group(2).strip(),
                    "severity": severity,
                    "page": page_num,
                    "context": match.group(0)[:200]
                })
                warning_idx += 1
        return warnings

    def _get_bounding_box(self, polygon) -> Optional[List[Dict]]:
        if not polygon or len(polygon) < 3:
            return None
        return [{"x": point.x, "y": point.y} for point in polygon]

    def _get_image_bbox(self, page, img_info) -> Optional[Dict]:
        try:
            xref = img_info[0]
            bbox = page.get_image_bbox(xref)
            return {
                "x0": bbox.x0,
                "y0": bbox.y0,
                "x1": bbox.x1,
                "y1": bbox.y1
            }
        except Exception:
            return None

    def _convert_table_to_markdown(self, table) -> str:
        if not table.cells:
            return ""
        try:
            max_row = max(cell.row_index for cell in table.cells)
            max_col = max(cell.column_index for cell in table.cells)
            grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            for cell in table.cells:
                if cell.row_index <= max_row and cell.column_index <= max_col:
                    grid[cell.row_index][cell.column_index] = cell.content
            markdown = []
            for row in grid:
                markdown.append("| " + " | ".join(row) + " |")
            if len(grid) > 1:
                markdown.insert(1, "|" + "|".join(["---"] * (max_col + 1)) + "|")
            return "\n".join(markdown)
        except Exception as e:
            logger.warning(f"Table conversion error: {str(e)}")
            return ""

    def _download_document_with_retry(self, blob_name: str) -> Optional[bytes]:
        blob_client = self.blob_service.get_blob_client(
            container=self.container_name,
            blob=blob_name)
        if not blob_client.exists():
            logger.warning(f"Blob {blob_name} does not exist")
            return None
        for attempt in range(3):
            try:
                stream = io.BytesIO()
                download_stream = blob_client.download_blob(max_concurrency=4)
                for chunk in download_stream.chunks():
                    stream.write(chunk)
                return stream.getvalue()
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to download {blob_name} after 3 attempts: {str(e)}")
                    return None
                wait_time = min((2 ** attempt) + random.random(), self.max_retry_delay)
                logger.warning(f"Attempt {attempt + 1} failed for {blob_name}, retrying in {wait_time:.1f}s: {str(e)}")
                time.sleep(wait_time)

    def _analyze_document_with_retry(self, pdf_bytes: bytes):
        for attempt in range(3):
            try:
                poller = self.form_recognizer.begin_analyze_document(
                    "prebuilt-layout",
                    pdf_bytes
                )
                return poller.result()
            except Exception as e:
                if attempt == 2:
                    raise
                wait_time = min((2 ** attempt) + random.random(), self.max_retry_delay)
                logger.warning(f"Document analysis attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {str(e)}")
                time.sleep(wait_time)

    def _save_processed_document(self, document: Dict):
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name,
                blob=f"{self.processed_prefix}{document['id']}.json")
            data = json.dumps(document, indent=2).encode('utf-8')
            with io.BytesIO(data) as stream:
                blob_client.upload_blob(
                    stream,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                    max_concurrency=2
                )
        except Exception as e:
            logger.error(f"Failed to save document {document['id']}: {str(e)}")
            raise

    def _generate_document_id(self, filename: str) -> str:
        base = os.path.splitext(os.path.basename(filename))[0]
        clean_base = re.sub(r'[^\w\-_]', '', base.lower())
        return f"{clean_base[:50]}_{uuid.uuid4().hex[:8]}"

    def _generate_document_title(self, filename: str) -> str:
        base = os.path.splitext(os.path.basename(filename))[0]
        return re.sub(r'[_\-]+', ' ', base).title()

def main():
    try:
        extractor = DocumentExtractor()
        extractor.process_all_documents()
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()