from typing import List, Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from azure.storage.blob import BlobServiceClient
import base64
import logging
import os
import io

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str):
        self.llm = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2048,
            request_timeout=30
        )
        
        # Initialize Blob Service Client
        self.blob_service = BlobServiceClient.from_connection_string(
            os.getenv("STORAGE_CONNECTION_STRING")
        )
        self.container_name = os.getenv("CONTAINER_NAME", "vehicle-manuals")

        # Technical prompt template
        self.technical_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a technical documentation assistant for Harley-Davidson motorcycles."),
            ("human", """Provide detailed, accurate answers based on the context below. Include:
- Exact specifications (torque values, measurements, etc.)
- Step-by-step procedures when applicable
- Safety warnings and precautions
- References to diagrams/tables when relevant

Context: {context}

Question: {question}

Answer:""")
        ])

    async def generate_response(self, query: str, context: str, images: List[bytes] = None) -> Dict:
        """Generate technical response with sources and multimedia content"""
        try:
            # Process the context to extract sources, images, and tables
            sources, extracted_images, tables = self._process_context(context)
            
            # Get image bytes for multimodal processing
            image_bytes = []
            for img in extracted_images:
                if "url" in img:
                    blob_path = img["url"].split(f"{self.container_name}/")[1].split("?")[0]
                    bytes_data = await self._get_image_bytes(blob_path)
                    if bytes_data:
                        image_bytes.append(bytes_data)
            
            # Add any user-uploaded images
            if images:
                image_bytes.extend(images)
            
            # Prepare the input based on whether we have images
            if image_bytes:
                # Multimodal processing with images
                messages = self._build_multimodal_prompt(query, context, image_bytes)
                response = await self.llm.ainvoke(messages)
                answer = response.content
            else:
                # Standard technical QA processing
                chain = self.technical_prompt | self.llm
                response = await chain.ainvoke({
                    "question": query,
                    "context": context
                })
                answer = response.content
            
            return {
                "answer": answer,
                "sources": sources,
                "images": extracted_images,
                "tables": tables,
                "not_found": not (sources or extracted_images or tables)
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    def _process_context(self, context: str) -> tuple:
        """Process context to extract sources, images, and tables"""
        sources = []
        images = []
        tables = []
        
        if isinstance(context, dict):
            sources.append({
                "id": context.get("id"),
                "title": context.get("title"),
                "content": context.get("content", "")[:500] + ("..." if len(context.get("content", "")) > 500 else ""),
                "source_url": context.get("source_url"),
                "page_number": context.get("page_number")
            })
            images.extend(self._process_images(context.get("images", [])))
            tables.extend(context.get("tables", []))
        elif isinstance(context, list):
            for doc in context:
                if isinstance(doc, dict):
                    sources.append({
                        "id": doc.get("id"),
                        "title": doc.get("title"),
                        "content": doc.get("content", "")[:500] + ("..." if len(doc.get("content", "")) > 500 else ""),
                        "source_url": doc.get("source_url"),
                        "page_number": doc.get("page_number")
                    })
                    images.extend(self._process_images(doc.get("images", [])))
                    tables.extend(doc.get("tables", []))
        
        return sources, images, tables

    def _process_images(self, images: List[Dict]) -> List[Dict]:
        """Process image data to include blob path and metadata"""
        processed = []
        for img in images:
            if "url" in img:
                blob_path = img["url"].split(f"{self.container_name}/")[1].split("?")[0]
                processed.append({
                    **img,
                    "blob_path": blob_path,
                    "is_image_content": True
                })
        return processed

    async def _get_image_bytes(self, blob_path: str) -> Optional[bytes]:
        """Get image bytes from blob storage"""
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name,
                blob=blob_path
            )
            
            with io.BytesIO() as stream:
                download_stream = blob_client.download_blob(max_concurrency=4)
                for chunk in download_stream.chunks():
                    stream.write(chunk)
                return stream.getvalue()
                
        except Exception as e:
            logger.warning(f"Failed to get image bytes for {blob_path}: {str(e)}")
            return None

    def _build_multimodal_prompt(self, query: str, context: str, images: List[bytes]) -> List[dict]:
        """Construct multimodal prompt compatible with Azure OpenAI"""
        prompt = [{
            "type": "text",
            "text": f"""You are a technical documentation assistant for Harley-Davidson motorcycles.
Provide detailed, accurate answers based on the context and images provided.
Include exact specifications, procedures, and safety information when applicable.

Context: {context}

Question: {query}

Answer:"""
        }]

        # Add images as base64 URLs
        for img_bytes in images:
            if img_bytes:
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                prompt.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                })
        
        return prompt