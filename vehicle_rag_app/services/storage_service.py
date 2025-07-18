from typing import Optional, List
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from urllib.parse import quote
import io
import logging

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self, connection_string: str, container_name: str):
        self.client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name

    def get_blob_url_with_sas(self, blob_name: str) -> str:
        """Generate accessible URL with SAS token for a blob"""
        if not blob_name:
            return ""
        
        try:
            account_name = self.client.account_name
            account_key = self.client.credential.account_key
            
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=1))
            
            encoded_blob_name = quote(blob_name)
            return f"https://{account_name}.blob.core.windows.net/{self.container_name}/{encoded_blob_name}?{sas_token}"
        except Exception as e:
            logger.warning(f"Failed to generate blob URL with SAS: {str(e)}")
            return ""

    async def download_blob_to_bytes(self, blob_name: str) -> Optional[bytes]:
        """Download blob content directly to memory"""
        try:
            blob_client = self.client.get_blob_client(container=self.container_name, blob=blob_name)
            with io.BytesIO() as stream:
                await blob_client.download_blob().readinto(stream)
                return stream.getvalue()
        except Exception as e:
            logger.warning(f"Blob download failed for {blob_name}: {str(e)}")
            return None

    async def get_image_bytes(self, image_url: str) -> Optional[bytes]:
        """Extract image bytes from URL"""
        if not image_url:
            return None
            
        try:
            # Extract blob name from URL
            parts = image_url.split(f"/{self.container_name}/")
            if len(parts) == 2:
                blob_name = parts[1].split("?")[0]  # Remove SAS token if present
                return await self.download_blob_to_bytes(blob_name)
        except Exception as e:
            logger.warning(f"Failed to get image bytes: {str(e)}")
            return None