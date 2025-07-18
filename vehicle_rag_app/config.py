import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class AzureConfig:
    """Centralized configuration for Azure services"""
    def __init__(self):
        # Search Configuration
        self.SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
        self.SEARCH_KEY = os.getenv("SEARCH_KEY")
        self.INDEX_NAME = os.getenv("INDEX_NAME", "vehicle-manuals")
        
        # Storage Configuration
        self.STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
        self.CONTAINER_NAME = os.getenv("CONTAINER_NAME", "vehicle-manuals")
        
        # OpenAI Configuration
        self.AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
        
        # Performance Configuration
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        self.validate()

    def validate(self):
        """Validate required configuration"""
        required_vars = [
            "SEARCH_ENDPOINT", "SEARCH_KEY", "STORAGE_CONNECTION_STRING",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY"
        ]
        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")