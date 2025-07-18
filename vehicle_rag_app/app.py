# import streamlit as st
# from config import AzureConfig
# from services.multimodal_agent import MultimodalRAGAgent
# from services.search_service import SearchService
# from services.storage_service import StorageService
# from services.llm_service import LLMService
# from ui.components import initialize_ui, get_user_input, display_results
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# def main():
#     # Initialize UI
#     initialize_ui()
    
#     try:
#         # Initialize configuration and services
#         config = AzureConfig()
        
#         search_service = SearchService(
#             endpoint=config.SEARCH_ENDPOINT,
#             key=config.SEARCH_KEY,
#             index_name=config.INDEX_NAME
#         )
        
#         storage_service = StorageService(
#             connection_string=config.STORAGE_CONNECTION_STRING,
#             container_name=config.CONTAINER_NAME
#         )
        
#         llm_service = LLMService(
#             endpoint=config.AZURE_OPENAI_ENDPOINT,
#             api_key=config.AZURE_OPENAI_KEY,
#             api_version=config.AZURE_OPENAI_API_VERSION,
#             deployment=config.AZURE_OPENAI_CHAT_DEPLOYMENT
#         )
        
#         agent = MultimodalRAGAgent(
#             search_service=search_service,
#             llm_service=llm_service,
#             storage_service=storage_service
#         )
        
#         # Get user input
#         query, images = get_user_input()
        
#         # Process query when button is clicked
#         if st.button("Search", type="primary") and query:
#             with st.spinner("Analyzing documents and relationships..."):
#                 try:
#                     result = agent.process_query(query, images)
#                     st.session_state.result = result
#                 except Exception as e:
#                     st.error(f"Error processing query: {str(e)}")
#                     logger.error(f"Query processing failed: {str(e)}")
        
#         # Display results if available
#         if "result" in st.session_state:
#             display_results(st.session_state.result)
            
#     except Exception as e:
#         st.error(f"Application initialization failed: {str(e)}")
#         logger.error(f"Critical application error: {str(e)}")

# if __name__ == "__main__":
#     main()

import streamlit as st
from config import AzureConfig
from services.multimodal_agent import MultimodalRAGAgent
from services.search_service import SearchService
from services.storage_service import StorageService
from services.llm_service import LLMService
from ui.components import initialize_ui, get_user_input, display_results
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main_async():
    """Async version of the main function"""
    # Initialize UI
    initialize_ui()
    
    try:
        # Initialize configuration and services
        config = AzureConfig()
        
        search_service = SearchService(
            endpoint=config.SEARCH_ENDPOINT,
            key=config.SEARCH_KEY,
            index_name=config.INDEX_NAME
        )
        
        storage_service = StorageService(
            connection_string=config.STORAGE_CONNECTION_STRING,
            container_name=config.CONTAINER_NAME
        )
        
        llm_service = LLMService(
            endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            deployment=config.AZURE_OPENAI_CHAT_DEPLOYMENT
        )
        
        agent = MultimodalRAGAgent(
            search_service=search_service,
            llm_service=llm_service,
            storage_service=storage_service
        )
        
        # Get user input
        query, images = get_user_input()
        
        # Process query when button is clicked
        if st.button("Search", type="primary") and query:
            with st.spinner("Analyzing documents and relationships..."):
                try:
                    result = await agent.process_query(query, images)
                    st.session_state.result = result
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Query processing failed: {str(e)}")
        
        # Display results if available
        if "result" in st.session_state:
            display_results(st.session_state.result)
            
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        logger.error(f"Critical application error: {str(e)}")

def main():
    # Run the async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()