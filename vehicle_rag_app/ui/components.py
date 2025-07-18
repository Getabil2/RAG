import streamlit as st
from typing import List, Dict
from models.documents import *
from models.responses import Relationship, RAGResponse
from ui.theme import DARK_THEME_CSS
def initialize_ui():
    """Initialize the UI with custom theme"""
    st.set_page_config(
        page_title="Vehicle Technical Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    st.title("🏍️ Vehicle Technical Documentation Assistant")
    st.markdown("Search through technical manuals with enhanced content relationships")

def get_user_input() -> tuple:
    """Get user input including query and any uploaded images"""
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your technical question:",
            placeholder="E.g., 'What is the torque specification for cylinder head bolts?'",
            key="query_input"
        )
    with col2:
        uploaded_image = st.file_uploader(
            "Upload image:",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
    
    images = []
    if uploaded_image:
        images.append(uploaded_image.read())
    
    return query, images

def display_relationships(rels: List[Relationship], content_map: Dict):
    """Display content relationships in an organized way"""
    if not rels:
        return
    
    with st.expander("🔗 Content Relationships", expanded=False):
        for rel in rels:
            source = content_map.get(rel.source_type, {}).get(rel.source_id)
            target = content_map.get(rel.target_type, {}).get(rel.target_id)
            
            if source and target:
                cols = st.columns([1, 1])
                with cols[0]:
                    display_content_card(source, "Source")
                with cols[1]:
                    display_content_card(target, "Related")
                st.markdown("---")

def display_content_card(content: any, title: str):
    """Unified content display"""
    content_type = getattr(content, "type", type(content).__name__.lower())
    
    st.markdown(f"""
    <div class="content-card {content_type}-card">
        <h4>{title}: {content_type.title()}</h4>
        {_get_content_preview(content)}
        {_get_content_link(content)}
    </div>
    """, unsafe_allow_html=True)

def _get_content_preview(content) -> str:
    """Get preview text for different content types"""
    if isinstance(content, DocumentChunk):
        return f"<p>{content.text[:200]}...</p>"
    elif isinstance(content, DocumentImage):
        return f"<p>Page {content.page_number}: {content.caption}</p>"
    elif isinstance(content, DocumentTable):
        return f"<p>{content.caption} ({content.row_count}x{content.column_count})</p>"
    elif isinstance(content, DocumentWarning):
        return f"<p>{content.text[:200]}...</p>"
    return ""

def _get_content_link(content) -> str:
    """Get link for different content types"""
    if hasattr(content, "source_url") and content.source_url:
        return f"<a href='{content.source_url}' target='_blank'>View Source</a>"
    elif hasattr(content, "url") and content.url:
        return f"<a href='{content.url}' target='_blank'>View Image</a>"
    return ""

def display_table_content(table: DocumentTable):
    """Render table content with proper formatting"""
    try:
        if table.content_markdown:
            # Parse markdown table
            lines = table.content_markdown.split('\n')
            if len(lines) >= 3:  # Minimum for markdown table
                st.markdown(f"**{table.caption}** (Page {table.page_number})")
                st.markdown(table.content_markdown)
                return
        
        # Fallback to simple display
        if table.content:
            st.markdown(f"**{table.caption}** (Page {table.page_number})")
            st.text(table.content)
    except Exception as e:
        st.text("Table content could not be displayed")

def display_results(result: RAGResponse):
    """Render results in Streamlit UI with enhanced relationship support"""
    # Create content maps for relationship resolution
    content_maps = {
        "chunk": {f"chunk-{chunk.chunk_index}": chunk for chunk in result.chunks},
        "image": {img.content_id: img for img in result.images},
        "table": {table.content_id: table for table in result.tables},
        "warning": {warning.content_id: warning for warning in result.warnings}
    }
    
    # Display answer
    st.markdown(f"""
    <div class="answer-section">
        <h3>Answer</h3>
        {result.answer}
    </div>
    """, unsafe_allow_html=True)

    # Display relationships if any
    display_relationships(result.relationships, content_maps)

    # Display content in tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Documents", "⚠️ Warnings", "📊 Tables", "📷 Images"])

    with tab1:
        if result.sources:
            for source in result.sources:
                st.markdown(f"""
                <div class="document-card">
                    <h4>{source.title} (Page {source.page_number})</h4>
                    <p>{source.content}</p>
                    <a href="{source.source_url}" target="_blank">View Full Document</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No document references found")

    with tab2:
        if result.warnings:
            for warning in result.warnings:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>Page {warning.page_number} - {warning.severity.title()} Warning</h4>
                    <p>{warning.text}</p>
                    <p><small>Context: {warning.context[:200]}...</small></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No warnings found")

    with tab3:
        if result.tables:
            for table in result.tables:
                with st.expander(f"{table.caption} (Page {table.page_number})", expanded=False):
                    display_table_content(table)
        else:
            st.info("No tables found")

    with tab4:
        if result.images:
            cols = st.columns(2)
            for i, img in enumerate(result.images):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="image-card">
                        <p><b>Page {img.page_number}:</b> {img.caption}</p>
                        <img src="{img.url}" style="max-width: 100%; max-height: 300px;">
                        <p><a href="{img.url}" target="_blank">View Full Image</a></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No images found")

    if result.not_found:
        st.warning("No relevant documents found. The answer is based on general knowledge.")