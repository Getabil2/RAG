DARK_THEME_CSS = """
<style>
    /* Main background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Text input */
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2e59d9;
        color: white;
        border: none;
    }
    
    /* Expanders */
    .stExpander {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* Document cards */
    .document-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2e59d9;
    }
    
    /* Image cards */
    .image-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #333;
    }
    
    /* Warning cards */
    .warning-card {
        background-color: #2d1a1a;
        border-left: 4px solid #d92e2e;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    
    /* Table cards */
    .table-card {
        background-color: #1a2d1a;
        border-left: 4px solid #2ed92e;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    
    /* Relationship cards */
    .relationship-card {
        background-color: #1a1a2d;
        border-left: 4px solid #2e2ed9;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    
    /* Links */
    a {
        color: #2e59d9 !important;
    }
    
    /* Answer section */
    .answer-section {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 5px solid #2e59d9;
        margin-bottom: 1.5rem;
    }
    
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    th, td {
        border: 1px solid #444;
        padding: 8px;
        text-align: left;
    }
    
    th {
        background-color: #2a2a2a;
    }
    
    tr:nth-child(even) {
        background-color: #252525;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #252525;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        margin-right: 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2e59d9;
    }
</style>
"""