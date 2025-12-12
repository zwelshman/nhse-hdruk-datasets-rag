"""
NHS Dataset RAG Search
A Streamlit application for searching NHS England SDE datasets using BM25 retrieval
and Anthropic Claude for intelligent question answering.
"""

import streamlit as st
import json
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import math

# Optional: Anthropic API
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Optional: BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="NHS Dataset RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional healthcare aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --nhs-blue: #005eb8;
        --nhs-dark-blue: #003087;
        --nhs-bright-blue: #0072ce;
        --nhs-light-blue: #41b6e6;
        --nhs-aqua: #00a9ce;
        --warm-yellow: #fae100;
        --surface: #f0f4f5;
        --surface-elevated: #ffffff;
        --text-primary: #212b32;
        --text-secondary: #4c6272;
        --border: #d8dde0;
    }
    
    .stApp {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(180deg, var(--surface) 0%, #e8f4f8 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--nhs-dark-blue) 0%, var(--nhs-blue) 50%, var(--nhs-bright-blue) 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 24px rgba(0, 62, 135, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(ellipse, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* Card styling */
    .result-card {
        background: var(--surface-elevated);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .result-card:hover {
        border-color: var(--nhs-light-blue);
        box-shadow: 0 4px 12px rgba(0, 94, 184, 0.1);
        transform: translateY(-1px);
    }
    
    .result-card.dataset {
        border-left: 4px solid var(--nhs-blue);
    }
    
    .result-card.field {
        border-left: 4px solid var(--nhs-aqua);
    }
    
    .result-type {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    
    .result-type.dataset {
        background: rgba(0, 94, 184, 0.1);
        color: var(--nhs-blue);
    }
    
    .result-type.field {
        background: rgba(0, 169, 206, 0.1);
        color: var(--nhs-aqua);
    }
    
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0.25rem 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .result-meta {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0.25rem 0;
    }
    
    .result-description {
        font-size: 0.9rem;
        color: var(--text-primary);
        line-height: 1.5;
        margin-top: 0.75rem;
    }
    
    .score-badge {
        display: inline-block;
        background: var(--warm-yellow);
        color: var(--text-primary);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-left: 0.5rem;
    }
    
    /* AI Response styling */
    .ai-response {
        background: linear-gradient(135deg, #f8fbff 0%, #f0f7ff 100%);
        border: 1px solid rgba(0, 94, 184, 0.2);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1.5rem 0;
        position: relative;
    }
    
    .ai-response::before {
        content: '✨';
        position: absolute;
        top: -12px;
        left: 20px;
        background: white;
        padding: 0 8px;
        font-size: 1.2rem;
    }
    
    .ai-response-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(0, 94, 184, 0.1);
    }
    
    .ai-response-header h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: var(--nhs-dark-blue);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--surface-elevated);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.1rem;
        color: var(--nhs-dark-blue);
        font-weight: 600;
    }
    
    /* Stats cards */
    .stat-card {
        background: var(--surface-elevated);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--nhs-blue);
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    /* Search input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid var(--border);
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--nhs-blue);
        box-shadow: 0 0 0 3px rgba(0, 94, 184, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--nhs-blue) 0%, var(--nhs-dark-blue) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 62, 135, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Code blocks */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(0, 94, 184, 0.05);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SearchResult:
    doc_id: str
    doc_type: str
    name: str
    text: str
    score: float
    metadata: dict
    parent_dataset: Optional[str] = None
    table: Optional[str] = None


# ============================================================================
# Search Engine
# ============================================================================

class BM25SearchEngine:
    """BM25-based search engine for the dataset catalog."""
    
    def __init__(self, search_index: list):
        self.docs = search_index
        self.texts = [doc['text'] for doc in self.docs]
        
        # Build BM25 index
        if HAS_BM25:
            tokenized = [self._tokenize(t) for t in self.texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
    
    def _tokenize(self, text: str) -> list:
        """Tokenize text for BM25."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def search(self, query: str, top_k: int = 20, doc_type: Optional[str] = None) -> list[SearchResult]:
        """Search using BM25."""
        if not self.bm25:
            return self._fallback_search(query, top_k, doc_type)
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top results
        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                doc = self.docs[idx]
                if doc_type and doc.get('type') != doc_type:
                    continue
                results.append(SearchResult(
                    doc_id=doc['id'],
                    doc_type=doc['type'],
                    name=doc['name'],
                    text=doc['text'],
                    score=score,
                    metadata=doc.get('metadata', {}),
                    parent_dataset=doc.get('parent_dataset'),
                    table=doc.get('table')
                ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _fallback_search(self, query: str, top_k: int, doc_type: Optional[str]) -> list[SearchResult]:
        """Simple keyword search fallback."""
        query_terms = set(self._tokenize(query))
        results = []
        
        for doc in self.docs:
            if doc_type and doc.get('type') != doc_type:
                continue
            
            doc_terms = set(self._tokenize(doc['text']))
            overlap = len(query_terms & doc_terms)
            
            if overlap > 0:
                score = overlap / len(query_terms) if query_terms else 0
                results.append(SearchResult(
                    doc_id=doc['id'],
                    doc_type=doc['type'],
                    name=doc['name'],
                    text=doc['text'],
                    score=score,
                    metadata=doc.get('metadata', {}),
                    parent_dataset=doc.get('parent_dataset'),
                    table=doc.get('table')
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ============================================================================
# RAG with Anthropic
# ============================================================================

def build_context(results: list[SearchResult], max_tokens: int = 4000) -> str:
    """Build context string from search results for RAG."""
    context_parts = []
    current_length = 0
    
    # Group by dataset
    datasets = {}
    fields_by_dataset = {}
    
    for r in results:
        if r.doc_type == 'dataset':
            datasets[r.name] = r
        elif r.doc_type == 'field':
            parent = r.parent_dataset or 'Unknown'
            if parent not in fields_by_dataset:
                fields_by_dataset[parent] = []
            fields_by_dataset[parent].append(r)
    
    # Add dataset descriptions
    for name, ds in datasets.items():
        text = f"## Dataset: {name}\n{ds.text}\n"
        if current_length + len(text) < max_tokens * 4:  # rough char estimate
            context_parts.append(text)
            current_length += len(text)
    
    # Add field information
    for dataset, fields in fields_by_dataset.items():
        header = f"\n### Fields in {dataset}:\n"
        if current_length + len(header) >= max_tokens * 4:
            break
        context_parts.append(header)
        current_length += len(header)
        
        for field in fields[:15]:  # Limit fields per dataset
            field_text = f"- **{field.name}** ({field.table}): {field.text}\n"
            if current_length + len(field_text) >= max_tokens * 4:
                break
            context_parts.append(field_text)
            current_length += len(field_text)
    
    return "".join(context_parts)


def query_claude(question: str, context: str, api_key: str) -> str:
    """Query Claude with RAG context."""
    if not HAS_ANTHROPIC:
        return "Error: anthropic package not installed. Run: pip install anthropic"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    system_prompt = """You are an expert on NHS England health datasets available in the Secure Data Environment (SDE). 
You help researchers understand what data is available, what fields contain, and how to use them effectively.

When answering questions:
- Be specific about dataset names and field names
- Explain what data types and coding systems are used (ICD-10, SNOMED CT, etc.)
- Mention relevant tables when discussing fields
- If you're uncertain, say so rather than guessing
- Format field names in `code style`

You have access to a curated catalog of NHS datasets including HES (Hospital Episode Statistics), 
GDPPR (GP data), MHSDS (Mental Health), mortality data, COVID datasets, and clinical audits like NICOR."""

    user_message = f"""Based on the following NHS dataset documentation, please answer this question:

**Question:** {question}

---

**Available Dataset Documentation:**

{context}

---

Please provide a clear, helpful answer based on the documentation above. If the information isn't in the documentation, say so."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text
    except anthropic.APIError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Streamlit App
# ============================================================================

@st.cache_data
def load_search_index(path: str) -> list:
    """Load and cache the search index."""
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def get_search_engine(index_path: str) -> BM25SearchEngine:
    """Get cached search engine instance."""
    index = load_search_index(index_path)
    return BM25SearchEngine(index)


def render_result_card(result: SearchResult):
    """Render a search result as a styled card."""
    type_class = result.doc_type
    
    # Build metadata line
    meta_parts = []
    if result.parent_dataset:
        meta_parts.append(f"📁 {result.parent_dataset}")
    if result.table:
        meta_parts.append(f"📋 {result.table}")
    meta_line = " · ".join(meta_parts) if meta_parts else ""
    
    # Truncate description
    description = result.text[:300] + "..." if len(result.text) > 300 else result.text
    
    html = f"""
    <div class="result-card {type_class}">
        <span class="result-type {type_class}">{result.doc_type}</span>
        <span class="score-badge">{result.score:.2f}</span>
        <div class="result-title">{result.name}</div>
        {f'<div class="result-meta">{meta_line}</div>' if meta_line else ''}
        <div class="result-description">{description}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 NHS Dataset RAG Search</h1>
        <p>Search NHS England SDE datasets with BM25 retrieval and Claude AI assistance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required for AI-powered answers. Get one at console.anthropic.com"
        )
        
        # Check for environment variable
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                st.success("Using API key from environment")
        
        st.markdown("---")
        
        # Search settings
        st.markdown("### 🔍 Search Settings")
        
        top_k = st.slider("Results to retrieve", 5, 50, 20)
        
        doc_filter = st.selectbox(
            "Filter by type",
            ["All", "Datasets only", "Fields only"]
        )
        doc_type_map = {"All": None, "Datasets only": "dataset", "Fields only": "field"}
        selected_doc_type = doc_type_map[doc_filter]
        
        st.markdown("---")
        
        # Index path
        index_path = st.text_input(
            "Search Index Path",
            value="combined_dataset_catalog_search_index.json",
            help="Path to the JSON search index file"
        )
        
        # Status
        st.markdown("---")
        st.markdown("### 📊 Status")
        
        status_items = [
            ("BM25", HAS_BM25, "rank_bm25"),
            ("Anthropic", HAS_ANTHROPIC, "anthropic"),
        ]
        
        for name, available, package in status_items:
            if available:
                st.success(f"✓ {name} available")
            else:
                st.warning(f"✗ {name} not installed (`pip install {package}`)")
    
    # Load search engine
    try:
        if Path(index_path).exists():
            engine = get_search_engine(index_path)
            index_loaded = True
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            
            n_datasets = len([d for d in engine.docs if d['type'] == 'dataset'])
            n_fields = len([d for d in engine.docs if d['type'] == 'field'])
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{n_datasets}</div>
                    <div class="stat-label">Datasets</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{n_fields:,}</div>
                    <div class="stat-label">Fields</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(engine.docs):,}</div>
                    <div class="stat-label">Total Docs</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(f"Search index not found: {index_path}")
            st.info("Please provide the path to `combined_dataset_catalog_search_index.json`")
            index_loaded = False
            engine = None
    except Exception as e:
        st.error(f"Error loading index: {e}")
        index_loaded = False
        engine = None
    
    st.markdown("---")
    
    # Search interface
    query = st.text_input(
        "🔎 Ask a question about NHS datasets",
        placeholder="e.g., What fields contain diagnosis codes in HES APC?",
        key="search_query"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    with col2:
        use_ai = st.checkbox("Use Claude AI for answer", value=True, disabled=not api_key)
    
    # Process search
    if (search_clicked or query) and query and index_loaded and engine:
        with st.spinner("Searching..."):
            results = engine.search(query, top_k=top_k, doc_type=selected_doc_type)
        
        if not results:
            st.warning("No results found. Try different keywords.")
        else:
            # AI Response
            if use_ai and api_key:
                with st.spinner("Generating AI response..."):
                    context = build_context(results)
                    ai_response = query_claude(query, context, api_key)
                
                st.markdown(f"""
                <div class="ai-response">
                    <div class="ai-response-header">
                        <h3>🤖 Claude's Answer</h3>
                    </div>
                    <div class="ai-response-content">
                        {ai_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Search results
            st.markdown(f"### 📋 Search Results ({len(results)} matches)")
            
            # Tabs for different views
            tab1, tab2 = st.tabs(["Card View", "Table View"])
            
            with tab1:
                for result in results:
                    render_result_card(result)
            
            with tab2:
                # Table view
                table_data = []
                for r in results:
                    table_data.append({
                        "Type": r.doc_type,
                        "Name": r.name,
                        "Dataset": r.parent_dataset or "-",
                        "Table": r.table or "-",
                        "Score": f"{r.score:.3f}"
                    })
                st.dataframe(table_data, use_container_width=True)
            
            # Show context (expandable)
            with st.expander("📄 View RAG Context (sent to Claude)"):
                context = build_context(results)
                st.code(context, language="markdown")
    
    # Example queries
    if not query:
        st.markdown("### 💡 Example Queries")
        
        examples = [
            "What diagnosis fields are available in HES APC?",
            "How do I link patients across different datasets?",
            "What COVID-19 vaccination data is available?",
            "Which datasets contain GP prescribing information?",
            "What fields record admission method and source?",
            "How is ethnicity coded in NHS datasets?",
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.search_query = example
                    st.rerun()


if __name__ == "__main__":
    main()
