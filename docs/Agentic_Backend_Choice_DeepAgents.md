# Agentic Backend Choice: DeepAgents

**Date:** January 2026
**Framework:** DeepAgents by LangChain
**Version:** Latest (via PyPI)

---

## 1. Why DeepAgents

DeepAgents is a standalone library for building agents that tackle complex, multi-step tasks. Built on LangGraph and inspired by applications like **Claude Code**, **Deep Research**, and **Manus**.

### Core Philosophy

> *"Deep Agents follow a 'trust the LLM' security model—the agent can execute any action permitted by underlying tools. Security boundaries should be enforced at the tool/sandbox layer."*

This aligns perfectly with our Full-Page Agentic RAG approach: give the LLM full control over retrieval, let it iterate and refine.

---

## 2. Key Features for Our RAG System

### 2.1 Built-in FilesystemMiddleware

DeepAgents ships with filesystem tools out of the box:

| Tool | Purpose | Our Use Case |
|------|---------|--------------|
| `grep` | Text search within files | Search across parsed paper pages |
| `glob` | Pattern-based file discovery | Find papers by pattern `papers/**/*.md` |
| `read_file` | Read with pagination | Fetch full page content |
| `ls` | List directory contents | Browse available papers |
| `write_file` | Create/overwrite files | Store parsed papers |
| `edit_file` | Exact string replacements | Update metadata |

**This eliminates the need to build custom search tools from scratch.**

### 2.2 Pluggable Backends

```python
from deepagents.backends import FilesystemBackend

# Point agent at our papers directory
agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/papers")
)
```

Backend options:

| Backend | Description | Our Use |
|---------|-------------|---------|
| `FilesystemBackend` | Real disk operations | **Primary** - read parsed papers |
| `StateBackend` | Ephemeral in-memory | Scratch space during queries |
| `CompositeBackend` | Route paths to different backends | Hybrid if needed |

### 2.3 TodoListMiddleware (Planning)

Built-in planning for complex queries:

```python
# Agent automatically gets write_todos / read_todos tools
# Enables multi-step reasoning:
# 1. Search for relevant papers
# 2. Read top candidates
# 3. Extract specific information
# 4. Synthesize answer
```

For research paper queries that require:
- Cross-paper comparison
- Multi-hop reasoning
- Iterative refinement

### 2.4 Sub-Agent Support

Delegate specialized tasks:

```python
table_expert = {
    "name": "table-analyzer",
    "description": "Analyze tables and extract structured data",
    "system_prompt": "You specialize in interpreting tables from research papers...",
    "tools": [read_file, analyze_table],
}

agent = create_deep_agent(subagents=[table_expert])
```

Potential sub-agents for our system:
- **Table Analyzer**: Handle complex table queries
- **Citation Extractor**: Find and format references
- **Methodology Expert**: Deep-dive into methods sections

### 2.5 Custom Middleware

Extend with domain-specific tools:

```python
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

@tool
def search_papers(query: str, scope: str = "all") -> str:
    """Search across all parsed research papers.

    Args:
        query: Search term or regex pattern
        scope: 'all', 'abstracts', 'methods', 'results'
    """
    # Custom implementation
    pass

@tool
def get_paper_metadata(paper_id: str) -> dict:
    """Get metadata for a specific paper (title, authors, sections)."""
    pass

class RAGMiddleware(AgentMiddleware):
    tools = [search_papers, get_paper_metadata]

agent = create_deep_agent(middleware=[RAGMiddleware()])
```

---

## 3. Architecture with DeepAgents

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   DeepAgent                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Middleware Stack                        │   │
│  │                                                      │   │
│  │  ┌──────────────────┐  ┌──────────────────┐        │   │
│  │  │ TodoListMiddleware│  │ FilesystemMiddleware│      │   │
│  │  │ (Planning)       │  │ (grep/glob/read)  │        │   │
│  │  └──────────────────┘  └──────────────────┘        │   │
│  │                                                      │   │
│  │  ┌──────────────────┐  ┌──────────────────┐        │   │
│  │  │ RAGMiddleware    │  │ SubAgentMiddleware│        │   │
│  │  │ (Custom tools)   │  │ (Delegation)     │        │   │
│  │  └──────────────────┘  └──────────────────┘        │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LangGraph StateGraph                    │   │
│  │         (Compiled execution with state)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 FilesystemBackend                           │
│                                                             │
│   papers/                                                   │
│   ├── arxiv_2401_12345/                                     │
│   │   ├── metadata.json                                     │
│   │   ├── page_001.md                                       │
│   │   └── ...                                               │
│   └── ...                                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Blueprint

### 4.1 Installation

```bash
pip install deepagents
```

### 4.2 Basic Agent Setup

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Create agent with filesystem access to papers
agent = create_deep_agent(
    system_prompt="""You are a research paper assistant.

Your task is to answer questions about research papers stored in the filesystem.
Each paper is stored as markdown files (one per page) with metadata headers.

Workflow:
1. Use grep to search for relevant terms across papers
2. Use read_file to examine promising pages in full
3. If results are insufficient, refine your search
4. Synthesize answers with precise citations (paper_id, page number)

Always cite your sources in the format: [paper_id, page X]
""",
    backend=FilesystemBackend(root_dir="./papers"),
)
```

### 4.3 With Custom RAG Tools

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
import json

@tool
def list_papers() -> str:
    """List all available research papers with their titles and IDs."""
    # Returns paper index
    pass

@tool
def get_paper_toc(paper_id: str) -> str:
    """Get the table of contents for a specific paper.

    Args:
        paper_id: The unique identifier for the paper

    Returns:
        JSON with sections, page numbers, and paper metadata
    """
    pass

@tool
def search_abstracts(query: str) -> str:
    """Search only within paper abstracts for faster initial filtering.

    Args:
        query: Search term

    Returns:
        List of matching papers with abstract snippets
    """
    pass

class RAGToolsMiddleware(AgentMiddleware):
    tools = [list_papers, get_paper_toc, search_abstracts]

agent = create_deep_agent(
    system_prompt="...",
    backend=FilesystemBackend(root_dir="./papers"),
    middleware=[RAGToolsMiddleware()],
)
```

### 4.4 With Sub-Agents

```python
table_subagent = {
    "name": "table-expert",
    "description": "Use this agent for questions about tables, charts, or structured data in papers",
    "system_prompt": """You are an expert at analyzing tables from research papers.
    Given a table in markdown/LaTeX format, you can:
    - Extract specific values
    - Compare rows/columns
    - Summarize findings
    - Identify trends""",
}

methodology_subagent = {
    "name": "methodology-expert",
    "description": "Use this agent for detailed questions about research methodology",
    "system_prompt": """You are an expert at analyzing research methodology.
    You can explain experimental setups, statistical methods, and research design.""",
}

agent = create_deep_agent(
    system_prompt="...",
    backend=FilesystemBackend(root_dir="./papers"),
    subagents=[table_subagent, methodology_subagent],
)
```

---

## 5. Query Execution Flow

```
User: "What attention mechanism does the Transformer paper propose?"
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: PLAN (TodoListMiddleware)                          │
│                                                             │
│ Agent creates todos:                                        │
│ - [ ] Search for "Transformer" and "attention"             │
│ - [ ] Read relevant pages                                  │
│ - [ ] Extract mechanism details                            │
│ - [ ] Formulate answer with citations                      │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: SEARCH (FilesystemMiddleware.grep)                 │
│                                                             │
│ grep("attention mechanism", "./papers/**/*.md")            │
│                                                             │
│ Results:                                                    │
│ - arxiv_1706_03762/page_003.md: "Scaled Dot-Product..."    │
│ - arxiv_1706_03762/page_004.md: "Multi-Head Attention..."  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: READ (FilesystemMiddleware.read_file)              │
│                                                             │
│ read_file("./papers/arxiv_1706_03762/page_003.md")         │
│ read_file("./papers/arxiv_1706_03762/page_004.md")         │
│                                                             │
│ → Full page content with tables, equations, figures        │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: SYNTHESIZE                                          │
│                                                             │
│ Agent formulates answer:                                    │
│                                                             │
│ "The Transformer paper proposes two attention mechanisms:   │
│                                                             │
│ 1. **Scaled Dot-Product Attention** [page 3]               │
│    Attention(Q,K,V) = softmax(QK^T/√d_k)V                  │
│                                                             │
│ 2. **Multi-Head Attention** [page 4]                       │
│    Parallel attention layers with different projections    │
│                                                             │
│ Citations: [arxiv_1706_03762, pages 3-4]"                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration Options

### 6.1 Model Selection

```python
from langchain.chat_models import init_chat_model

# Default: Claude Sonnet 4.5
agent = create_deep_agent(...)

# Override with other models
model = init_chat_model("openai:gpt-4o")
agent = create_deep_agent(model=model, ...)

# Local models
model = init_chat_model("ollama:llama3")
agent = create_deep_agent(model=model, ...)
```

### 6.2 Token Management

```python
# SummarizationMiddleware auto-triggers at 170k tokens
# Prevents context overflow on long research sessions
```

### 6.3 Human-in-the-Loop (Optional)

```python
# Require approval for certain operations
agent = create_deep_agent(
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]},
    }
)
```

---

## 7. Why DeepAgents Fits Our Requirements

| Requirement | DeepAgents Capability |
|-------------|----------------------|
| **Full-page retrieval** | FilesystemBackend + read_file |
| **No vector store** | Pure filesystem, grep-based search |
| **Iterative refinement** | LangGraph state machine + planning |
| **Research paper structure** | Custom middleware for paper-aware tools |
| **Accurate citations** | Full page context enables precise references |
| **Complex queries** | Sub-agents for specialized tasks |
| **Extensibility** | Middleware architecture for custom tools |

---

## 8. References

- **DeepAgents GitHub**: https://github.com/langchain-ai/deepagents
- **DeepAgents Documentation**: https://docs.langchain.com/oss/python/deepagents/overview
- **DeepAgents PyPI**: https://pypi.org/project/deepagents/
- **LangChain 1.0 Announcement**: https://blog.langchain.com/langchain-langgraph-1dot0/
- **DataCamp Tutorial**: https://www.datacamp.com/tutorial/deep-agents

---

*Document prepared for architectural alignment before implementation.*
