"""RAG agent setup using DeepAgents."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv(Path(__file__).parent.parent / ".env")
from langgraph.checkpoint.memory import MemorySaver

from . import tools as tools_module
from .tools import read_images, ReadOnlyBackend


# Get project root from this file's location
_PROJECT_ROOT = Path(__file__).parent.parent


SYSTEM_PROMPT = """You are a research assistant answering questions about academic papers.

## Workflow: Map → Search → Understand

1. **Map**: Use `ls` with `path: "/"` to see available papers
2. **Search**: Use `grep` to find content (use offset/limit for large results)
3. **Understand**: Use `read_file` with offset/limit to read specific sections

## Filesystem

```
/
├── Paper Name/
│   ├── document.md      # Full paper content (use grep + offset/limit)
│   └── figures/*.png    # Extracted figures
```

## Tools

- **ls**: List papers
- **grep**: Search content (returns line numbers for targeting)
- **read_file**: Read content (use `offset` and `limit` for sections)
- **glob**: Find files by pattern
- **read_images**: Analyze figures with vision

## Key: Use offset/limit

For large documents, DON'T read the entire file. Instead:
1. `grep` to find relevant line numbers
2. `read_file` with `offset` and `limit` to read just that section

## Response

Cite sources: [paper_name, page X] or [paper_name, Figure Y]
"""


# ============================================================================
# Agent Configuration
# ============================================================================


@dataclass
class AgentConfig:
    """Configuration for the RAG agent."""

    papers_dir: str = field(default_factory=lambda: str(_PROJECT_ROOT / "papers"))
    model: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
    )
    checkpointer: Literal["memory", "sqlite", "postgres"] = "memory"
    sqlite_path: str = "./threads.db"
    postgres_uri: str | None = None
    enable_vision: bool = True
    system_prompt: str | None = None


def get_checkpointer(config: AgentConfig):
    """Get the checkpointer based on configuration."""
    if config.checkpointer == "memory":
        return MemorySaver()
    elif config.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(config.sqlite_path)
    elif config.checkpointer == "postgres":
        if not config.postgres_uri:
            raise ValueError("postgres_uri required when checkpointer='postgres'")
        from langgraph.checkpoint.postgres import PostgresSaver

        return PostgresSaver.from_conn_string(config.postgres_uri)
    else:
        raise ValueError(f"Unknown checkpointer: {config.checkpointer}")


def create_rag_agent(config: AgentConfig | None = None):
    """Create a RAG agent for research papers using DeepAgents.

    Args:
        config: Agent configuration. If None, uses defaults.

    Returns:
        Compiled LangGraph agent
    """
    from deepagents import create_deep_agent

    if config is None:
        config = AgentConfig()

    # Set papers root directory (use absolute path)
    papers_path = Path(config.papers_dir)
    if not papers_path.is_absolute():
        papers_path = _PROJECT_ROOT / config.papers_dir

    # Set PAPERS_ROOT in tools module for read_images path resolution
    tools_module.PAPERS_ROOT = papers_path

    # Initialize model (ChatOpenAI works with vLLM's OpenAI-compatible API)
    llm = ChatOpenAI(
        model=config.model,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
    )

    # Build custom tool list (filesystem tools are built-in to DeepAgents)
    custom_tools = []
    if config.enable_vision:
        custom_tools.append(read_images)

    # Create read-only backend for the papers directory
    backend = ReadOnlyBackend(root_dir=str(papers_path))

    # Get checkpointer
    checkpointer = get_checkpointer(config)

    # Create agent using DeepAgents
    agent = create_deep_agent(
        model=llm,
        tools=custom_tools,
        backend=backend,
        checkpointer=checkpointer,
        system_prompt=config.system_prompt or SYSTEM_PROMPT,
    )

    return agent


def create_rag_agent_default():
    """Create agent with default configuration (for LangGraph server)."""
    return create_rag_agent(AgentConfig())
