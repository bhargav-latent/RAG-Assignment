# Tool Specification: Bare DeepAgents + READ_IMAGES

**Date:** January 2026
**Approach:** Minimal tooling - leverage DeepAgents built-ins, add only what's missing

---

## 1. Philosophy

> Keep it simple. DeepAgents provides battle-tested filesystem tools. We add only one custom tool for image understanding.

**Why minimal?**
- Less code to maintain
- Fewer failure points
- LLM has cleaner tool selection
- DeepAgents tools mirror Claude Code (production-proven)

---

## 2. Built-in Tools (DeepAgents FilesystemMiddleware)

These come free with DeepAgents. No implementation needed.

| Tool | Signature | Purpose |
|------|-----------|---------|
| `grep` | `grep(pattern, path, glob?)` | Regex search across files (ripgrep-powered) |
| `glob` | `glob(pattern, path?)` | Find files by pattern (`**/*.md`) |
| `read_file` | `read_file(path, offset?, limit?)` | Read file content with pagination |
| `write_file` | `write_file(path, content)` | Create/overwrite files |
| `edit_file` | `edit_file(path, old, new)` | String replacement in files |
| `ls` | `ls(path)` | List directory contents |

### How Agent Uses Them for RAG

```
Query: "What does Figure 3 in the Transformer paper show?"

Agent:
1. grep("Transformer", "./papers/") → finds paper directory
2. grep("Figure 3", "./papers/arxiv_1706/") → finds page_005.md
3. read_file("./papers/arxiv_1706/page_005.md") → gets full page
4. Sees: [IMAGE: figures/fig_003.png]
5. read_images("./papers/arxiv_1706/figures/fig_003.png", "Describe this figure")
6. Synthesizes answer with visual context
```

---

## 3. Custom Tool: READ_IMAGES

### 3.1 Purpose

Research papers contain figures, diagrams, charts, and tables that are extracted as images by Nemotron Parse. The agent needs to "see" these to answer visual questions.

### 3.2 Specification

```python
@tool
def read_images(
    image_paths: list[str],
    query: str,
    detail: str = "high"
) -> str:
    """Analyze images using a vision model.

    Use this tool when you need to understand figures, diagrams,
    charts, or tables that are stored as images.

    Args:
        image_paths: List of paths to image files (supports PNG, JPG, WEBP)
        query: Question or instruction about the image(s)
        detail: Image detail level - "low" (faster) or "high" (more accurate)

    Returns:
        Vision model's analysis of the image(s) based on the query

    Examples:
        - read_images(["fig_001.png"], "What does this architecture diagram show?")
        - read_images(["table_2.png"], "Extract the values from this table")
        - read_images(["fig_a.png", "fig_b.png"], "Compare these two charts")
    """
```

### 3.3 Implementation

```python
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Initialize vision model (OpenAI-compatible)
vision_model = ChatOpenAI(
    model="gpt-4o",  # or any OpenAI-compatible vision model
    max_tokens=4096,
)

@tool
def read_images(
    image_paths: list[str],
    query: str,
    detail: str = "high"
) -> str:
    """Analyze images using a vision model.

    Use this tool when you need to understand figures, diagrams,
    charts, or tables that are stored as images.

    Args:
        image_paths: List of paths to image files (PNG, JPG, WEBP)
        query: Question or instruction about the image(s)
        detail: "low" (faster, 85 tokens) or "high" (more accurate, more tokens)

    Returns:
        Vision model's analysis of the image(s)
    """
    content = [{"type": "text", "text": query}]

    for path in image_paths:
        image_path = Path(path)
        if not image_path.exists():
            return f"Error: Image not found at {path}"

        # Read and encode image
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_map.get(suffix, "image/png")

        # Add image to content (OpenAI format)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}",
                "detail": detail,
            },
        })

    # Call vision model
    message = HumanMessage(content=content)
    response = vision_model.invoke([message])

    return response.content
```

### 3.4 Configuration Options

```python
# Option 1: OpenAI GPT-4o
vision_model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Option 2: Azure OpenAI
from langchain_openai import AzureChatOpenAI
vision_model = AzureChatOpenAI(
    deployment_name="gpt-4o",
    api_version="2024-02-15-preview",
)

# Option 3: Any OpenAI-compatible endpoint (vLLM, Ollama, etc.)
vision_model = ChatOpenAI(
    model="llava-v1.6",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Option 4: Other LangChain vision models
from langchain_anthropic import ChatAnthropic
vision_model = ChatAnthropic(model="claude-sonnet-4-20250514")

from langchain_google_genai import ChatGoogleGenerativeAI
vision_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
```

### 3.5 Token Costs

| Detail Level | Tokens per Image | Use When |
|--------------|------------------|----------|
| `low` | ~85 tokens | Quick overview, simple diagrams |
| `high` | ~765 tokens (512px) to ~1105 tokens (2048px) | Detailed analysis, tables, complex figures |

---

## 4. Complete Tool Set Summary

| Tool | Source | Purpose |
|------|--------|---------|
| `grep` | DeepAgents | Search text in files |
| `glob` | DeepAgents | Find files by pattern |
| `read_file` | DeepAgents | Read file content |
| `write_file` | DeepAgents | Write files |
| `edit_file` | DeepAgents | Edit files |
| `ls` | DeepAgents | List directories |
| **`read_images`** | **Custom** | **Vision model for figures/tables** |

**Total: 6 built-in + 1 custom = 7 tools**

---

## 5. Integration with DeepAgents

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware

# Define custom middleware with read_images tool
class VisionMiddleware(AgentMiddleware):
    tools = [read_images]

# Create agent
agent = create_deep_agent(
    system_prompt="""You are a research paper assistant.

You have access to parsed research papers stored as markdown files.
Each paper directory contains:
- page_XXX.md files (text content per page)
- figures/ directory with extracted images

When answering questions about figures, charts, or visual elements:
1. First use grep/read_file to find references to the figure
2. Then use read_images to analyze the actual image file

Always cite your sources: [paper_id, page X] or [paper_id, Figure Y]
""",
    backend=FilesystemBackend(root_dir="./papers"),
    middleware=[VisionMiddleware()],
)
```

---

## 6. Example Query Flow

```
User: "Explain the multi-head attention diagram in the Transformer paper"

Agent thinks:
├── grep("multi-head attention", "./papers/")
│   └── Found: arxiv_1706/page_004.md
│
├── read_file("./papers/arxiv_1706/page_004.md")
│   └── Content mentions: [IMAGE: figures/fig_002.png - Multi-Head Attention]
│
├── read_images(
│       ["./papers/arxiv_1706/figures/fig_002.png"],
│       "Describe this multi-head attention architecture diagram in detail"
│   )
│   └── "The diagram shows parallel attention layers (heads) with..."
│
└── Synthesize:
    "The multi-head attention mechanism in the Transformer paper
    [arxiv_1706, page 4, Figure 2] shows..."
```

---

## 7. File Structure After Parsing

```
papers/
├── index.json                      # Paper metadata index
├── arxiv_1706_03762/               # "Attention Is All You Need"
│   ├── metadata.json               # Title, authors, sections
│   ├── page_001.md                 # Abstract, intro
│   ├── page_002.md
│   ├── page_003.md
│   ├── page_004.md                 # Contains [IMAGE: figures/fig_002.png]
│   ├── ...
│   └── figures/
│       ├── fig_001.png             # Architecture overview
│       ├── fig_002.png             # Multi-head attention
│       ├── table_001.png           # Performance comparison
│       └── ...
└── arxiv_2401_12345/
    └── ...
```

---

## 8. References

- [LangChain Multimodal Inputs](https://python.langchain.com/v0.2/docs/how_to/multimodal_inputs/)
- [ChatOpenAI Documentation](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/images-vision)
- [DeepAgents GitHub](https://github.com/langchain-ai/deepagents)

---

*Document prepared for implementation alignment.*
