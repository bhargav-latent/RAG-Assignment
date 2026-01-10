# RAG Architecture Decision Document

**Date:** January 2026
**Domain:** Research Paper Question-Answering

---

## 1. Overview

This system implements an agentic RAG approach where an LLM agent explores documents using filesystem tools. Documents are stored as full-page markdown files without chunking.

### Architecture Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Parsing** | NVIDIA Nemotron Parse v1.1 | Document structure extraction |
| **Storage** | File System (Markdown) | Document storage |
| **Retrieval** | LLM-Controlled (Agentic) | Content discovery and extraction |
| **Agent Framework** | DeepAgents | Agent orchestration with filesystem tools |
| **Model** | Google Gemini 3 Flash | Query understanding and synthesis |

---

## 2. Document Processing

### PDF Parsing

PDFs are converted to markdown using NVIDIA Nemotron Parse v1.1.

**Specifications:**
- Parameters: 885M (600M vision encoder + 250M decoder)
- Architecture: ViT-H encoder + mBART decoder
- Speed: ~4-5 pages/second
- Table Accuracy (S-TEDS): 93.99%
- Output: Structured Markdown + Bounding Boxes

**Capabilities:**
- Semantic segmentation (titles, headers, tables, figures, captions, footnotes)
- Reading order preservation for multi-column layouts
- Table extraction in LaTeX format with multirow/multicolumn support
- Mathematical formula preservation in LaTeX
- Bounding box coordinates for spatial grounding

**References:**
- Paper: https://arxiv.org/abs/2511.20478
- Model: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1
- Blog: https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/

### Storage Structure

```
papers/
├── Paper_Name_1/
│   ├── document.md      # Full extracted text with page markers
│   └── figures/         # Extracted images
│       ├── figure_1.png
│       └── figure_2.png
└── Paper_Name_2/
    └── ...
```

### Page Format Example

```markdown
---
paper: "Attention Is All You Need"
page: 3
---

## 3.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention"...

| Layer Type | Complexity per Layer | Sequential Ops |
|------------|---------------------|----------------|
| Self-Attention | O(n²·d) | O(1) |
| Recurrent | O(n·d²) | O(n) |

[FIGURE: fig_2, bbox: [100, 200, 400, 500]]
Caption: Scaled Dot-Product Attention mechanism...
```

---

## 3. Retrieval System

### Agent Tools

| Tool | Signature | Purpose |
|------|-----------|---------|
| `ls` | `(path) → [files]` | List directories and files |
| `grep` | `(pattern, path) → [matches]` | Search content with regex |
| `read_file` | `(path, offset?, limit?) → content` | Read file or section |
| `glob` | `(pattern) → [paths]` | Find files by pattern |
| `read_images` | `(paths) → analysis` | Analyze figures with vision model |

### Retrieval Process

```
1. Search: Use grep to locate relevant content
2. Read: Fetch full pages or sections with read_file
3. Iterate: Refine searches as needed
4. Synthesize: Construct answer with citations
```

### System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      USER QUERY                              │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENTIC LLM                               │
│         (Query Understanding & Retrieval Control)            │
└──────────────────────────┬───────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  grep    │    │  glob    │    │  read    │
    │ (search) │    │ (find)   │    │ (fetch)  │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    FILE SYSTEM                               │
│                                                              │
│   papers/                                                    │
│   ├── Paper_Name_1/                                          │
│   │   ├── document.md                                        │
│   │   └── figures/                                           │
│   └── Paper_Name_2/                                          │
│       └── ...                                                │
└──────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌──────────────────────────────────────────────────────────────┐
│               NEMOTRON PARSE v1.1                            │
│   - Semantic segmentation                                    │
│   - Reading order preservation                               │
│   - LaTeX table/formula extraction                           │
│   - Bounding box coordinates                                 │
└──────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌──────────────────────────────────────────────────────────────┐
│                    PDF INPUT                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Agent Harness: DeepAgents

### What is an Agent Harness?

An **agent harness** is the execution framework that enables LLMs to interact with environments through:
- **Tool provisioning** - Standardized actions the agent can invoke
- **Environment abstraction** - Filesystem, APIs, databases, shells
- **Safety enforcement** - Sandboxing, permission controls, resource limits
- **State management** - Conversation persistence, checkpointing
- **Observability** - Logging, tracing, monitoring

### DeepAgents as the Agent Harness

DeepAgents provides the infrastructure for this RAG system:

**Built-in capabilities:**
- Filesystem tools (grep, read_file, glob, ls, write_file, edit_file)
- Multiple backend types (FilesystemBackend, StateBackend, CompositeBackend)
- Virtual filesystem mode (platform-independent paths)
- Integration with LangGraph (state graphs, checkpointing)
- Tool middleware system (custom tool injection)

**Safety implementation:**
```python
# ReadOnlyBackend prevents destructive operations
backend = ReadOnlyBackend(root_dir="/path/to/papers")
# Agent can read, search, glob - but cannot write, edit, delete
```

**State management:**
```python
# LangGraph checkpointing for multi-turn conversations
checkpointer = MemorySaver()  # or SqliteSaver, PostgresSaver
agent = create_deep_agent(backend=backend, checkpointer=checkpointer)
```

### Comparison with Other Agent Harnesses

| Agent Harness | Tools Provided | Safety Model | Use Case |
|---------------|----------------|--------------|----------|
| **DeepAgents** | Filesystem, custom | Sandboxed backends | Document exploration, coding |
| **Claude Code** | Terminal, files, git | Human approval for writes | Software engineering |
| **SWE-Agent** | Git, shell, editor | Docker containers | GitHub issue resolution |
| **OpenAI Codex** | Python interpreter | Isolated runtime | Code generation/execution |
| **Agent-Harness-RAG** | Document tools | Read-only | RAG benchmarking |
| **LangChain Agents** | Configurable | Tool-level validation | General purpose |

### Role in Benchmarking

Agent harnesses enable **reproducible benchmarks** by:

1. **Standardized interfaces** - All agents use same tool signatures (grep, read_file)
2. **Controlled environments** - Identical filesystem structure across runs
3. **Fair comparisons** - Same tools available to different models
4. **Safety guarantees** - Read-only prevents benchmark contamination

**Example: Agent-Harness-RAG benchmark**
- 44 questions on 3-document corpus
- All agents get same tools: search, read_page, list_papers
- Enables comparison: Hybrid RAG vs Agentic FileSearch
- Results: Agentic 4.67/5 vs Hybrid 4.20/5 (11% improvement)

### Implementation in This System

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Custom read-only backend for safety
class ReadOnlyBackend(FilesystemBackend):
    """Filesystem backend that prevents writes."""
    def write_file(self, path, content):
        raise PermissionError("Write operations disabled")
    def edit_file(self, path, old, new):
        raise PermissionError("Edit operations disabled")

# Create agent with read-only harness
backend = ReadOnlyBackend(root_dir="./papers", virtual_mode=True)
agent = create_deep_agent(
    backend=backend,
    tools=[read_images],  # Add custom vision tool
    checkpointer=MemorySaver(),
)
```

**Benefits for this system:**
- Agent cannot modify source documents (read-only safety)
- Virtual paths work on Windows/Mac/Linux
- Built-in tools (grep, read) are production-tested
- Checkpointing enables conversation history
- Easy to extend with custom tools (read_images)

---

## 5. Architecture Characteristics

### Full-Page Storage

Documents are stored as complete markdown files without chunking. This preserves:
- Table structures (row-column relationships)
- Formula context (equations with explanations)
- Section continuity (multi-paragraph discussions)
- Citation context (references with claims)

### Iterative Retrieval

The agent can:
- Search multiple times with refined queries
- Read sections based on initial search results
- Navigate by section headings
- Analyze figures when text references them

### Resource Profile

Based on evaluation with 25 questions across 4 papers:

**Latency:**
- Median: 40.7s
- P25: 24.0s
- P75: 87.5s
- Max: 1,202.6s (20 minutes)

**Token Usage:**
- Median: 99,483 tokens
- P25: 54,017 tokens
- P75: 166,336 tokens
- Max: 1,567,130 tokens

**Cost (Gemini 3 Flash):**
- Median: $0.056 per query
- P25: $0.029 per query
- P75: $0.091 per query
- Max: $0.79 per query

**Accuracy:**
- Correctness: 93.6% mean, 5/5 median
- Groundedness: 97.6% mean, 5/5 median
- Success rate: 96% (≥4/5 score)

### Observed Behavior

**Precision queries** (formulas, specific values):
- Fast (median 24.6s)
- Efficient (median 65K tokens)
- High accuracy (98.4%)

**Recall queries** (list items, comprehensive coverage):
- Variable latency (median 60.6s, outliers to 20 minutes)
- Higher token usage (median 139K tokens)
- Moderate accuracy (90%)
- Risk: Ambiguous questions can trigger excessive iteration

**Cross-document queries** (synthesis across papers):
- Moderate latency (median 91.5s)
- Moderate token usage (median 107K tokens)
- Lower accuracy (88%)
- Risk: Citation hallucination observed (4% of questions)

---

## 6. Implementation

### Phase 1: Document Processing

1. Set up Nemotron Parse v1.1 (HuggingFace or NVIDIA NIM)
2. Build PDF → Markdown conversion pipeline
3. Implement file system storage structure
4. Extract and save figures with bounding boxes

### Phase 2: Agent System

1. Implement custom tools (read_images for vision)
2. Set up DeepAgents with ReadOnlyBackend
3. Configure iteration limits and token budgets
4. Build citation extraction system

### Phase 3: Evaluation

1. Create question-answer test set
2. Implement LLM-as-judge evaluation
3. Track correctness, groundedness, latency, tokens
4. Monitor for runaway behavior and hallucination

---

## 7. Known Limitations

**Runaway behavior:** Complex or ambiguous queries can trigger excessive iteration (observed: up to 1.5M tokens).
- Mitigation: Token budgets and iteration limits

**Citation hallucination:** Cross-document queries show 4% hallucination rate (fabricated page numbers).
- Mitigation: Verification mechanisms for citations

**High resource variance:** Worst-case queries can take 30x longer and use 15x more tokens than median.
- Mitigation: Timeouts and cost limits per query

**Negation queries:** "Which papers don't mention X" requires exhaustive search.
- Mitigation: Explicit exhaustive search tool when negation detected

---

## 8. References

1. **NVIDIA Nemotron Parse v1.1 Paper**
   https://arxiv.org/abs/2511.20478

2. **NVIDIA Nemotron Parse v1.1 Model**
   https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1

3. **NVIDIA Technical Blog**
   https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/

4. **DeepAgents Documentation**
   https://github.com/langchain-ai/deepagents

5. **Agent-Harness-RAG Benchmark**
   https://github.com/bhargav-latent/Agent-Harness-RAG
   *Comparative study of RAG approaches*

6. **Rebuilding Chat LangChain**
   https://blog.langchain.com/rebuilding-chat-langchain/
   *Production agentic retrieval experience*

---

*Document prepared for architectural reference and implementation guidance.*
