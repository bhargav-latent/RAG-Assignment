# RAG Assignment

An agentic RAG system for academic papers using tool-based exploration.

## Overview

This system uses an LLM agent with filesystem tools (grep, read_file, glob) to answer questions about academic papers. Instead of traditional vector search, the agent explores documents directly - searching for relevant content, reading specific sections, and analyzing figures.

**Key approach:** Map → Search → Understand

```
User: "What is the attention formula?"
Agent:
  1. ls → find papers
  2. grep "attention" → locate mentions
  3. read_file with offset/limit → read relevant section
  4. Respond with citation
```

## Tools Available to Agent

| Tool | Purpose |
|------|---------|
| `ls` | List paper directories |
| `grep` | Search content (returns line numbers) |
| `read_file` | Read content with offset/limit |
| `glob` | Find files by pattern |
| `read_images` | Analyze figures with vision model |

## Architecture

This system uses an agentic RAG approach where an LLM agent directly explores documents using filesystem tools.

### Approach

**Full-page retrieval** - Documents are stored as markdown files without chunking to preserve table structures, formulas, and cross-references.

**Tool-based exploration** - The agent uses grep for content search, read_file for targeted reading with offset/limit, and read_images for figure analysis.

### Agent Harness

An **agent harness** is the execution framework that connects an LLM to an environment with tools, state management, and safety boundaries.

**DeepAgents** serves as the agent harness for this system, providing:
- **Filesystem tools** - grep, read_file, glob, ls (built-in)
- **Sandboxed environment** - ReadOnlyBackend restricts agent to read-only operations
- **State management** - LangGraph checkpointing for conversation persistence
- **Tool abstraction** - Virtual filesystem paths, standardized tool interfaces

**Comparison with other agent harnesses:**

| Agent Harness | Environment | Primary Use Case | Safety Model |
|---------------|-------------|------------------|--------------|
| **DeepAgents** | Filesystem | Document exploration, coding | Sandboxed backends |
| Claude Code | Terminal + Files | Software engineering | Approval workflows |
| SWE-Agent | Git + Shell | Issue resolution | Containerized execution |
| OpenAI Codex | Interpreter | Code generation | Isolated runtime |
| Agent-Harness-RAG | Documents | RAG benchmarking | Read-only access |

**Why agent harnesses matter:**
- **Reproducible benchmarks** - Standardized tool interfaces enable fair comparisons
- **Safety enforcement** - Prevent destructive operations (e.g., ReadOnlyBackend)
- **Tool reuse** - Built-in tools (grep, read) shared across agents
- **State persistence** - Checkpointing enables multi-turn conversations

### Components

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **PDF Parsing** | NVIDIA Nemotron Parse v1.1 | Structure preservation (93.99% table accuracy) |
| **Storage** | File System (Markdown) | Full page integrity |
| **Retrieval** | LLM-Controlled (Agentic) | Iterative refinement |
| **Agent Framework** | DeepAgents | Agent orchestration with filesystem tools |
| **Model** | Google Gemini 3 Flash | Cost-effective reasoning |

See [docs/RAG_Architecture_Decision_Document.md](docs/RAG_Architecture_Decision_Document.md) for detailed architectural analysis.

### Document Processing

PDFs are converted to markdown using NVIDIA Nemotron Parse v1.1, which provides:
- Semantic segmentation (titles, tables, figures, equations)
- Table extraction in LaTeX format (93.99% accuracy)
- Mathematical formula preservation
- Figure detection with bounding boxes
- Reading order preservation for multi-column layouts

**Technical details:** [NVIDIA Nemotron Parse Paper](https://arxiv.org/abs/2511.20478) | [Model on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)

## Evaluation Results

The RAG agent achieved **perfect median scores (5/5)** for both correctness and groundedness across 25 evaluation questions covering 4 technical papers.

### Performance Summary

| Metric | Median | Mean | Notes |
|--------|--------|------|-------|
| **Correctness** | **5.0 / 5** | 4.68 / 5 | Typical question: perfect |
| **Groundedness** | **5.0 / 5** | 4.88 / 5 | Typical question: perfect |
| **Latency** | **40.7s** | 103.2s | Typical question: fast |
| **Tokens** | **99,483** | 209,733 | Typical question: efficient |
| **Cost** | **$0.056** | $0.114 | Typical question: ~6 cents |
| **Success Rate** | - | 96% | 24/25 scored ≥4/5 |

**Key Insight:** Median scores reveal the true story - **typical questions receive perfect answers (5/5)** in ~41 seconds using ~100K tokens. The higher means are driven by a few outlier questions requiring exhaustive searches (up to 1.5M tokens, 20 minutes).

### Performance by Question Type

| Category | Count | Median Correctness | Median Latency | Median Tokens |
|----------|-------|-------------------|----------------|---------------|
| **Precision** | 12 | 5/5 | 24.6s | 65K |
| **Recall** | 8 | 5/5 | 60.6s | 139K |
| **Cross-document** | 5 | 4/5 | 91.5s | 107K |

### Evaluation Methodology

- **Dataset:** 25 grounded questions across 4 technical papers
- **Categories:** Precision (formulas, values), Recall (lists, comprehensive details), Cross-document (synthesis)
- **Difficulty:** Easy (2), Medium (10), Hard (13)
- **Evaluation:** LLM-as-judge (Gemini 3 Flash) scoring correctness and groundedness (1-5 scale)
- **Token Tracking:** Deduplicated streaming API messages for accurate counting

See the full [Evaluation Report](EVALUATION_REPORT.md) for detailed analysis, including failure modes, optimization recommendations, and production readiness assessment.

## How to Use

### Deployment

**Agent framework:** DeepAgents (creates LangGraph graphs under the hood)

**Deployment:** LangGraph CLI for local development (`langgraph dev`)

**Docker support:** LangGraph provides Docker containerization for production. See [deployment documentation](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/cloud/deployment/standalone_container.md) for details.

### Platform Compatibility

This project runs on Windows, macOS, and Linux. The agent uses a virtual filesystem abstraction (`virtual_mode=True`) that translates Unix-style paths to OS-specific paths automatically via `pathlib.Path`.

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Google Gemini API key (free at [ai.google.dev](https://ai.google.dev))
- NVIDIA API key for PDF processing (optional)
- LangSmith API key for observability (optional)

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
cd RAG-Assignment

# Create virtual environment with Python 3.12 or 3.13
uv venv --python 3.13

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv sync
```

### Step 2: Configure API Keys

Create a `.env` file in the project root:

```bash
# Google Gemini API (required)
GOOGLE_API_KEY=your_google_api_key_here

# Agent Model (Gemini 3 Flash for main agent)
AGENT_MODEL=gemini-3-flash-preview

# Vision Model (Gemini 3 Flash for figure analysis)
VISION_MODEL_NAME=gemini-3-flash-preview

# NVIDIA API (optional - for PDF preprocessing)
NVIDIA_API_KEY=your_nvidia_api_key_here

# LangSmith Observability (optional - for debugging)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-assignment
```

**Getting API Keys:**
- **Google Gemini**: Visit [Google AI Studio](https://ai.google.dev) → Get API Key
- **NVIDIA**: Visit [NVIDIA API Catalog](https://build.nvidia.com/) → Sign up for free tier
- **LangSmith**: Visit [LangSmith](https://smith.langchain.com/) → Create account

### Step 3: Add Papers to the System

#### Option A: Command Line (Recommended for testing)

```bash
# Process a PDF file
python preprocessing/run.py path/to/your/paper.pdf

# The system will:
# 1. Convert PDF to high-resolution images
# 2. Extract text using NVIDIA Nemotron-Parse
# 3. Extract and save figures
# 4. Save to papers/{paper_name}/
```

#### Option B: Web UI (User-friendly)

```bash
# Start the preprocessing web interface
cd preprocessing
python server.py

# Open http://localhost:2031 in your browser
# Drag and drop PDF files to process them
```

**Output Structure:**
```
papers/
└── Paper_Name/
    ├── document.md      # Full extracted text with page markers
    └── figures/         # Extracted images (PNG format)
        ├── figure_1.png
        ├── figure_2.png
        └── ...
```

### Step 4: Start the RAG Agent Server

```bash
# Start agent server on port 2030 (DeepAgents agent deployed via LangGraph)
langgraph dev --port 2030

# You should see:
# ✓ API ready at http://127.0.0.1:2030
# ✓ Docs at http://127.0.0.1:2030/docs
```

### Step 5: Interact with the Agent

#### Option A: LangSmith Studio (Recommended)

1. Open [LangSmith Studio](https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2030)
2. Enter your questions in the chat interface
3. View agent's tool usage and reasoning in real-time

#### Option B: API Endpoint

```bash
# Using curl
curl -X POST http://127.0.0.1:2030/rag_agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "What papers are available?"}]
    },
    "config": {
      "configurable": {
        "thread_id": "1"
      }
    }
  }'
```

#### Option C: Python SDK

```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2030")

# Create a thread
thread = client.threads.create()

# Send a message
response = client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="rag_agent",
    input={"messages": [{"role": "user", "content": "List available papers"}]}
)
```

## System Diagram

```
┌─────────────────────────────────────────────────────┐
│                    LangGraph Server                  │
│  ┌───────────────────────────────────────────────┐  │
│  │              DeepAgents Framework             │  │
│  │  ┌─────────┐  ┌──────┐  ┌───────────────┐   │  │
│  │  │   LLM   │→ │Tools │→ │ReadOnlyBackend│   │  │
│  │  └─────────┘  └──────┘  └───────────────┘   │  │
│  │       ↓          ↓              ↓            │  │
│  │  [grep, read_file, glob, ls, read_images]   │  │
│  └───────────────────────────────────────────────┘  │
│                        ↓                             │
│              papers/ (sandboxed)                     │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
RAG-Assignment/
├── src/
│   ├── agent.py          # Agent configuration
│   └── tools.py          # ReadOnlyBackend + read_images
├── preprocessing/
│   ├── server.py         # FastAPI upload UI
│   ├── run.py            # PDF extraction pipeline
│   └── config.py         # Paths and settings
├── papers/               # Processed documents
│   └── {paper_name}/
│       ├── document.md   # Full extracted text
│       └── figures/      # Extracted images
├── docs/                 # Documentation
│   ├── EVALUATION.md     # Evaluation framework
│   ├── MODEL_SELECTION.md
│   └── AGENTIC_WORKFLOW.md
├── langgraph.json        # LangGraph deployment config
└── pyproject.toml        # Dependencies
```

## Configuration

### Environment Variables

All configuration is managed through the `.env` file:

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key | - |
| `AGENT_MODEL` | No | Gemini model for main agent | `gemini-3-flash-preview` |
| `VISION_MODEL_NAME` | No | Gemini model for vision tasks | `gemini-3-flash-preview` |
| `NVIDIA_API_KEY` | No | NVIDIA API for PDF parsing | - |
| `LANGSMITH_API_KEY` | No | LangSmith for observability | - |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing | `true` |
| `LANGSMITH_PROJECT` | No | LangSmith project name | `rag-assignment` |

### Available Gemini Models

You can use any of these Gemini 3 models:

- **gemini-3-flash-preview** - Fast, Pro-level intelligence (recommended)
- **gemini-3-pro-preview** - Flagship model for complex reasoning
- **gemini-3-pro-image-preview** - Highest quality image generation

### Customizing Agent Behavior

Edit [src/agent.py](src/agent.py) to customize:

```python
config = AgentConfig(
    papers_dir="papers",              # Directory containing papers
    model="gemini-3-flash-preview",   # LLM model
    checkpointer="memory",            # "memory", "sqlite", or "postgres"
    enable_vision=True,               # Enable figure analysis
    system_prompt=SYSTEM_PROMPT       # Custom instructions
)
```

## Troubleshooting

### Common Issues

**Issue: `uv: command not found`**
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env  # Reload shell
```

**Issue: `API key required for Gemini`**
- Ensure `GOOGLE_API_KEY` is set in `.env`
- Verify the key is valid at [Google AI Studio](https://ai.google.dev)

**Issue: `Model not found: gemini-3-flash-preview`**
- Your API key might not have access to Gemini 3 models
- Try using `gemini-pro` or `gemini-1.5-flash` instead

**Issue: `No papers found`**
- Add papers using `python preprocessing/run.py your-paper.pdf`
- Or use the web UI at `http://localhost:2031`
- Verify papers exist in the `papers/` directory

**Issue: PDF preprocessing fails**
- Ensure `NVIDIA_API_KEY` is set in `.env`
- Check PDF is not password-protected or corrupted
- Verify you have `poppler-utils` installed for PDF conversion

**Issue: Agent server won't start (`langgraph dev` fails)**
- Check port 2030 is not in use: `lsof -i :2030`
- Verify all dependencies installed: `uv sync`
- Check `.env` file exists and has `GOOGLE_API_KEY`
