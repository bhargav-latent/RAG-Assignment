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

## Architecture

### Why No Vector Database? Why No Chunking?

This system deliberately **avoids traditional vector store and chunking approaches** in favor of **full-page agentic retrieval**. This is not a simplification - it's an architectural decision backed by empirical research and production experience.

#### The Problem with Traditional RAG

**Chunking destroys context** - When you split a research paper into 512-token chunks:
- Tables break across chunk boundaries (row-column relationships lost)
- Formulas get separated from their explanations
- Multi-step derivations fragment across chunks
- Citations disconnect from the claims they support

**Vector similarity is too coarse** - Embeddings excel at semantic similarity but struggle with:
- Exact keyword matching ("What is the VRMSE formula?")
- Structural queries ("What is in Table 3?")
- Negation ("Which papers don't use attention?")
- Precise citations (similarity scores ≠ locations)

#### Production Evidence: LangChain's Experience

LangChain rebuilt their documentation assistant, moving **away from vector embeddings** to an agentic approach:

> *"The original Chat LangChain used vector embeddings and document chunking, which fragmented structured content, required constant reindexing, and produced vague citations. Internal teams preferred manual workflows because they needed something more thorough than just using docs."*

**Their new approach:**
- Full-page returns via API with iterative refinement
- Scan titles first, then read selected articles fully
- Pattern matching (ripgrep) + targeted file reading for code

> *"Rather than reinventing search, the team automated what already worked—observing engineers following a specific pattern and systematizing it."*

**Source:** [Rebuilding Chat LangChain](https://blog.langchain.com/rebuilding-chat-langchain/) (2025)

#### Empirical Performance: Agent-Harness-RAG Benchmark

Controlled study comparing Hybrid RAG (BM25 + Vector Search) vs FileSearch (LLM-controlled agentic retrieval) on 44 questions:

| Metric | Agentic RAG | Hybrid RAG (Vector+BM25) | Delta |
|--------|-------------|--------------------------|-------|
| Mean Accuracy | **4.67/5** | 4.20/5 | **+11%** |
| Perfect Scores | **80%** | 53% | **+27pp** |
| Median Latency | 58s | 31s | 1.9x slower |
| Median Tokens | 37,294 | 12,137 | 3.1x more |

**When Agentic RAG excels:**
- Tables and structured data (iterative extraction)
- Scattered information requiring multiple searches
- Semantic disambiguation through refinement

**When Vector RAG wins:**
- Simple keyword queries
- Cost-sensitive scenarios
- High-throughput requirements

**Source:** [Agent-Harness-RAG Benchmark](https://github.com/bhargav-latent/Agent-Harness-RAG)

#### Our Architectural Choice

For research paper Q&A, we prioritize **accuracy and citation quality over cost and latency**. This system achieves:

✅ **93.6% correctness** (mean), **perfect median scores (5/5)**
✅ **No chunking loss** - Full tables, formulas, section context preserved
✅ **Iterative refinement** - Agent searches multiple times to find scattered information
✅ **Precise citations** - Page numbers with bounding box coordinates
✅ **No reindexing** - Add papers instantly without embedding regeneration

**Trade-offs accepted:**
- ⚠️ Median latency: 40.7s (vs ~30s for vector RAG)
- ⚠️ Median tokens: 99K (vs ~12K for vector RAG)
- ⚠️ Custom tooling required (LangGraph agent orchestration)

### Components

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **PDF Parsing** | NVIDIA Nemotron Parse v1.1 | State-of-the-art structure preservation (93.99% table accuracy) |
| **Storage** | File System (Markdown) | Full page integrity, no chunking loss |
| **Retrieval** | LLM-Controlled (Agentic) | Iterative refinement, higher accuracy |
| **Agent Framework** | LangGraph | Production-grade agent orchestration |
| **Model** | Google Gemini 3 Flash | Cost-effective with strong reasoning |

**Detailed documentation:** See [docs/RAG_Architecture_Decision_Document.md](docs/RAG_Architecture_Decision_Document.md) for full architectural rationale, empirical evidence, and trade-off analysis.

### PDF Preprocessing: Why It Matters

Traditional PDF extractors (PyPDF, pdfplumber) struggle with research papers because they:
- Break table structures into disconnected text
- Lose multi-column reading order
- Strip mathematical formulas
- Fail to identify figures and captions

This system uses **NVIDIA Nemotron Parse v1.1**, a specialized Vision-Language Model designed for document understanding.

#### Nemotron Parse Capabilities

| Feature | Capability | Impact |
|---------|-----------|--------|
| **Semantic Segmentation** | Classifies titles, tables, figures, equations | Agent knows what it's reading |
| **Reading Order** | Multi-column layouts handled correctly | Coherent text extraction |
| **Table Extraction** | LaTeX format with 93.99% accuracy | Perfect table preservation |
| **Formula Support** | Mathematical notation in LaTeX | Formulas remain readable |
| **Spatial Grounding** | Bounding boxes for all elements | Precise page citations |

#### Example: What Nemotron Parse Preserves

**Input:** Research paper with complex table

**Traditional Parser Output:**
```
Self-Attention O(n²·d) O(1) Recurrent O(n·d²) O(n) Convolutional ...
```
*Reading order lost, table structure destroyed*

**Nemotron Parse Output:**
```markdown
| Layer Type | Complexity per Layer | Sequential Ops |
|------------|---------------------|----------------|
| Self-Attention | O(n²·d) | O(1) |
| Recurrent | O(n·d²) | O(n) |
| Convolutional | O(n·d²) | O(1) |

[FIGURE: fig_2, bbox: [100, 200, 400, 500]]
Caption: Scaled Dot-Product Attention mechanism...
```
*Structure preserved, spatially grounded, ready for agent consumption*

#### Why This Matters for Accuracy

When tables and formulas are preserved:
✅ Agent can answer "What is in Table 3?" accurately
✅ Mathematical notation remains readable for verification
✅ Cross-references between figures and text stay intact
✅ Citations can point to exact page locations with bounding boxes

**Technical details:** [NVIDIA Nemotron Parse Paper](https://arxiv.org/abs/2511.20478) | [Model on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)

## Features

- **Agentic RAG** - Tool-based document exploration (not vector search)
- **Vision support** - Analyze figures, charts, and diagrams
- **PDF preprocessing** - Extract text and figures via NVIDIA Nemotron-Parse
- **Web UI** - Drag-and-drop PDF upload with progress tracking
- **LangSmith observability** - Full tracing of agent behavior
- **Comprehensive Evaluation** - LLM-as-judge framework with 25 grounded questions

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

### Deployment Note

**Current setup:** Local development with `uv` and LangGraph CLI

**Docker containerization:** Not implemented in this submission due to time constraints, but LangGraph provides first-class Docker support with significant production benefits.

#### What LangGraph Docker Would Provide

**Built-in Infrastructure:**
- **Redis** - Pub-sub broker for streaming real-time agent outputs
- **PostgreSQL** - Persistent storage for threads, runs, checkpoints, and task queue
- **Health checks** - Automatic service dependency management
- **Wolfi-based images** - Smaller, more secure containers ([langgraph-cli>=0.2.11](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/cloud/deployment/custom_docker.md))

**Production Advantages:**
- **Scalability** - Deploy multiple agent instances behind load balancer
- **Portability** - Run anywhere (Kubernetes, AWS ECS, Azure Container Instances)
- **Isolation** - Separate databases per deployment using PostgreSQL schemas
- **State persistence** - Checkpoint long-running agent executions across restarts
- **Observability** - Built-in health endpoints (`/ok`) for monitoring

**Implementation Scope:**

Simple deployment via LangGraph CLI:
```bash
langgraph build              # Builds Docker image from langgraph.json
langgraph up                 # Launches API + Redis + PostgreSQL with docker-compose
```

Or custom deployment:
```bash
langgraph dockerfile         # Generate Dockerfile for manual workflows
docker run --env-file .env -p 8123:8000 my-image
```

Configuration requires:
- `langgraph.json` - Define graphs, dependencies, environment
- `REDIS_URI` - Connection to Redis instance
- `DATABASE_URI` - PostgreSQL connection string
- Multi-service orchestration via Docker Compose

**Why deprioritized:** Given assignment timeline, focused on demonstrating core competencies (agentic RAG architecture, evaluation rigor, architectural justification) over operational packaging. The system is architecturally ready for containerization.

**References:**
- [LangGraph Standalone Container Deployment](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/cloud/deployment/standalone_container.md)
- [Custom Docker Configuration](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/cloud/deployment/custom_docker.md)
- [LangGraph CLI Documentation](https://docs.langchain.com/langsmith/cli)

### Cross-Platform Compatibility (Windows/Mac/Linux)

✅ **This project works on Windows, macOS, and Linux** without modification.

#### Path Handling

**Virtual filesystem abstraction** - The agent uses Unix-style paths (`/Paper Name/document.md`) regardless of OS:
- Agent sees: `ls path: "/"` → `["/Latent_Diffusion/", "/Writing_Effective_Use_Cases/"]`
- Windows translates to: `C:\Users\...\papers\Latent_Diffusion\`
- Linux/Mac translates to: `/home/.../papers/Latent_Diffusion/`

**Why it works:**
1. All filesystem operations use `pathlib.Path` (Python's cross-platform path library)
2. DeepAgents `FilesystemBackend` with `virtual_mode=True` translates virtual paths to real OS paths automatically
3. Virtual path `/Paper/document.md` becomes `papers\Paper\document.md` on Windows or `papers/Paper/document.md` on Unix

**Example from source code:**
```python
# src/tools.py:122
super().__init__(root_dir=root_dir, virtual_mode=True)

# DeepAgents backend (filesystem.py:78)
full = (self.cwd / vpath.lstrip("/")).resolve()  # Works on all OS
```

**No manual path conversion needed** - The `/` vs `\` difference is handled transparently.

#### Known Platform-Specific Notes

**Windows:**
- Virtual environment activation: `.venv\Scripts\activate` (backslash)
- SQLite checkpoint path: `./threads.db` works (relative paths are cross-platform)
- Evaluation script: May need to install `ripgrep` manually for grep-based search (LangGraph's built-in grep tool should work without it)

**macOS/Linux:**
- Virtual environment activation: `source .venv/bin/activate` (forward slash)
- All POSIX tools (grep, find) available by default

**All platforms:**
- `uv` package manager works identically
- LangGraph server uses Python's `aiohttp`, which is cross-platform
- Environment variables via `.env` file (no platform-specific syntax)

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
# Start LangGraph server on port 2030
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

### Understanding the Agent Workflow

The agent uses a **Map → Search → Understand** approach:

1. **Map**: Uses `ls` to discover available papers
2. **Search**: Uses `grep` to find relevant content (returns line numbers)
3. **Understand**: Uses `read_file` with offset/limit to read specific sections
4. **Analyze**: Uses `read_images` to understand figures and diagrams

**Example Agent Behavior:**
```
User: "What is the attention mechanism?"

Agent:
  1. ls → discovers "Attention_Is_All_You_Need/"
  2. grep "attention mechanism" → finds matches at lines 245-260
  3. read_file(offset=245, limit=20) → reads relevant section
  4. Responds: "The attention mechanism is... [Attention_Is_All_You_Need, page 3]"
```

## Architecture

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

## Tools Available to Agent

| Tool | Purpose |
|------|---------|
| `ls` | List paper directories |
| `grep` | Search content (returns line numbers) |
| `read_file` | Read content with offset/limit |
| `glob` | Find files by pattern |
| `read_images` | Analyze figures with vision model |

## Preprocessing Pipeline

Upload PDFs via the web UI or command line:

```bash
# Web UI (port 2031)
cd preprocessing && python server.py

# Command line
python preprocessing/run.py path/to/paper.pdf
```

**Pipeline steps:**
1. Convert PDF to images (300 DPI)
2. Extract text via NVIDIA Nemotron-Parse
3. Detect and crop figures from bounding boxes
4. Generate `document.md` + `figures/`

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

## Documentation

- [EVALUATION.md](docs/EVALUATION.md) - Evaluation framework and metrics
- [MODEL_SELECTION.md](docs/MODEL_SELECTION.md) - Model selection guide
- [AGENTIC_WORKFLOW.md](docs/AGENTIC_WORKFLOW.md) - Tool usage paradigm

## Tech Stack

- **Google Gemini 3 Flash** - Latest LLM with vision capabilities
- **DeepAgents** - Agent framework with FilesystemBackend
- **LangGraph** - Agent deployment and state management
- **LangChain** - LLM integration and tool orchestration
- **FastAPI** - Preprocessing web UI
- **NVIDIA Nemotron-Parse** - PDF text/figure extraction
- **LangSmith** - Observability and tracing

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

**Issue: LangGraph server won't start**
- Check port 2030 is not in use: `lsof -i :2030`
- Verify all dependencies installed: `uv sync`
- Check `.env` file exists and has `GOOGLE_API_KEY`
