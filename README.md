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

## Features

- **Agentic RAG** - Tool-based document exploration (not vector search)
- **Vision support** - Analyze figures, charts, and diagrams
- **PDF preprocessing** - Extract text and figures via NVIDIA Nemotron-Parse
- **Web UI** - Drag-and-drop PDF upload with progress tracking
- **LangSmith observability** - Full tracing of agent behavior

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the agent server
langgraph dev --port 2030

# (Optional) Start preprocessing UI
cd preprocessing && python server.py
```

**Chat interface:** https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2030

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

Environment variables (`.env`):

```bash
# Agent model (OpenAI-compatible API)
OPENAI_BASE_URL=<provider_url>
OPENAI_API_KEY=<api_key>
AGENT_MODEL=<model_name>

# Vision model (for figure analysis)
VLLM_BASE_URL=<vision_provider_url>
VLLM_MODEL_NAME=<vision_model_name>

# NVIDIA API (for PDF parsing)
NVIDIA_API_KEY=<nvidia_api_key>

# LangSmith (observability)
LANGSMITH_API_KEY=<langsmith_key>
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-assignment
```

## Documentation

- [EVALUATION.md](docs/EVALUATION.md) - Evaluation framework and metrics
- [MODEL_SELECTION.md](docs/MODEL_SELECTION.md) - Model selection guide
- [AGENTIC_WORKFLOW.md](docs/AGENTIC_WORKFLOW.md) - Tool usage paradigm

## Tech Stack

- **DeepAgents** - Agent framework with FilesystemBackend
- **LangGraph** - Agent deployment and state management
- **LangChain** - LLM integration
- **FastAPI** - Preprocessing web UI
- **NVIDIA Nemotron-Parse** - PDF text/figure extraction
- **LangSmith** - Observability and tracing
