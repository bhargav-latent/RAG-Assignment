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
| **Agent Framework** | LangGraph | Agent orchestration |
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

## 4. Architecture Characteristics

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

## 5. Implementation

### Phase 1: Document Processing

1. Set up Nemotron Parse v1.1 (HuggingFace or NVIDIA NIM)
2. Build PDF → Markdown conversion pipeline
3. Implement file system storage structure
4. Extract and save figures with bounding boxes

### Phase 2: Agent System

1. Implement agent tools (grep, read_file, glob, read_images)
2. Set up LangGraph for agent orchestration
3. Configure iteration limits and token budgets
4. Build citation extraction system

### Phase 3: Evaluation

1. Create question-answer test set
2. Implement LLM-as-judge evaluation
3. Track correctness, groundedness, latency, tokens
4. Monitor for runaway behavior and hallucination

---

## 6. Known Limitations

**Runaway behavior:** Complex or ambiguous queries can trigger excessive iteration (observed: up to 1.5M tokens).
- Mitigation: Token budgets and iteration limits

**Citation hallucination:** Cross-document queries show 4% hallucination rate (fabricated page numbers).
- Mitigation: Verification mechanisms for citations

**High resource variance:** Worst-case queries can take 30x longer and use 15x more tokens than median.
- Mitigation: Timeouts and cost limits per query

**Negation queries:** "Which papers don't mention X" requires exhaustive search.
- Mitigation: Explicit exhaustive search tool when negation detected

---

## 7. References

1. **NVIDIA Nemotron Parse v1.1 Paper**
   https://arxiv.org/abs/2511.20478

2. **NVIDIA Nemotron Parse v1.1 Model**
   https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1

3. **NVIDIA Technical Blog**
   https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/

4. **LangGraph Documentation**
   https://github.com/langchain-ai/langgraph

5. **Agent-Harness-RAG Benchmark**
   https://github.com/bhargav-latent/Agent-Harness-RAG
   *Comparative study of RAG approaches*

6. **Rebuilding Chat LangChain**
   https://blog.langchain.com/rebuilding-chat-langchain/
   *Production agentic retrieval experience*

---

*Document prepared for architectural reference and implementation guidance.*
