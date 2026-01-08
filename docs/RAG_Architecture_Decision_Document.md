# RAG Architecture Decision Document
## Full-Page Agentic Retrieval for Research Papers

**Date:** January 2026
**Domain:** Research Paper Question-Answering
**Priority:** Accuracy & Reliability > Cost & Latency

---

## 1. Executive Summary

This document outlines the architectural decision to implement a **Full-Page Agentic RAG system** for research paper retrieval, deliberately avoiding traditional vector store and chunking approaches. This choice is backed by empirical research demonstrating superior accuracy in agentic retrieval systems.

### Architecture at a Glance

| Component | Choice | Rationale |
|-----------|--------|-----------|
| PDF Parsing | NVIDIA Nemotron Parse v1.1 | State-of-the-art structure preservation |
| Storage | File System (Markdown) | Full page integrity, no chunking loss |
| Retrieval | LLM-Controlled (Agentic) | Iterative refinement, higher accuracy |
| Vector Store | **None** | Eliminates embedding-induced information loss |

---

## 2. The Problem with Traditional RAG

### 2.1 Chunking Destroys Context

Traditional RAG pipelines chunk documents into 256-1024 token segments. For research papers, this causes:

- **Table fragmentation**: Row-column relationships broken across chunks
- **Section splitting**: Methods described across multiple chunks lose coherence
- **Citation loss**: References separated from the claims they support
- **Formula isolation**: Mathematical notation stripped of surrounding explanation

> *"Chunking destroys structural context, reindexing overhead slows iteration, and similarity scores produce vague citations."*
> — LangChain Engineering Team, "Rebuilding Chat LangChain" (2025)

### 2.2 Embedding Limitations for Structured Content

Vector embeddings excel at semantic similarity but struggle with:

- **Exact keyword matching**: Technical terms, author names, specific values
- **Structural queries**: "What is in Table 3?" or "First equation in Section 4"
- **Negation**: "Which papers don't use attention mechanisms?"
- **Precise citations**: Embeddings return similarity scores, not locations

---

## 3. Evidence Supporting Agentic Retrieval

### 3.1 Agent-Harness-RAG Benchmark Study

A controlled study comparing Hybrid RAG (BM25 + Vector Search) against FileSearch (LLM-controlled agentic retrieval) on a 3-document corpus with 44 questions.

**Source:** https://github.com/bhargav-latent/Agent-Harness-RAG

#### Results

| Metric | Hybrid RAG | FileSearch (Agentic) | Delta |
|--------|-----------|---------------------|-------|
| Accuracy Score | 4.20/5 | **4.67/5** | +11% |
| Perfect Scores (5/5) | 53% | **80%** | +27pp |
| Median Latency | **31s** | 58s | 1.9x slower |
| Median Tokens | **12,137** | 37,294 | 3.1x more |

#### Key Findings

**FileSearch excels at:**
- Scattered information requiring iteration
- Tables and structured data extraction
- Semantic disambiguation through multiple search attempts

**Hybrid RAG excels at:**
- Simple keyword-matching queries
- Cost-sensitive production scenarios
- High-throughput requirements

### 3.2 LangChain's Production Experience

LangChain rebuilt their public documentation assistant, moving away from vector embeddings to an agentic approach.

**Source:** https://blog.langchain.com/rebuilding-chat-langchain/

#### Why They Abandoned Vector Search

> *"The original Chat LangChain used vector embeddings and document chunking, which fragmented structured content, required constant reindexing, and produced vague citations. Internal teams preferred manual workflows because they needed something more thorough than just using docs."*

#### Their New Approach

1. **Documentation**: Full-page returns via API with iterative refinement
2. **Knowledge Base**: Scan titles first, then read selected articles fully
3. **Codebase**: Pattern matching with ripgrep + targeted file reading

#### Core Insight

> *"Rather than reinventing search, the team automated what already worked—observing engineers following a specific pattern and systematizing it."*

---

## 4. Proposed Architecture

### 4.1 System Overview

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
│   ├── arxiv_2401_12345/                                      │
│   │   ├── metadata.json                                      │
│   │   ├── page_001.md                                        │
│   │   ├── page_002.md                                        │
│   │   └── ...                                                │
│   ├── arxiv_2402_67890/                                      │
│   │   └── ...                                                │
│   └── index.json                                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌──────────────────────────────────────────────────────────────┐
│               NEMOTRON PARSE v1.1                            │
│                                                              │
│   - Semantic segmentation (titles, tables, figures)          │
│   - Reading order preservation                               │
│   - LaTeX table/formula extraction                           │
│   - Bounding box coordinates                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌──────────────────────────────────────────────────────────────┐
│                    PDF INPUT                                 │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 PDF Parsing Layer: Nemotron Parse v1.1

**Source:** https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/

#### Specifications

| Attribute | Value |
|-----------|-------|
| Parameters | 885M (600M vision encoder + 250M decoder) |
| Architecture | ViT-H encoder + mBART decoder |
| Speed | ~4-5 pages/second |
| Table Accuracy (S-TEDS) | 93.99% |
| Output Format | Structured Markdown + Bounding Boxes |

#### Why Nemotron Parse

1. **Semantic Segmentation**: Classifies titles, headers, tables, figures, captions, footnotes
2. **Reading Order**: Multi-column layouts extracted correctly
3. **Table Excellence**: LaTeX format with multirow/multicolumn (best-in-class benchmarks)
4. **Formula Support**: Mathematical notation preserved in LaTeX
5. **Spatial Grounding**: Bounding boxes enable precise citations

### 4.3 Storage Layer: Structured File System

#### File Structure

```
papers/
├── index.json                    # Corpus-level metadata
├── arxiv_2401_12345/
│   ├── metadata.json             # Paper metadata
│   ├── page_001.md               # Full page content
│   ├── page_002.md
│   └── ...
└── arxiv_2402_67890/
    └── ...
```

#### Page File Format

```markdown
---
paper_id: arxiv_2401_12345
title: "Attention Is All You Need"
authors: ["Vaswani et al."]
page: 3
total_pages: 15
sections: ["3.1 Scaled Dot-Product Attention", "3.2 Multi-Head Attention"]
has_tables: true
has_figures: true
has_equations: true
---

## 3.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2).
The input consists of queries and keys of dimension $d_k$, and values of
dimension $d_v$...

| Layer Type | Complexity per Layer | Sequential Ops |
|------------|---------------------|----------------|
| Self-Attention | O(n²·d) | O(1) |
| Recurrent | O(n·d²) | O(n) |

[FIGURE: fig_2, bbox: [100, 200, 400, 500]]
Scaled Dot-Product Attention diagram showing Q, K, V matrices...
```

### 4.4 Retrieval Layer: Agentic Tools

#### Tool Definitions

| Tool | Signature | Purpose |
|------|-----------|---------|
| `list_papers` | `() → [{id, title, authors}]` | Discover available papers |
| `get_paper_toc` | `(paper_id) → {sections, pages}` | Get paper structure |
| `search` | `(query, scope?) → [{paper, page, snippet}]` | Keyword/regex search |
| `read_page` | `(paper_id, page_num) → content` | Fetch full page |
| `read_section` | `(paper_id, section) → content` | Fetch by section name |

#### Retrieval Loop

```
1. UNDERSTAND: Parse user query, identify key concepts
2. SEARCH: grep/glob across corpus for relevant terms
3. EVALUATE: Assess search results for relevance
4. READ: Fetch full pages for promising hits
5. ITERATE: If insufficient, refine query and repeat
6. SYNTHESIZE: Construct answer with precise citations
```

---

## 5. Trade-offs and Honest Assessment

### 5.1 What We Gain

| Benefit | Mechanism |
|---------|-----------|
| **Higher accuracy** | Full context preserved, iterative refinement |
| **Better table handling** | No chunking fragmentation |
| **Precise citations** | Page + bounding box coordinates |
| **Structural queries** | Agent can navigate by section |
| **No reindexing** | Add new papers instantly |

### 5.2 What We Pay

| Cost | Mitigation |
|------|------------|
| **Higher latency** (2x) | Acceptable per requirements |
| **More tokens** (3x) | Acceptable per requirements |
| **Custom tooling** | Leverage LangGraph/DeepAgents frameworks |
| **Less community support** | Architecture is well-documented in sources |

### 5.3 Known Limitations

From the Agent-Harness-RAG study:

1. **Runaway behavior**: Complex queries can spiral (9M tokens in extreme cases)
   - *Mitigation*: Implement token budgets and iteration limits

2. **Negation queries**: Both approaches struggle with "what is NOT mentioned"
   - *Mitigation*: Explicit tool for exhaustive search when negation detected

3. **Cross-document synthesis**: Challenging across any architecture
   - *Mitigation*: Multi-step agent workflow for comparative queries

---

## 6. Implementation Recommendations

### 6.1 Phase 1: Core Pipeline

1. Set up Nemotron Parse v1.1 inference (via HuggingFace or NVIDIA NIM)
2. Build PDF → Markdown conversion with metadata extraction
3. Implement file system storage structure
4. Create basic agent tools (search, read_page)

### 6.2 Phase 2: Agent Development

1. Implement agentic retrieval loop (LangGraph recommended)
2. Add iteration limits and token budgets
3. Build citation extraction (page + bounding box)
4. Test against known question-answer pairs

### 6.3 Phase 3: Optimization

1. Add caching for frequently accessed pages
2. Implement paper-level summaries for faster triage
3. Build specialized tools for table queries
4. Add evaluation framework (LLM-as-Judge)

---

## 7. References

1. **Agent-Harness-RAG Benchmark Study**
   https://github.com/bhargav-latent/Agent-Harness-RAG
   *Empirical comparison of Hybrid RAG vs Agentic FileSearch*

2. **Rebuilding Chat LangChain**
   https://blog.langchain.com/rebuilding-chat-langchain/
   *Production experience moving away from vector embeddings*

3. **NVIDIA Nemotron Parse v1.1 Technical Blog**
   https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/
   *Parser capabilities and benchmarks*

4. **NVIDIA Nemotron Parse v1.1 Paper**
   https://arxiv.org/abs/2511.20478
   *Academic paper with full benchmark details*

5. **NVIDIA Nemotron Parse v1.1 Model**
   https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1
   *Model weights and documentation*

---

## 8. Conclusion

The Full-Page Agentic RAG architecture represents a deliberate trade-off: accepting higher latency and token costs in exchange for superior accuracy and reliability. This is not a speculative choice—it is backed by:

1. **Empirical evidence**: 11% accuracy improvement, 80% vs 53% perfect scores
2. **Production experience**: LangChain's move away from vector embeddings
3. **Technical capability**: Nemotron Parse's state-of-the-art document understanding

For research paper Q&A where accuracy is paramount, this architecture is the right choice.

---

*Document prepared for architectural review and implementation planning.*
