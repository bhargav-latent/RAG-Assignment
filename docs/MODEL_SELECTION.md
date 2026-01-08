# Model Selection Guide

A decision framework for selecting models for this RAG system.

---

## Requirements

This system needs models that can:

| Capability | Required | Why |
|------------|----------|-----|
| Tool calling | Yes | Filesystem operations (ls, read, grep, glob) |
| Vision | Yes | Analyzing figures, charts, tables in papers |
| Long output | Preferred | Detailed responses with citations |
| Structured output | Preferred | Consistent formatting |

---

## Evaluation Criteria

### 1. Tool Use Reliability

The agent makes 5-15 tool calls per query. Measure:
- Tool call success rate
- Correct parameter formatting
- Recovery from errors

**Benchmark to check:** Computer-use scores, function calling benchmarks

### 2. Vision Quality

Figures contain equations, charts, diagrams. Measure:
- Accuracy on mathematical notation
- Chart data extraction
- Diagram understanding

**Benchmark to check:** MMMU, MathVista, ChartQA

### 3. Context Efficiency

Documents can be 1000+ lines. Measure:
- Follows offset/limit patterns
- Uses grep before read
- Doesn't request full files

**Test:** Ask "list all formulas" - should grep, not read everything

### 4. Instruction Following

System prompt defines workflow patterns. Measure:
- Follows Map → Search → Understand
- Uses correct citation format
- Respects read limits

---

## Cost vs Capability Trade-offs

| Tier | Example Models | Use When |
|------|----------------|----------|
| **Frontier** | Claude Sonnet, GPT-4o, Gemini Pro | Complex reasoning needed |
| **Mid-tier** | Claude Haiku, Gemini Flash, GPT-4o-mini | Balance of cost/capability |
| **Budget** | Grok Fast, DeepSeek | High volume, simpler queries |
| **Self-hosted** | Qwen, Llama, Mistral | Privacy, zero marginal cost |

---

## Key Metrics by Model Class

### API Models

| Model | Tool Use | Vision | Context | Cost (1M tok) |
|-------|----------|--------|---------|---------------|
| Gemini 3 Flash | Good | Yes | 1M | $0.50/$3 |
| Claude Haiku 4.5 | Excellent | No* | 200K | $1/$5 |
| GPT-4o-mini | Good | Yes | 128K | $0.15/$0.60 |
| Grok 4.1 Fast | Good | Limited | 2M | $0.20/- |

*Requires separate vision model

### Self-Hosted

| Model | Tool Use | Vision | Context | Notes |
|-------|----------|--------|---------|-------|
| Qwen3-VL | Good | Integrated | 32K | 3B active (MoE) |
| Llama 3.3 | Good | Separate | 128K | Large community |
| Mistral | Good | Separate | 32K | Fast inference |

---

## Decision Tree

```
Start
  │
  ├─ Need vision integrated?
  │   ├─ Yes → Qwen-VL, GPT-4o, Gemini
  │   └─ No → Any model + separate vision
  │
  ├─ Self-hosted required?
  │   ├─ Yes → Qwen, Llama, Mistral
  │   └─ No → API options available
  │
  ├─ Context > 100K needed?
  │   ├─ Yes → Gemini (1M), Grok (2M)
  │   └─ No → Most models work
  │
  └─ Budget priority?
      ├─ Lowest cost → Grok Fast, self-hosted
      ├─ Balanced → Gemini Flash, GPT-4o-mini
      └─ Best quality → Claude Haiku, Gemini Flash
```

---

## Testing Checklist

Before deploying a new model, verify:

- [ ] `ls` returns paper directories correctly
- [ ] `grep` finds content with line numbers
- [ ] `read_file` respects offset/limit
- [ ] `read_images` analyzes figures accurately
- [ ] Citations format correctly [paper, page X]
- [ ] Doesn't read entire files for targeted queries
- [ ] Handles "file not found" gracefully

---

## Configuration

Models are configured via environment variables:

```bash
# Agent model
OPENAI_BASE_URL=<provider_url>
OPENAI_API_KEY=<api_key>
AGENT_MODEL=<model_name>

# Vision model (can be same or different)
VLLM_BASE_URL=<vision_provider_url>
VLLM_MODEL_NAME=<vision_model_name>
```

---

## Monitoring

Track these metrics in LangSmith:

- **Tokens per query** - Context efficiency
- **Tool calls per query** - Workflow adherence
- **Error rate** - Model reliability
- **Latency** - User experience

---

*Use this guide when evaluating model changes.*
