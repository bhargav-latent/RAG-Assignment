# Evaluation Framework

A systematic approach to evaluating the agentic RAG system.

---

## Overview

This framework measures how well the agent retrieves and synthesizes information from academic papers using tool-based exploration (grep, read_file, glob).

---

## Question Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Exact Match** | Find specific terms, codes, values | "What is the learning rate used?" |
| **Semantic** | Understand meaning despite different wording | "How does the model handle long sequences?" |
| **Table Data** | Extract from tables and structured content | "What are the benchmark results on dataset X?" |
| **Formulas** | Retrieve equations and mathematical expressions | "What is the attention formula?" |
| **Multi-hop** | Synthesize from multiple sections/pages | "How does method A compare to baseline B?" |
| **Figures** | Analyze visual content with vision model | "What does Figure 3 show?" |
| **Acronyms** | Bridge technical shorthand | "What does BLEU stand for and how is it calculated?" |
| **Contextual** | Same term, different meanings across papers | "What does 'head' refer to in this context?" |
| **Negation** | Understand "not", "except", "without" | "Which methods don't use attention?" |
| **Temporal** | Time-based or version information | "What changed between v1 and v2?" |
| **Factual** | Exact numbers, dates, names | "How many parameters does the model have?" |
| **Conceptual** | High-level "why" and "how" questions | "Why is self-attention used instead of RNNs?" |

---

## Dataset Schema

Questions stored in JSONL format:

```json
{
  "id": "q001",
  "question": "What optimizer is used for training?",
  "ground_truth": "Adam optimizer with beta1=0.9, beta2=0.98",
  "category": "exact_match",
  "source_documents": ["attention_paper"],
  "metadata": {
    "difficulty": "easy",
    "expected_location": "Page 7, Training section",
    "reasoning_type": "single-hop"
  }
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier (q001, q002, ...) |
| `question` | Yes | Natural language query |
| `ground_truth` | Yes | Expected correct answer |
| `category` | Yes | One of the 12 categories above |
| `source_documents` | Yes | Paper folder names containing the answer |
| `metadata.difficulty` | Yes | easy, medium, hard |
| `metadata.expected_location` | No | Where in the document (for debugging) |
| `metadata.reasoning_type` | No | single-hop or multi-hop |

---

## Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Correctness** | LLM-as-judge score (1-5 scale) | >= 4.0 average |
| **Latency** | End-to-end response time | < 60s median |
| **Token Usage** | Total input + output tokens | < 50K median |
| **Tool Calls** | Number of tool invocations | < 10 median |

### Correctness Rubric

| Score | Meaning |
|-------|---------|
| 5 | Perfect - complete and accurate |
| 4 | Mostly correct with minor omissions |
| 3 | Partially correct, missing key details |
| 2 | Mostly incorrect or very incomplete |
| 1 | Wrong or irrelevant |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Grep-before-read ratio** | % of read_file calls preceded by grep |
| **Offset/limit usage** | % of read_file calls using pagination |
| **Figure analysis accuracy** | Correctness on figure-related questions |

---

## LLM-as-Judge Prompt

```
You are evaluating a RAG system's answer quality.

Question: {question}
Ground Truth: {ground_truth}
Agent Answer: {agent_answer}

Score the answer from 1-5:
- 5: Perfect, complete, accurate
- 4: Mostly correct, minor omissions
- 3: Partially correct, missing key details
- 2: Mostly incorrect or incomplete
- 1: Wrong or irrelevant

Respond in JSON:
{
  "score": <1-5>,
  "explanation": "<brief reasoning>"
}
```

---

## Evaluation Process

### 1. Prepare Dataset

Create `evaluation/questions.jsonl` with 30-50 questions:
- ~3-5 questions per category
- Mix of difficulties (30% easy, 50% medium, 20% hard)
- Cover all papers in the corpus

### 2. Run Evaluation

```bash
# Pseudocode - implement based on your setup
python evaluation/run_benchmark.py \
  --questions evaluation/questions.jsonl \
  --output evaluation/results/
```

### 3. Collect Results

For each question:
1. Invoke agent via LangGraph API
2. Record latency (start to final response)
3. Extract token usage from LangSmith trace
4. Count tool calls from message history
5. Run LLM-as-judge on (question, ground_truth, answer)

### 4. Analyze

```
evaluation/results/
├── benchmark_YYYYMMDD.csv    # Summary metrics
├── benchmark_YYYYMMDD.jsonl  # Full details
└── analysis.md               # Findings
```

---

## Performance Targets by Category

| Category | Target Score | Notes |
|----------|-------------|-------|
| Formulas | 4.5-5.0 | High precision expected |
| Table Data | 4.5-5.0 | Structured content |
| Exact Match | 4.0-4.5 | Direct retrieval |
| Semantic | 4.0-4.5 | Iterative refinement possible |
| Figures | 4.0-4.5 | Depends on vision model quality |
| Multi-hop | 3.5-4.0 | Multiple retrieval steps required |
| Contextual | 3.0-4.0 | Disambiguation challenging |
| Negation | 3.0-3.5 | Exhaustive search required |

---

## Known Challenges

### 1. Runaway Queries
Some questions cause excessive tool calls. Monitor for:
- Token usage > 100K
- Latency > 5 minutes
- Tool calls > 20

**Mitigation:** Set hard limits in agent configuration.

### 2. Context Window Pressure
Large documents can fill context quickly.

**Mitigation:**
- Use grep before read_file
- Always specify offset/limit
- 500-line read limit (already configured)

### 3. Figure Analysis Variance
Vision model accuracy varies by figure complexity.

**Mitigation:** Test vision model separately on figure-only questions.

---

## Sample Questions

### Easy - Exact Match
```json
{
  "id": "q001",
  "question": "What is the model dimension (d_model) in the Transformer?",
  "ground_truth": "512",
  "category": "exact_match",
  "source_documents": ["attention_paper"],
  "metadata": {"difficulty": "easy", "reasoning_type": "single-hop"}
}
```

### Medium - Multi-hop
```json
{
  "id": "q015",
  "question": "How does the training time compare between the base and big model configurations?",
  "ground_truth": "Base model: 12 hours on 8 P100 GPUs. Big model: 3.5 days on 8 P100 GPUs.",
  "category": "multi_hop",
  "source_documents": ["attention_paper"],
  "metadata": {"difficulty": "medium", "reasoning_type": "multi-hop"}
}
```

### Hard - Conceptual
```json
{
  "id": "q030",
  "question": "Why does the Transformer use multi-head attention instead of a single attention function?",
  "ground_truth": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.",
  "category": "conceptual",
  "source_documents": ["attention_paper"],
  "metadata": {"difficulty": "hard", "reasoning_type": "single-hop"}
}
```

---

## Reporting

After evaluation, report:

1. **Overall Metrics**
   - Average correctness score
   - Median latency, tokens, tool calls
   - Success rate (responses with score >= 4)

2. **By Category**
   - Score distribution per category
   - Identify weak categories for improvement

3. **By Difficulty**
   - Easy/medium/hard performance breakdown

4. **Failure Analysis**
   - Questions scoring 1-2
   - Common failure patterns
   - Recommendations

---

## Continuous Evaluation

Integrate evaluation into development:

1. **Regression Testing** - Run benchmark after model/prompt changes
2. **New Paper Validation** - Add 3-5 questions per new paper
3. **LangSmith Monitoring** - Track production metrics over time

---

*Use this framework to measure and improve retrieval quality systematically.*
