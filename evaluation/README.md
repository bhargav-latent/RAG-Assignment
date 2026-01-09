# RAG Agent Evaluation

This directory contains evaluation scripts and datasets for testing the RAG agent's performance.

## Quick Start

### 1. Start the LangGraph Server

```bash
langgraph dev --port 2030
```

### 2. Test with Single Question

Run a test with one question to verify the evaluation flow:

```bash
cd evaluation
python test_single_question.py
```

This will test the first question and track:
- **Correctness** - LLM-as-judge scoring (1-5) on answer accuracy
- **Groundedness** - LLM-as-judge scoring (1-5) on source attribution
- **Latency** - Time taken to generate answer
- **Token Usage** - Estimated tokens used
- **Tool Calls** - Number and types of tools used

### 3. Run Full Benchmark

Run evaluation on all 25 questions:

```bash
python run_benchmark.py
```

## Evaluation Metrics

### Correctness (LLM-as-Judge)
Scores from 1-5:
- **5**: Completely correct, all key facts match ground truth
- **4**: Mostly correct, minor missing details
- **3**: Partially correct, some right information but missing key facts
- **2**: Mostly incorrect, major errors
- **1**: Completely wrong or irrelevant

### Groundedness (LLM-as-Judge)
Scores from 1-5:
- **5**: Answer explicitly cites sources, all claims verifiable
- **4**: Answer is grounded but citations could be more explicit
- **3**: Answer appears grounded but lacks clear source attribution
- **2**: Answer makes claims that may not be from specified sources
- **1**: Answer is not grounded, makes up information

### Performance Metrics
- **Latency**: Time from question submission to answer completion
- **Token Usage**: Total tokens processed across all agent AI calls
- **Tool Calls**: Number of tool invocations (grep, read_file, etc.)

## Question Dataset

The `questions.jsonl` file contains 25 grounded questions:

### By Paper (5 questions each):
1. **Latent Diffusion Models** - Formulas, compression rates, datasets
2. **Science-Datasets (The Well)** - Equations, software, dataset specs
3. **Writing Effective Use Cases** - Components, levels, stakeholders
4. **Vector Symbolic FSMs** - Operations, formulas, architectures

### Cross-Document (5 questions):
Questions requiring synthesis across multiple papers

### By Category:
- **Precision** - Exact formulas, specific values, definitions
- **Recall** - Complete lists, all components, comprehensive coverage
- **Cross-Document** - Multi-hop reasoning across papers

## Output Files

Results are saved to `results/` directory:

- **single_question_test.json** - Detailed single question test results
- **benchmark_YYYYMMDD_HHMMSS.jsonl** - Full benchmark results with timestamp

Each result includes:
```json
{
  "id": "question_id",
  "question": "...",
  "ground_truth": "...",
  "agent_answer": "...",
  "correctness_score": 4,
  "correctness_explanation": "...",
  "groundedness_score": 5,
  "groundedness_explanation": "...",
  "latency": 3.42,
  "tool_calls": 5,
  "tools_used": ["grep", "read_file"],
  "total_tokens": 1250,
  "input_tokens": 900,
  "output_tokens": 350
}
```

## Expected Performance

For a well-functioning RAG agent:
- **Correctness Score**: ≥ 4.0 average
- **Groundedness Score**: ≥ 4.0 average
- **Latency**: < 5s per question
- **Success Rate**: ≥ 80% of questions score ≥ 4

## Troubleshooting

**Issue: "LangGraph server not running"**
- Start the server: `langgraph dev --port 2030`
- Verify it's running: `curl http://127.0.0.1:2030/ok`

**Issue: "GOOGLE_API_KEY not found"**
- Ensure `.env` file exists in project root
- Add `GOOGLE_API_KEY=your_key_here`

**Issue: Evaluation scores are low**
- Check that papers are in `papers/` directory
- Verify document.md files exist for each paper
- Review agent logs in LangSmith for tool usage patterns

## Question Format

Each question in `questions.jsonl` follows this format:

```json
{
  "id": "paper_qXXX",
  "question": "What is the formula for...?",
  "ground_truth": "The formula is...",
  "category": "precision|recall|cross-document",
  "source_documents": ["Paper Name"],
  "metadata": {
    "difficulty": "easy|medium|hard",
    "reasoning_type": "single-hop|multi-hop"
  }
}
```
