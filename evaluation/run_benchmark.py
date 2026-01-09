"""Simple evaluation script for RAG agent."""

import json
import time
from datetime import datetime
from pathlib import Path

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
LANGGRAPH_URL = "http://127.0.0.1:2030"
QUESTIONS_FILE = Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_questions(questions_file: Path) -> list[dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(questions_file, "r") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def invoke_agent(question: str, thread_id: str = "eval-thread") -> tuple[str, dict]:
    """
    Invoke the RAG agent and return the answer and metadata.

    Returns:
        (answer, metadata) where metadata includes latency, tool_calls, and tokens
    """
    start_time = time.time()

    try:
        # Step 1: Create a thread
        create_thread_url = f"{LANGGRAPH_URL}/threads"
        thread_response = requests.post(create_thread_url, json={}, timeout=10)
        thread_response.raise_for_status()
        thread_id = thread_response.json()["thread_id"]

        # Step 2: Invoke the agent with streaming
        invoke_url = f"{LANGGRAPH_URL}/threads/{thread_id}/runs/stream"

        payload = {
            "assistant_id": "rag_agent",
            "input": {
                "messages": [{"role": "user", "content": question}]
            },
            "stream_mode": ["messages", "updates"]
        }

        response = requests.post(invoke_url, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        # Collect streaming events
        all_messages = []
        tool_calls = 0
        tools_used = []

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')

                if line_str.startswith('data: '):
                    try:
                        event_data = json.loads(line_str[6:])

                        # Handle list format
                        if isinstance(event_data, list):
                            for msg in event_data:
                                if isinstance(msg, dict):
                                    all_messages.append(msg)

                                    # Track tool calls
                                    msg_type = msg.get("type")
                                    if msg_type == "ai" and "tool_calls" in msg:
                                        tool_calls_list = msg.get("tool_calls", [])
                                        if tool_calls_list:
                                            tool_calls += len(tool_calls_list)
                                            for tc in tool_calls_list:
                                                if isinstance(tc, dict) and "name" in tc:
                                                    tools_used.append(tc["name"])
                                    elif msg_type == "tool":
                                        if "name" in msg:
                                            tools_used.append(msg["name"])

                        # Handle dict format
                        elif isinstance(event_data, dict):
                            if "messages" in event_data:
                                msgs = event_data["messages"]
                                if isinstance(msgs, list):
                                    for msg in msgs:
                                        if isinstance(msg, dict):
                                            all_messages.append(msg)

                    except (json.JSONDecodeError, AttributeError, TypeError):
                        continue

        latency = time.time() - start_time

        # Extract final answer from messages
        answer = ""
        for msg in reversed(all_messages):
            if isinstance(msg, dict):
                if msg.get("type") == "ai" and msg.get("content"):
                    content = msg["content"]
                    if isinstance(content, str):
                        answer = content
                        break
                    elif isinstance(content, list) and len(content) > 0:
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                text_parts.append(block["text"])
                            elif isinstance(block, str):
                                text_parts.append(block)
                            else:
                                text_parts.append(str(block))
                        answer = "\n".join(text_parts)
                        break
                    elif isinstance(content, dict) and "text" in content:
                        answer = content["text"]
                        break

        # Sum token usage from all unique AI messages (deduplicate by ID)
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        seen_message_ids = set()

        for msg in all_messages:
            if isinstance(msg, dict) and msg.get("type") == "ai" and "usage_metadata" in msg:
                msg_id = msg.get("id")
                if msg_id and msg_id not in seen_message_ids:
                    seen_message_ids.add(msg_id)
                    metadata = msg.get("usage_metadata", {})
                    if isinstance(metadata, dict):
                        input_tokens += metadata.get("input_tokens", 0)
                        output_tokens += metadata.get("output_tokens", 0)

        total_tokens = input_tokens + output_tokens

        metadata = {
            "latency": latency,
            "tool_calls": tool_calls,
            "tools_used": list(set(tools_used)),
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "status": "success"
        }

        return answer, metadata

    except Exception as e:
        latency = time.time() - start_time
        metadata = {
            "latency": latency,
            "tool_calls": 0,
            "tools_used": [],
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "status": "error",
            "error": str(e)
        }
        return "", metadata


def llm_as_judge_correctness(question: str, ground_truth: str, agent_answer: str) -> dict:
    """
    Use LLM to score correctness of the agent's answer.

    Returns:
        {"score": 1-5, "explanation": "..."}
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Escape the agent answer to prevent JSON issues
    import json as json_module
    question_safe = json_module.dumps(question)[1:-1]
    ground_truth_safe = json_module.dumps(ground_truth)[1:-1]
    agent_answer_safe = json_module.dumps(agent_answer)[1:-1]

    prompt = f"""You are evaluating a RAG system's answer for CORRECTNESS.

Question: {question_safe}
Ground Truth: {ground_truth_safe}
Agent Answer: {agent_answer_safe}

Score the CORRECTNESS from 1-5:
- 5: Completely correct, all key facts match ground truth
- 4: Mostly correct, minor missing details or slight inaccuracies
- 3: Partially correct, has some right information but missing key facts
- 2: Mostly incorrect, major errors or omissions
- 1: Completely wrong or irrelevant

Respond in JSON format only:
{{
  "score": <1-5>,
  "explanation": "<brief reasoning about correctness>"
}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content

        # Handle different response formats
        if isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text += block["text"]
                elif isinstance(block, str):
                    text += block
            content = text
        elif isinstance(content, dict) and "text" in content:
            content = content["text"]

        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        return result
    except Exception as e:
        return {"score": 0, "explanation": f"Evaluation error: {str(e)}"}


def llm_as_judge_groundedness(agent_answer: str, source_documents: list[str]) -> dict:
    """
    Use LLM to score groundedness - whether the answer is supported by source documents.

    Returns:
        {"score": 1-5, "explanation": "..."}
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Escape the agent answer to prevent JSON issues
    import json as json_module
    agent_answer_safe = json_module.dumps(agent_answer)[1:-1]
    docs_list = ", ".join(source_documents)

    prompt = f"""You are evaluating whether a RAG system's answer is GROUNDED in the source documents.

Agent Answer: {agent_answer_safe}
Expected Source Documents: {docs_list}

Score the GROUNDEDNESS from 1-5:
- 5: Answer explicitly cites sources and all claims are verifiable from documents
- 4: Answer is grounded but citations could be more explicit
- 3: Answer appears grounded but lacks clear source attribution
- 2: Answer makes claims that may not be from the specified sources
- 1: Answer is not grounded, makes up information or uses wrong sources

Look for:
- Explicit document citations (e.g., "[Paper Name, page X]")
- References to specific formulas, values, or facts that would be in the documents
- Appropriate attribution of information to sources

Respond in JSON format only:
{{
  "score": <1-5>,
  "explanation": "<brief reasoning about groundedness>"
}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content

        # Handle different response formats
        if isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text += block["text"]
                elif isinstance(block, str):
                    text += block
            content = text
        elif isinstance(content, dict) and "text" in content:
            content = content["text"]

        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        return result
    except Exception as e:
        return {"score": 0, "explanation": f"Evaluation error: {str(e)}"}


def run_evaluation():
    """Run the evaluation benchmark."""
    print("🚀 Starting RAG Agent Evaluation\n")

    # Load questions
    questions = load_questions(QUESTIONS_FILE)
    print(f"📝 Loaded {len(questions)} questions\n")

    # Check if server is running
    try:
        response = requests.get(f"{LANGGRAPH_URL}/ok", timeout=5)
        if response.status_code != 200:
            print("❌ LangGraph server not responding. Start it with: langgraph dev --port 2030")
            return
    except Exception:
        print("❌ LangGraph server not running. Start it with: langgraph dev --port 2030")
        return

    print("✅ LangGraph server is running\n")

    # Run evaluation
    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:60]}...")

        # Invoke agent
        answer, metadata = invoke_agent(q["question"], thread_id=f"eval-{q['id']}")

        if metadata["status"] == "success":
            # Evaluate correctness and groundedness
            correctness = llm_as_judge_correctness(q["question"], q["ground_truth"], answer)
            groundedness = llm_as_judge_groundedness(answer, q["source_documents"])

            result = {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "difficulty": q["metadata"]["difficulty"],
                "ground_truth": q["ground_truth"],
                "source_documents": q["source_documents"],
                "agent_answer": answer,
                "correctness_score": correctness["score"],
                "correctness_explanation": correctness["explanation"],
                "groundedness_score": groundedness["score"],
                "groundedness_explanation": groundedness["explanation"],
                "latency": metadata["latency"],
                "tool_calls": metadata["tool_calls"],
                "tools_used": metadata["tools_used"],
                "total_tokens": metadata["total_tokens"],
                "input_tokens": metadata["input_tokens"],
                "output_tokens": metadata["output_tokens"],
            }

            print(f"   Correctness: {correctness['score']}/5 | Groundedness: {groundedness['score']}/5 | Latency: {metadata['latency']:.1f}s | Tokens: {metadata['total_tokens']:,}")
        else:
            result = {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "difficulty": q["metadata"]["difficulty"],
                "ground_truth": q["ground_truth"],
                "source_documents": q["source_documents"],
                "agent_answer": "",
                "correctness_score": 0,
                "correctness_explanation": f"Error: {metadata.get('error', 'Unknown')}",
                "groundedness_score": 0,
                "groundedness_explanation": "N/A - Agent failed",
                "latency": metadata["latency"],
                "tool_calls": 0,
                "tools_used": [],
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            print(f"   ❌ Error: {metadata.get('error', 'Unknown')}")

        results.append(result)
        print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed JSONL
    jsonl_path = RESULTS_DIR / f"benchmark_{timestamp}.jsonl"
    with open(jsonl_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Calculate summary statistics
    valid_results = [r for r in results if r["correctness_score"] > 0]

    if valid_results:
        avg_correctness = sum(r["correctness_score"] for r in valid_results) / len(valid_results)
        avg_groundedness = sum(r["groundedness_score"] for r in valid_results) / len(valid_results)
        avg_latency = sum(r["latency"] for r in valid_results) / len(valid_results)
        avg_tools = sum(r["tool_calls"] for r in valid_results) / len(valid_results)
        avg_tokens = sum(r["total_tokens"] for r in valid_results) / len(valid_results)
        success_rate = len([r for r in valid_results if r["correctness_score"] >= 4]) / len(valid_results) * 100

        # Print summary
        print("=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Questions: {len(questions)}")
        print(f"Successful: {len(valid_results)}")
        print(f"Failed: {len(results) - len(valid_results)}")
        print()
        print(f"Average Correctness:  {avg_correctness:.2f}/5")
        print(f"Average Groundedness: {avg_groundedness:.2f}/5")
        print(f"Success Rate (≥4):    {success_rate:.1f}%")
        print(f"Average Latency:      {avg_latency:.1f}s")
        print(f"Average Tool Calls:   {avg_tools:.1f}")
        print(f"Average Tokens:       {avg_tokens:,.0f}")
        print()

        # By category
        print("By Category:")
        categories = {}
        for r in valid_results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"correctness": [], "groundedness": []}
            categories[cat]["correctness"].append(r["correctness_score"])
            categories[cat]["groundedness"].append(r["groundedness_score"])

        for cat, scores in sorted(categories.items()):
            avg_c = sum(scores["correctness"]) / len(scores["correctness"])
            avg_g = sum(scores["groundedness"]) / len(scores["groundedness"])
            print(f"  {cat:20s}: C={avg_c:.2f} G={avg_g:.2f} (n={len(scores['correctness'])})")

        print()
        print(f"Results saved to: {jsonl_path}")
        print("=" * 60)
    else:
        print("❌ No successful evaluations")


if __name__ == "__main__":
    run_evaluation()
