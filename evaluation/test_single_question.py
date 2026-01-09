"""Test evaluation flow with a single question and enhanced metrics."""

import json
import time
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


def load_first_question() -> dict:
    """Load the first question from JSONL file."""
    with open(QUESTIONS_FILE, "r") as f:
        first_line = f.readline()
        return json.loads(first_line)


def invoke_agent(question: str, thread_id: str = "eval-thread") -> tuple[str, dict]:
    """
    Invoke the RAG agent and return the answer and metadata.

    Returns:
        (answer, metadata) where metadata includes latency, tool_calls, tokens
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

                # Parse SSE events - they come in format "event: <type>\ndata: <json>"
                if line_str.startswith('data: '):
                    try:
                        event_data = json.loads(line_str[6:])  # Remove 'data: ' prefix

                        # Handle list format (from messages/partial and messages/complete events)
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

                        # Handle dict format (updates events with messages)
                        elif isinstance(event_data, dict):
                            # Check for messages in update events
                            if "messages" in event_data:
                                msgs = event_data["messages"]
                                if isinstance(msgs, list):
                                    for msg in msgs:
                                        if isinstance(msg, dict):
                                            all_messages.append(msg)

                    except (json.JSONDecodeError, AttributeError, TypeError):
                        # Skip malformed events
                        continue

        latency = time.time() - start_time

        # Extract final answer from messages (last AI message with content)
        answer = ""
        for msg in reversed(all_messages):
            if isinstance(msg, dict):
                if msg.get("type") == "ai" and msg.get("content"):
                    content = msg["content"]
                    # Content might be a string, list, or dict
                    if isinstance(content, str):
                        answer = content
                        break
                    elif isinstance(content, list) and len(content) > 0:
                        # Content is a list of content blocks
                        # Extract text from each block
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

        # Sum token usage from ALL AI messages (each tool loop has a separate AI call)
        # This gives us the total across the entire agent execution
        # IMPORTANT: Deduplicate by message ID since streaming sends partial updates
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        seen_message_ids = set()

        for msg in all_messages:
            if isinstance(msg, dict) and msg.get("type") == "ai" and "usage_metadata" in msg:
                msg_id = msg.get("id")
                # Only count each unique message once
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
            "tools_used": list(set(tools_used)),  # Remove duplicates
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

    # Escape the agent answer to prevent JSON issues with LaTeX and special chars
    import json as json_module
    question_safe = json_module.dumps(question)[1:-1]  # Remove outer quotes
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
        # Extract JSON from response
        content = response.content

        # Handle different response formats
        if isinstance(content, list):
            # Content is a list of blocks
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
        # Extract JSON from response
        content = response.content

        # Handle different response formats
        if isinstance(content, list):
            # Content is a list of blocks
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


def run_single_question_test():
    """Run evaluation on a single question to test the flow."""
    print("=" * 70)
    print("🧪 TESTING EVALUATION FLOW WITH SINGLE QUESTION")
    print("=" * 70)
    print()

    # Check if server is running
    try:
        response = requests.get(f"{LANGGRAPH_URL}/ok", timeout=5)
        if response.status_code != 200:
            print("❌ LangGraph server not responding.")
            print("   Start it with: langgraph dev --port 2030")
            return
    except Exception:
        print("❌ LangGraph server not running.")
        print("   Start it with: langgraph dev --port 2030")
        return

    print("✅ LangGraph server is running\n")

    # Load first question
    question_data = load_first_question()
    print(f"📝 Testing with question: {question_data['id']}")
    print(f"   Category: {question_data['category']}")
    print(f"   Difficulty: {question_data['metadata']['difficulty']}")
    print()
    print(f"Question: {question_data['question']}")
    print()

    # Invoke agent
    print("🤖 Invoking RAG agent...")
    answer, metadata = invoke_agent(
        question_data["question"],
        thread_id=f"test-{question_data['id']}"
    )

    if metadata["status"] != "success":
        print(f"❌ Error: {metadata.get('error', 'Unknown')}")
        return

    print(f"✅ Agent responded in {metadata['latency']:.2f}s")
    print(f"✅ Token usage: {metadata['total_tokens']:,} total")
    print(f"   - Input tokens:  {metadata['input_tokens']:,}")
    print(f"   - Output tokens: {metadata['output_tokens']:,}")
    print()
    print("=" * 70)
    print("AGENT ANSWER:")
    print("=" * 70)
    print(answer)
    print()

    # Evaluate correctness
    print("🔍 Evaluating correctness...")
    correctness = llm_as_judge_correctness(
        question_data["question"],
        question_data["ground_truth"],
        answer
    )

    # Evaluate groundedness
    print("🔍 Evaluating groundedness...")
    groundedness = llm_as_judge_groundedness(
        answer,
        question_data["source_documents"]
    )

    # Print detailed results
    print()
    print("=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print()
    print(f"Question ID: {question_data['id']}")
    print(f"Question: {question_data['question']}")
    print()
    print(f"Ground Truth:")
    print(f"  {question_data['ground_truth']}")
    print()
    print(f"Agent Answer:")
    print(f"  {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print()
    print("METRICS:")
    print("-" * 70)
    print(f"✓ Correctness Score:    {correctness['score']}/5")
    print(f"  Explanation: {correctness['explanation']}")
    print()
    print(f"✓ Groundedness Score:   {groundedness['score']}/5")
    print(f"  Explanation: {groundedness['explanation']}")
    print()
    print(f"✓ Latency:              {metadata['latency']:.2f}s")
    print(f"✓ Tool Calls:           {metadata['tool_calls']}")
    print(f"✓ Tools Used:           {', '.join(metadata['tools_used']) if metadata['tools_used'] else 'None'}")
    print(f"✓ Total Tokens:         {metadata['total_tokens']:,}")
    print(f"  - Input Tokens:       {metadata['input_tokens']:,}")
    print(f"  - Output Tokens:      {metadata['output_tokens']:,}")
    print()
    print("=" * 70)

    # Save detailed result
    result = {
        "id": question_data["id"],
        "question": question_data["question"],
        "category": question_data["category"],
        "difficulty": question_data["metadata"]["difficulty"],
        "ground_truth": question_data["ground_truth"],
        "source_documents": question_data["source_documents"],
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

    # Save to file
    output_file = Path(__file__).parent / "results" / "single_question_test.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"💾 Detailed results saved to: {output_file}")
    print()

    # Summary
    if correctness["score"] >= 4 and groundedness["score"] >= 4:
        print("✅ EVALUATION FLOW TEST PASSED!")
        print("   The agent provided a correct and grounded answer.")
    elif correctness["score"] >= 3 or groundedness["score"] >= 3:
        print("⚠️  EVALUATION FLOW TEST PARTIAL SUCCESS")
        print("   The agent's answer needs improvement.")
    else:
        print("❌ EVALUATION FLOW TEST NEEDS ATTENTION")
        print("   The agent's answer was not satisfactory.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_single_question_test()
