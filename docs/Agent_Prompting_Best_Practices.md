# Agent Prompting Best Practices

Research-backed guidelines for designing effective agent system prompts.

---

## The Core Insight

From Anthropic's "Building Effective Agents" (2024):

> "Invest as much effort in agent-computer interfaces (ACI) as you would in human-computer interfaces."

**Translation:** The quality of your TOOL DESCRIPTIONS matters more than anything else. Not workflows. Not examples. Not instructions.

---

## What Makes a Good Agent Prompt

### 1. Clear Data Structure

The agent needs to understand what it's working with. Show the actual structure:

```
papers/
  {paper_name}/
    page_0001.md
    page_0002.md
    figures/
      page_0001_fig_01.png
```

Not just "papers are stored as markdown files" - show the EXACT layout.

### 2. Excellent Tool Documentation

Each tool needs:
- **What it does** (one line)
- **What it returns** (exact format)
- **When to use it** (the reasoning hint)

**Example:**

```
**grep(pattern, path)**
Search for regex pattern across all files in path.
Returns: List of matches with `filepath:line_number:matching_line`
Use for: Finding which pages mention a topic, locating specific terms
```

The "Use for" line is NOT a workflow instruction - it's helping the model understand the tool's PURPOSE.

### 3. No Workflow Instructions

**Bad:**
```
1. First use grep to find relevant pages
2. Then use read_file on those pages
3. Finally synthesize the answer
```

**Why it fails:**
- Model follows rigidly even when inappropriate
- Breaks on edge cases (what if grep finds nothing?)
- Reduces autonomous reasoning

**Good:** Just describe the tools well. The model figures out the workflow.

### 4. No Rigid Examples

**Bad:**
```
Example:
User: "Find formulas"
Agent: grep("equation|formula", "papers/")
```

**Why it fails:**
- Model copies the pattern instead of reasoning
- Becomes a template, not a guide
- Fails on queries that don't match the example

---

## The Three-Part Structure

A good agent prompt has exactly three parts:

### Part 1: Context (2-3 lines)
What is this agent? What's its purpose?

```
You are a research assistant answering questions about academic papers.
```

### Part 2: Data Structure (show, don't tell)
What does the data look like? Use actual examples.

```
papers/
  {paper_name}/
    page_0001.md      # Page content
    figures/
      page_0001_fig_01.png
```

### Part 3: Tool Documentation (the bulk of the prompt)
Detailed docs for each tool: what, returns, use for.

---

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| "ALWAYS use X before Y" | Rigid workflow | Describe tools, let model decide |
| "NEVER do X" | Defensive, assumes failure | Trust model or fix tool |
| Step-by-step examples | Becomes template | Describe tool purposes instead |
| Vague tool descriptions | Model guesses wrong | Add "Returns:" and "Use for:" |
| Too minimal | Missing critical context | Data structure + tool docs |
| Too verbose | Dilutes important info | Cut workflow instructions |

---

## Debugging Agent Behavior

When the agent does something wrong:

### Step 1: Check Tool Output
Is the tool returning what you expect? Is the format clear?

### Step 2: Check Tool Description
Does "Returns:" match what the tool actually returns?
Does "Use for:" make the tool's purpose clear?

### Step 3: Check Data Description
Does the agent know what files exist and where?

### Step 4: Last Resort
Only if the above are good: Add a constraint (not workflow).

Constraints are for SAFETY, not behavior:
- "Never delete files"
- "Maximum 10 tool calls per query"
- "Don't access paths outside papers/"

---

## Research Sources

**Anthropic - Building Effective Agents (2024)**
- Simplicity over complexity
- Tool documentation > workflow instructions
- Test with real inputs, iterate on tool design

**AgentBench (ICLR 2024)**
- Main failures: poor reasoning, decision-making, instruction following
- Fix: better tool design, not more instructions
- High-quality alignment data helps, more prompting doesn't

**Practical Observations**
- Claude Code uses minimal system prompts, excellent tool docs
- Devin focuses on environment clarity, not step-by-step guides
- Production agents fail when prompts are too prescriptive

---

## Final Checklist

Before deploying an agent prompt:

- [ ] Data structure is shown with actual examples
- [ ] Each tool has: description, returns format, use case
- [ ] No step-by-step workflows
- [ ] No rigid examples
- [ ] No "ALWAYS" or "NEVER" rules (except safety)
- [ ] Tested with varied queries, not just happy path
