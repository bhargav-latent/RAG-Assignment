# Agentic AI Tool Usage: The New Paradigm

A practical guide to how modern AI coding agents approach software engineering tasks.

---

## Core Principle: Structured Discovery Before Action

The fundamental shift in agentic AI is moving from **imperative commands** to **declarative goals** with autonomous tool orchestration.

```
Old: "grep -r 'error' src/ | head -20"
New: "Find where errors are handled" → Agent selects tools, iterates, synthesizes
```

---

## The Three-Phase Pattern

### Phase 1: Map (Broad → Narrow)

| Goal | Tool | Example |
|------|------|---------|
| Project structure | `ls`, `Glob` | `Glob("**/*.py")` |
| File patterns | `Glob` | `Glob("src/**/*test*.ts")` |
| Entry points | `Read` | `Read("package.json")` |

### Phase 2: Search (Pattern Matching)

| Goal | Tool | Example |
|------|------|---------|
| Content search | `Grep` | `Grep("async def", type="py")` |
| Definition lookup | `Grep` | `Grep("class UserService")` |
| Usage tracking | `Grep` | `Grep("UserService", output_mode="files_with_matches")` |

### Phase 3: Understand (Deep Reading)

| Goal | Tool | Example |
|------|------|---------|
| Full context | `Read` | `Read("src/auth.py")` |
| Specific section | `Read` | `Read("src/auth.py", offset=50, limit=30)` |
| Multiple files | Parallel `Read` | Batch related files together |

---

## Tool Selection Decision Tree

```
Need to find something?
├── Know exact path? → Read
├── Know filename pattern? → Glob
├── Know content pattern? → Grep
├── Uncertain/exploratory? → Task (subagent)
└── Need directory listing? → ls (Bash)

Need to change something?
├── Single edit? → Edit
├── New file? → Write
├── Multiple related edits? → Sequential Edit calls
└── Run command? → Bash
```

---

## Parallelization Rules

**Parallelize when:**
- Reading multiple independent files
- Searching with different patterns
- Running independent commands

**Serialize when:**
- Output of one informs the next
- File must exist before reading
- Command depends on prior result

```
# Parallel (no dependencies)
[Read file1] [Read file2] [Read file3]

# Sequential (has dependencies)
[Grep "pattern"] → [Read matching files] → [Edit based on content]
```

---

## The Subagent Pattern

For complex, multi-step exploration, delegate to specialized agents:

| Agent Type | Use Case | Tools Available |
|------------|----------|-----------------|
| `Explore` | Quick codebase search | Glob, Grep, Read (Haiku model) |
| `general-purpose` | Complex research | All tools (full model) |
| `Plan` | Architecture design | All tools |
| `Bash` | Command execution | Bash only |

**When to use subagents:**
- Open-ended questions ("How does auth work?")
- Multi-file investigation
- When you'd otherwise make 5+ sequential tool calls

---

## Task Tracking Integration

Every non-trivial task follows this pattern:

```
1. TodoWrite → Plan tasks
2. [Execute Phase 1-3 for each task]
3. TodoWrite → Mark in_progress
4. [Make changes]
5. TodoWrite → Mark completed
```

This creates:
- Visible progress for users
- Resumable work sessions
- Audit trail of decisions

---

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| `Bash("grep -r ...")` | `Grep("pattern")` |
| `Bash("cat file.py")` | `Read("file.py")` |
| `Bash("find . -name ...")` | `Glob("**/pattern*")` |
| Sequential reads of known files | Parallel `Read` calls |
| Editing without reading first | Always `Read` → then `Edit` |
| Guessing file locations | `Glob` or `Grep` to discover |

---

## The RAG Agent Application

In our RAG system, this paradigm maps directly:

```
User Query: "What does Figure 3 in the ML paper show?"

Agent Workflow:
1. Glob("papers/**/*.md") → Find paper directories
2. Grep("Figure 3", path="papers/") → Locate reference
3. Read(matching_page.md) → Get context
4. read_images(figure_path) → Analyze visual
5. Synthesize response with citations
```

---

## Key Insight

The new paradigm treats the AI agent as an **autonomous researcher** with a toolkit, not a command executor. The agent:

1. **Understands intent** from natural language
2. **Selects appropriate tools** based on the task
3. **Iterates** until sufficient information is gathered
4. **Synthesizes** findings into actionable output

This inverts the traditional CLI model: instead of users composing tool pipelines, the agent composes them dynamically based on goals.

---

*Generated for the RAG Assignment project - January 2026*
