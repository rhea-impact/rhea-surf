# Architecture

## Overview

rhea-surf is a browser automation system that uses local LLMs (via Ollama) to navigate websites and extract information. It prioritizes correctness over speed, with multiple fallback mechanisms.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              rhea-surf                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ SurfAgent   │───►│ Navigator   │───►│   Debate    │───►│  Execute    │  │
│  │             │    │             │    │ (if needed) │    │             │  │
│  └─────────────┘    └──────┬──────┘    └─────────────┘    └─────────────┘  │
│                            │                                                 │
│         ┌──────────────────┼──────────────────┐                             │
│         ▼                  ▼                  ▼                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │ Action Cache│    │   Learner   │    │    LLM      │                      │
│  │ (SQLite)    │    │ (ChromaDB)  │    │  (Ollama)   │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
│                                                │                             │
│                                         ┌──────┴──────┐                      │
│                                         ▼             ▼                      │
│                                  ┌───────────┐ ┌───────────┐                 │
│                                  │ Trajectory│ │  Vision   │                 │
│                                  │  Memory   │ │ Fallback  │                 │
│                                  └───────────┘ └───────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         │ Playwright
         ▼
┌─────────────────┐
│    Browser      │
│  (Chromium)     │
└─────────────────┘
```

## Decision Pipeline

When the agent receives a task, it follows this decision pipeline:

```
                                   Task
                                    │
                                    ▼
                         ┌──────────────────┐
                         │   Cache Check    │──── hit ────► Replay Action
                         └────────┬─────────┘
                                  │ miss
                                  ▼
                         ┌──────────────────┐
                         │  Pattern Match   │──── match ──► Use Pattern
                         └────────┬─────────┘
                                  │ no match
                                  ▼
                         ┌──────────────────┐
                         │  LLM Decision    │◄──── Trajectory Memory
                         └────────┬─────────┘      (few-shot examples)
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Confidence Check │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │ < 0.7                     │ >= 0.7
                    ▼                           ▼
           ┌──────────────────┐        ┌──────────────────┐
           │  Multi-Agent     │        │    Execute       │
           │    Debate        │        │    Action        │
           └────────┬─────────┘        └──────────────────┘
                    │
                    ▼
           ┌──────────────────┐
           │  Execute with    │
           │   Consensus      │
           └──────────────────┘
```

## Components

### SurfAgent (`surf/agent.py`)

Entry point for browser automation. Manages browser lifecycle and task execution.

```python
agent = SurfAgent()
result = await agent.run("What is the top story on Hacker News?")
```

### RecursiveNavigator (`surf/navigator.py`)

Core decision logic. Handles:
- DOM state analysis
- Action selection
- Loop detection
- Depth management

Key features:
- **Loop detection**: Breaks infinite action loops after 3+ same-page calls
- **Content extraction**: Smart selection of content elements vs navigation
- **Force-done safeguard**: Prevents endless exploration

### DOM Extraction (`surf/dom.py`)

Two extraction modes:

1. **Playwright JS injection** (primary)
   - Executes `buildDomTree.js` in browser
   - Paint-order indexing
   - Computed styles and visibility
   - Viewport filtering

2. **BeautifulSoup static** (fallback)
   - Fast, no browser needed
   - Misses JS-rendered content

Output format for LLM:
```
[1] link "Top Story Title" href="https://..."
[2] link "Comments" [NAV]
[3] button "Sign In" [NAV]
[4] input[text] placeholder="Search"
```

### Action Cache (`surf/cache.py`)

Deterministic replay of successful actions.

```python
cache = ActionCache()

# Check for cached action
cached = cache.get(task, url, dom_hash)
if cached:
    return cached.action  # Replay without LLM

# Store successful action
cache.store(task, url, dom_hash, action, success=True)
```

Cache key: `hash(task + domain + dom_hash)`

### Pattern Learner (`surf/learner.py`)

ChromaDB-backed pattern storage. Learns from successful runs.

```python
learner = Learner()

# Store pattern
learner.store_pattern(
    url_pattern="news.ycombinator.com",
    task_pattern="read top story",
    action="click",
    selector="[1]"
)

# Search patterns
matches = learner.search_patterns("get first headline", limit=3)
```

### Trajectory Memory (`surf/memory.py`)

Few-shot learning from complete task trajectories.

```python
memory = TrajectoryMemory()

# Retrieve similar successful runs
examples = memory.get_relevant(task, url, limit=3)

# Add to LLM prompt as few-shot examples
prompt = f"""
Previous successful runs:
{format_examples(examples)}

Current task: {task}
"""
```

### Multi-Agent Debate (`surf/debate.py`)

Multiple models vote on uncertain decisions.

```python
debate = Debate(models=["llama3:latest", "llama3.2:latest", "qwen3:14b"])

result = await debate.run(
    task=task,
    dom_state=state,
    initial_action=action
)
# Returns consensus action with vote breakdown
```

### RSA Reasoning (`surf/recursive.py`)

Recursive Self-Aggregation for complex decisions.

- Multi-model: llama3, qwen3, llama3.2
- Multi-language: English, Chinese, Spanish
- Aggregates 9 candidates (3 models × 3 languages)

### Vision Fallback (`surf/vision.py`)

Uses llama3.2-vision when DOM extraction fails.

Triggers:
- DOM has <5 interactive elements
- Action confidence <0.5
- Action failed multiple times

```python
if should_use_vision(dom_element_count, confidence):
    action = await vision_decide(page, task)
```

### Meta-Study Framework (`surf/study.py`)

Self-improvement tracking with loss functions.

```python
runner = StudyRunner()
session = await runner.run_session()

loss = LossMetrics.from_session(session)
print(f"Total loss: {loss.total_loss}")  # Lower is better
```

See [meta-study.md](meta-study.md) for details.

## Data Flow: Example Task

```
1. User: "What is the top story on Hacker News?"

2. SurfAgent creates Navigator, navigates to news.ycombinator.com

3. Navigator extracts DOM:
   [1] link "Hacker News" [NAV]
   [2] link "Some Interesting Article" href="https://..."
   [3] link "123 points" [NAV]
   ...

4. Navigator checks cache → miss

5. Navigator checks patterns → miss

6. Navigator retrieves trajectory examples from memory

7. Navigator asks LLM:
   "Task: What is the top story?
    Elements: [1] link "Some Interesting Article"...
    Previous examples: [similar successful runs]
    What action?"

8. LLM returns: read element [2]

9. Navigator extracts text from element [2]

10. LLM returns: done "Some Interesting Article"

11. Navigator caches successful action

12. Returns result to user
```

## Configuration

### Models (Ollama)

| Model | Purpose | Size |
|-------|---------|------|
| llama3:latest | Primary decisions | 8B |
| llama3.2:latest | Debate participant | 3B |
| qwen3:14b | Debate participant | 14B |
| llama3.2-vision | Vision fallback | 11B |

### Storage

| Component | Location |
|-----------|----------|
| Action Cache | `~/.rhea-surf/cache.db` |
| Patterns | `~/.rhea-surf/patterns/` (ChromaDB) |
| Trajectories | `~/.rhea-surf/trajectories.db` |
| Study Sessions | `~/.rhea-surf/study.db` |

## Performance Characteristics

### Token Efficiency

| Data | Tokens | Notes |
|------|--------|-------|
| Screenshot (base64) | ~1500 | Vision only |
| Raw DOM | 3000+ | Too large |
| Simplified DOM | ~200-400 | Our approach |
| Few-shot examples | ~300 | 3 examples |

### Latency

| Operation | Time |
|-----------|------|
| DOM extraction | ~100ms |
| LLM decision | 2-5s |
| Cache hit | <10ms |
| Pattern match | ~50ms |
| Vision fallback | 5-10s |

### Correctness

- Simple tasks: ~100% success
- Medium tasks: ~33% success (needs improvement)
- Cache hit rate: ~17% (grows with usage)
