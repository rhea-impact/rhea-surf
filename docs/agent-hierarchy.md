# Agent Hierarchy

## Architecture: Main Agent + Surf Subagents

```
┌─────────────────────────────────────────────────────────────┐
│                      MAIN AGENT                              │
│  (OpenCode with DeepSeek-R1)                                │
│                                                              │
│  Responsibilities:                                           │
│  - Understand user intent                                    │
│  - Break task into surf operations                          │
│  - Coordinate subagents                                      │
│  - Aggregate results                                         │
│  - Handle failures/retries                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  SURF AGENT 1   │ │  SURF AGENT 2   │ │  SURF AGENT 3   │
│  (Tab 1)        │ │  (Tab 2)        │ │  (Tab 3)        │
│                 │ │                 │ │                 │
│  - Single tab   │ │  - Single tab   │ │  - Single tab   │
│  - DOM reading  │ │  - DOM reading  │ │  - DOM reading  │
│  - Screenshots  │ │  - Screenshots  │ │  - Screenshots  │
│  - Actions      │ │  - Actions      │ │  - Actions      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Why This Architecture?

### 1. Separation of Concerns

**Main Agent:**
- High-level reasoning about the task
- Decides WHAT needs to happen
- Doesn't need to see DOM details
- Works with summaries from subagents

**Surf Subagents:**
- Low-level browser interaction
- Decides HOW to accomplish a specific action
- Sees full DOM/screenshots
- Reports results back to main agent

### 2. Context Window Management

Local models have limited context. By splitting:
- Main agent context: task + subagent summaries (~4k tokens)
- Surf agent context: single tab DOM + recent actions (~8k tokens)

Neither exceeds typical 32k limits.

### 3. Parallel Operations

Subagents can potentially work in parallel across tabs:
- Tab 1: Research product prices
- Tab 2: Check shipping options
- Tab 3: Prepare checkout form

Main agent coordinates and aggregates.

### 4. Failure Isolation

If a surf agent fails on one tab:
- Main agent can retry with fresh subagent
- Other tabs continue unaffected
- Checkpoints are per-tab

## Communication Protocol

### Main → Surf Agent

```json
{
  "task": "Find the 'Add to Cart' button and click it",
  "tab_id": "tab-123",
  "context": {
    "goal": "Purchase item X",
    "previous_action": "Navigated to product page"
  },
  "constraints": {
    "max_steps": 5,
    "timeout_seconds": 30
  }
}
```

### Surf Agent → Main

```json
{
  "status": "completed",
  "result": {
    "action_taken": "Clicked 'Add to Cart' button",
    "page_state": "Cart now shows 1 item",
    "screenshot_summary": "Product page with cart count = 1"
  },
  "tokens_used": 1234,
  "steps_taken": 2
}
```

## Implementation Options

### Option A: OpenCode Native

If OpenCode supports spawning subagents:
- Main agent runs in OpenCode
- Subagents are OpenCode subprocess calls
- Communication via stdout/stdin

### Option B: Custom Python Orchestrator

Build `surf/orchestrator.py`:
- Main agent logic in Python
- Spawns OpenCode processes for surf agents
- Or uses Ollama API directly for subagents

### Option C: Single Agent, Sequential

Simpler fallback:
- One agent handles everything
- Works on one tab at a time
- Clears context between tab switches

## Recommended Approach

Start with **Option C** (single agent, sequential) for validation.
Evolve to **Option B** (Python orchestrator) if:
- Multi-tab parallelism is needed
- Context limits are hit
- Complex coordination required

## Model Assignment

| Role | Model | Reason |
|------|-------|--------|
| Main Agent | qwen3:14b | Better reasoning for planning |
| Surf Agent | llama3.1:8b | Faster, sufficient for simple actions |
| Vision | llama3.2-vision | Screenshot analysis |

Main agent can use larger model since it doesn't process DOM directly.

## Hierarchical Models for Context Reduction

**Key insight from testing:** Local models struggle with maintaining task state across steps.

**Solution:** Use hierarchical model calls:

```
┌─────────────────────────────────────────────────────────┐
│  COORDINATOR (cheap/fast model - llama3)                │
│  - Receives task                                        │
│  - Breaks into subtasks                                 │
│  - Routes to specialists                                │
└─────────────────┬───────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐   ┌────────┐   ┌────────┐
│READER  │   │CLICKER │   │FORM    │
│llama3  │   │llama3  │   │FILLER  │
│        │   │        │   │qwen3   │
│One-shot│   │One-shot│   │        │
│extract │   │click   │   │Multi-  │
└────────┘   └────────┘   │step    │
                          └────────┘
```

Benefits:
1. Each model call has minimal context (just one subtask)
2. Coordinator maintains overall state
3. Specialists are stateless - can be cheap/fast
4. Context never accumulates across steps
