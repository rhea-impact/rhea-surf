# Architecture

## Overview

rhea-surf connects local LLMs to browser automation through the MCP protocol.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User's Machine                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐ │
│  │   opencode   │◄──MCP──►│    Helios    │◄──WS───►│   Chrome     │ │
│  │   (client)   │         │   (server)   │         │  Extension   │ │
│  └──────┬───────┘         └──────────────┘         └──────┬───────┘ │
│         │                                                  │        │
│         │ Ollama API                              chrome.* APIs     │
│         ▼                                                  ▼        │
│  ┌──────────────┐                                 ┌──────────────┐  │
│  │    Ollama    │                                 │   Browser    │  │
│  │  (runtime)   │                                 │  (logged in) │  │
│  │              │                                 │              │  │
│  │ DeepSeek-R1  │                                 │  - Sessions  │  │
│  │ Llama 3     │                                 │  - Cookies   │  │
│  │ Mistral      │                                 │  - State     │  │
│  └──────────────┘                                 └──────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### opencode (MCP Client)
- Hosts the local LLM conversation
- Calls MCP tools based on model decisions
- Manages the agent loop (perceive → decide → act)

### Helios MCP Server
- Exposes browser automation tools via MCP protocol
- Bridges to Chrome extension via native messaging
- Handles site knowledge and guides

### Chrome Extension
- Executes browser actions (click, navigate, read DOM)
- Runs in your actual logged-in browser
- Communicates via WebSocket to native host

### Local LLM (via Ollama)
- Makes decisions about which actions to take
- Interprets DOM structure
- Plans multi-step workflows

## Data Flow: Example Workflow

```
1. User: "Log into my bank and download last month's statement"

2. opencode → LLM: "What tool should I call first?"

3. LLM: "Call navigate(url='bank.com')"

4. opencode → Helios: navigate(url='bank.com')

5. Helios → Extension: NAVIGATE bank.com

6. Extension → Browser: chrome.tabs.update(...)

7. Extension → Helios: {success: true, url: 'bank.com/home'}

8. Helios → opencode: {result: 'navigated to bank.com/home'}

9. opencode → LLM: "Navigation complete. What next?"

10. LLM: "Call read_page() to see the current state"

... continues until task complete
```

## Token Efficiency

| Approach | Tokens | Notes |
|----------|--------|-------|
| Screenshot (base64) | ~1500 | Requires vision model |
| Raw DOM | ~3000+ | Usually too large |
| Helios read_page | ~400 | Structured, semantic |
| Simplified DOM | ~200 | If we build it |

## Optional: DOM Simplifier

If Helios's `read_page` is still too large for 8B models:

```python
# surf/dom.py
def simplify_dom(structured_dom: dict) -> str:
    """
    Reduce DOM to essentials:
    - Interactive elements only (buttons, links, inputs)
    - Element indices for reference: [1] Login [2] Password
    - Strip decorative content
    """
    pass
```

## Optional: Agent Loop

If opencode doesn't handle the perceive→decide→act loop:

```python
# surf/agent.py
def run_agent(task: str, mcp_client, llm_client):
    """
    Simple agent loop:
    1. Call read_page() to perceive
    2. Ask LLM what action to take
    3. Execute the action via MCP
    4. Repeat until done or stuck
    """
    pass
```
