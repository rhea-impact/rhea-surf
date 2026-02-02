# Initial Research: Browser Automation with Local LLMs

**Date:** 2025-02-02
**Source:** Perplexity research session

## Problem Statement

Perplexity Comet's browser agent hits token limits on complex workflows. Need browser automation that works with local/offline models.

## Key Findings

### 1. Architecture Pattern

The standard pattern for local browser automation:

```
Headless Browser (Playwright/Selenium)
        ↕
Local LLM Runtime (Ollama/LM Studio)
        ↕
Tooling Layer (MCP or custom API)
```

The model never talks to browser directly - it gets structured DOM and returns tool calls.

### 2. Existing Projects

- **BrowserOS** - Open-source Chromium fork connecting to local LLM (Ollama on localhost:11434)
- **Browser-automation MCP servers** - Generic MCP servers for web agents
- **LLM-augmented Playwright** - Pattern of LLM for decisions, Playwright for execution

### 3. Can Local Models Handle It?

**Yes**, for most workflows. Requirements:
- Reasonable tool-use/planning ability
- DOM reasoning from text
- Multi-step workflow memory

**What helps:**
- Keep DOM summarized/chunked (not raw dump)
- Strict, schema-bound tool definitions
- Let code handle robustness (retries, timeouts, waiting)

### 4. Model Recommendations

| Model | Size | Use Case |
|-------|------|----------|
| DeepSeek-R1-Distill | 8-14B | Primary agent brain |
| Llama 3 Instruct | 8B | Fallback/conservative |
| Mistral Instruct | 7B | Resource-constrained |

30B+ models improve reliability but aren't required for typical line-of-business flows.

### 5. Our Advantage

We already have **Helios** which:
- Connects to real browser (logged-in sessions)
- Exposes MCP tools
- Has site knowledge/learning
- Token-efficient DOM (~400 tokens vs ~1500 for screenshots)

## Next Steps

1. Verify opencode supports MCP
2. Configure opencode to use Helios
3. Test with DeepSeek-R1 8B via Ollama
4. Profile actual token usage on real pages
5. Build DOM simplifier only if needed

## Questions to Answer

- [ ] Does opencode support MCP servers?
- [ ] What's opencode's agent loop like?
- [ ] How does Helios `read_page` compare to what 8B models can handle?
- [ ] Do we need vision (screenshots) or is text DOM sufficient?
