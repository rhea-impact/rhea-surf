# rhea-surf

Browser automation for local/offline LLMs via opencode.

## Problem

Perplexity Comet's browser agent hits token limits on complex workflows. We need browser automation that works with **local models** (Ollama, LM Studio) to avoid cloud rate limits.

## Solution

Leverage existing infrastructure:
- **Helios** - Already have MCP server + Chrome extension for browser control
- **opencode** - CLI tool like Claude Code but for open-source models
- **Local LLMs** - DeepSeek-R1-Distill, Llama 3, Mistral via Ollama

## Architecture

```
┌─────────────────┐      MCP         ┌─────────────────┐
│    opencode     │◄────────────────►│  Helios Server  │
│  (local LLM)    │                  │                 │
└─────────────────┘                  └────────┬────────┘
        │                                     │
        │ Ollama API                   Native Messaging
        ▼                                     ▼
┌─────────────────┐                  ┌─────────────────┐
│  Local Model    │                  │ Chrome Extension│
│  (DeepSeek/     │                  │  (your browser) │
│   Llama 3)      │                  └─────────────────┘
└─────────────────┘
```

## Status

✅ **Validated** - OpenCode confirmed to support MCP servers + Ollama. Ready for implementation.

See [docs/research/opencode-validated.md](docs/research/opencode-validated.md) for full research findings.

## Components

| Component | Status | Location |
|-----------|--------|----------|
| Helios MCP Server | ✅ Exists | `~/repos-aic/helios` |
| Chrome Extension | ✅ Exists | Helios package |
| opencode | ✅ Validated | [opencode.ai](https://opencode.ai) |
| Example Config | ✅ Created | `configs/opencode.example.json` |
| DOM Simplifier | 📋 If needed | This repo |
| Agent Loop | 📋 If needed | This repo |

## Model Recommendations

For browser automation agents, prioritize tool-use and planning:

| Model | Size | Strengths | Use Case |
|-------|------|-----------|----------|
| DeepSeek-R1-Distill | 8-14B | Reasoning, planning | Primary agent brain |
| Llama 3 Instruct | 8B | Instruction following | Fallback, smoother language |
| Mistral Instruct | 7B | Efficient | Resource-constrained |

## Key Insights

From research on local browser agents:

1. **DOM must be simplified** - Raw DOM destroys token budgets. Need semantic compression.
2. **Strict tool schemas** - Local models need tighter constraints than Claude.
3. **Element indexing** - Give clickable elements IDs like `[1]`, `[2]` for easy reference.
4. **Retry at code level** - Let automation framework handle robustness, not the model.

## Next Steps

1. [ ] Verify opencode MCP support
2. [ ] Test Helios with opencode
3. [ ] Profile token usage on real pages
4. [ ] Build DOM simplifier if needed
5. [ ] Create agent loop wrapper

## Related Projects

- [Helios](~/repos-aic/helios) - Browser automation MCP server
- [reeves-web](~/repos-personal/reeves-web) - Task management with MCP
- [BrowserOS](https://github.com/anthropics/browserOS) - Reference for local browser agents
