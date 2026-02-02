# OpenCode Validation - CONFIRMED

**Date:** 2025-02-02
**Status:** ✅ Validated - OpenCode supports our requirements

## Key Findings

### MCP Server Support ✅

OpenCode implements MCP protocol for external tools. Configuration:

```json
{
  "mcp": {
    "helios": {
      "type": "local",
      "command": ["node", "/path/to/helios/packages/server/dist/index.js"]
    }
  }
}
```

- **Local servers**: Use `command` array (stdio transport)
- **Remote servers**: Use `url` with optional OAuth
- **Tool discovery**: Automatic - tools appear alongside built-in tools
- **Permissions**: User approval required before execution

### Ollama Support ✅

OpenCode works with Ollama via OpenAI-compatible API:

```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "deepseek-r1:8b": {
          "name": "DeepSeek R1 8B",
          "limit": {
            "context": 32000,
            "output": 8192
          }
        }
      }
    }
  }
}
```

**Critical note:** If tool calls aren't working, increase `num_ctx` in Ollama to 16k-32k.

### Context Window Warning ⚠️

From Composio research: "Adding even a single GitHub server will take 20k tokens from your context window."

**Implication:** Helios tool definitions will consume context. Local 8B models with ~32k context may struggle if Helios exposes many tools.

## Gaps Identified (from Gemini analysis)

### Technical Risks
1. **Context overload** - History + DOM + screenshots could exceed 8k-32k limits
2. **DOM brittleness** - Element indices invalidate on re-render
3. **Vision hallucinations** - LLaVA may misinterpret for verification

### Missing Considerations
1. **Security** - Need domain allowlist/denylist for browser control
2. **Observability** - Structured logging for debugging agent failures
3. **State persistence** - Checkpoints should persist to disk

### Alternative Approach
If OpenCode's agent loop is insufficient, build custom Python agent:
- Use `ollama` Python client directly
- Connect to Helios via MCP client library
- Full control over perceive→decide→act loop

## Architecture Validated

```
┌─────────────────┐      MCP         ┌─────────────────┐
│    OpenCode     │◄────────────────►│     Helios      │
│  (Go CLI/TUI)   │                  │  (MCP Server)   │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
         │ OpenAI-compatible API       Native Messaging
         ▼                                    ▼
┌─────────────────┐                  ┌─────────────────┐
│     Ollama      │                  │ Chrome Extension│
│  DeepSeek-R1    │                  │  (your browser) │
│  LLaVA (vision) │                  └─────────────────┘
└─────────────────┘
```

## Next Steps

1. ✅ OpenCode MCP support - VALIDATED
2. ⏳ Install OpenCode and configure Helios
3. ⏳ Test with DeepSeek-R1 8B via Ollama
4. ⏳ Profile token usage on real pages
5. ⏳ Add LLaVA for vision if needed

## Sources

- [OpenCode MCP Servers Docs](https://opencode.ai/docs/mcp-servers/)
- [OpenCode Providers Docs](https://opencode.ai/docs/providers/)
- [Setting Up OpenCode with Local Models](https://theaiops.substack.com/p/setting-up-opencode-with-local-models)
- [MCP with OpenCode - Composio](https://composio.dev/blog/mcp-with-opencode)
