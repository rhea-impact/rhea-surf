# rhea-surf

Browser automation adapter for local LLMs via opencode.

## What This Is

A bridge that enables browser automation (via Helios) to work with local/offline models through opencode, avoiding cloud token limits.

## Project Structure

```
rhea-surf/
├── docs/
│   ├── architecture.md    # System design
│   ├── models.md          # Model recommendations
│   └── research/          # Investigation notes
├── surf/                  # Python package (if needed)
│   ├── dom.py            # DOM simplification
│   ├── agent.py          # Agent loop wrapper
│   └── cli.py            # Optional CLI
└── configs/              # Example configurations
```

## Dependencies

- **Helios** (`~/repos-aic/helios`) - Browser automation MCP server
- **opencode** - Local LLM CLI with MCP support
- **Ollama** - Local model runtime

## Quick Reference

### Helios Tools Available

```
tabs_list          - List open tabs
navigate           - Go to URL
read_page          - Get structured DOM
screenshot         - Capture visible area
click              - Click elements
type               - Type text
scroll             - Scroll page
site_knowledge_*   - Learning/memory
```

### Recommended Models (Ollama)

```bash
ollama pull deepseek-r1:8b      # Primary - reasoning
ollama pull llama3:8b-instruct  # Fallback - smooth
ollama pull mistral:7b-instruct # Lightweight
```

## Development Notes

- This is primarily configuration + documentation
- Only write code if DOM simplification or agent loop is needed
- Test with Helios first to understand token usage
