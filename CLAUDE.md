# rhea-surf

Standalone browser automation for local LLMs.

## What This Is

A complete browser automation system that runs entirely offline using Ollama. No external dependencies, no cloud APIs, no data leaving your machine.

## Project Structure

```
rhea-surf/
├── surf/                  # Core Python package
│   ├── agent.py          # SurfAgent entry point
│   ├── navigator.py      # Decision logic
│   ├── dom.py            # DOM extraction
│   ├── cache.py          # Action caching
│   ├── learner.py        # Pattern learning
│   ├── memory.py         # Trajectory memory
│   ├── debate.py         # Multi-agent debate
│   ├── recursive.py      # RSA reasoning
│   ├── vision.py         # Vision fallback
│   ├── study.py          # Self-improvement tracking
│   └── js/
│       └── buildDomTree.js
├── tests/                 # Test suite
├── docs/                  # Documentation
│   ├── architecture.md   # System design
│   ├── meta-study.md     # Loss functions
│   └── research/         # Historical notes
└── learned/              # ChromaDB patterns (gitignored)
```

## Dependencies

- **Playwright** - Browser automation
- **Ollama** - Local model runtime
- **ChromaDB** - Pattern storage

## Models Used

```bash
ollama pull llama3.1:8b           # Primary decisions
ollama pull gemma3n:e4b         # Debate participant
ollama pull qwen3:14b        # Debate participant
ollama pull gemma3n:e4b-vision  # Vision fallback
```

## Running

```bash
# Single task
python -m surf.cli "What is the top story on Hacker News?"

# Study session (self-improvement)
python -m surf.study

# Tests
pytest tests/
```

## Key Design Decisions

1. **Playwright over Chrome extension** - More reliable, works headless
2. **Multiple models for consensus** - Reduces errors on uncertain decisions
3. **Cache-first architecture** - Deterministic replay when possible
4. **Vision fallback** - Handle JS-heavy SPAs that break DOM extraction
5. **Self-tracking** - Loss functions measure improvement over time

## Development Notes

- All inference is local via Ollama
- No cloud API calls anywhere in the codebase
- Research docs reference Helios (prior art we learned from, not a dependency)
