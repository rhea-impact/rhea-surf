# rhea-surf

Browser automation for local LLMs. $0 inference cost via Ollama.

## What It Does

Autonomous browser agent that navigates websites and extracts information using local models only. Designed for correctness over speed.

```
┌─────────────────┐      MCP         ┌─────────────────┐
│   SurfAgent     │◄────────────────►│  Helios Server  │
│  (Playwright)   │                  │                 │
└─────────────────┘                  └────────┬────────┘
        │                                     │
        │ Ollama API                   Native Messaging
        ▼                                     ▼
┌─────────────────┐                  ┌─────────────────┐
│  Local Models   │                  │ Chrome Extension│
│  llama3, qwen3  │                  │  (your browser) │
│  llama3.2-vision│                  └─────────────────┘
└─────────────────┘
```

## Goals

### Short-Term
- **85%+ task completion** on common browser tasks (currently 100% simple, 33% medium)
- **40%+ cache hit rate** for deterministic replay (currently 17%)
- **Sub-10 second** average task completion for simple tasks
- Robust handling of JS-heavy SPAs via vision fallback

### Long-Term
- **Fully autonomous web agent** that can complete multi-step workflows (shopping, form filling, research)
- **Zero cloud dependency** - all inference runs locally on consumer hardware
- **Self-improving system** that learns from every interaction and measurably improves over time
- **Open benchmark suite** for comparing local browser automation approaches
- **Plugin ecosystem** for site-specific adapters and custom actions

## Quick Start

```bash
# Install
pip install -e .

# Run a task
python -m surf.cli "What is the top story on Hacker News?"

# Run study session (self-improvement tracking)
python -m surf.study
```

## Components

| Component | Status | Description |
|-----------|--------|-------------|
| DOM Extraction | ✅ Complete | JS injection via Playwright, paint-order indexing |
| Action Cache | ✅ Complete | Deterministic replay of successful actions |
| Pattern Learner | ✅ Complete | Learns from successful runs, ChromaDB-backed |
| Trajectory Memory | ✅ Complete | Few-shot learning from past runs |
| Multi-Agent Debate | ✅ Complete | Multiple models vote on uncertain decisions |
| RSA Reasoning | ✅ Complete | Multi-model + multi-language aggregation |
| Vision Fallback | ✅ Complete | llama3.2-vision when DOM fails |
| Meta-Study | ✅ Complete | Self-improvement tracking with loss functions |

## Architecture

### Decision Pipeline

```
Task → Cache Check → Pattern Match → LLM Decision → Debate (if uncertain) → Execute
         ↓ hit           ↓ match
      Replay          Use Pattern
```

### Key Files

```
surf/
├── agent.py       # Main SurfAgent class
├── navigator.py   # RecursiveNavigator - decision logic
├── dom.py         # DOM extraction and formatting
├── cache.py       # Action cache for deterministic replay
├── learner.py     # Pattern learning (ChromaDB)
├── memory.py      # Trajectory memory (few-shot)
├── debate.py      # Multi-agent debate
├── recursive.py   # RSA reasoning aggregation
├── vision.py      # Vision fallback (llama3.2-vision)
├── study.py       # Meta-study framework
└── js/
    └── buildDomTree.js  # DOM extraction script
```

## Self-Improvement Tracking

The meta-study framework tracks performance with loss functions:

```bash
# Run benchmark tasks
python -m surf.study

# View trends
python -m surf.study --history

# Deep analysis
python -m surf.study --analyze
```

### Loss Functions

| Component | Weight | Measures |
|-----------|--------|----------|
| Task Failure | 50% | 1 - success_rate |
| Efficiency | 25% | LLM calls above target |
| Speed | 15% | Time above target |
| Cache Miss | 10% | 1 - cache_utilization |

**Target total loss: < 0.10** (excellent)

See [docs/meta-study.md](docs/meta-study.md) for details.

## Design Principles

1. **Correctness over speed** - Autonomous operation that's "mostly correct" even if slow
2. **Deterministic when possible** - Cache hits and patterns before LLM calls
3. **Learn from success** - Trajectory memory and pattern promotion
4. **Multi-model consensus** - Debate reduces errors on uncertain decisions
5. **Vision as fallback** - Don't fail on JS-heavy SPAs

## Models Used

| Model | Purpose |
|-------|---------|
| `llama3:latest` | Primary decision making |
| `llama3.2:latest` | Secondary model for debate |
| `qwen3:14b` | Third model for consensus |
| `llama3.2-vision` | Vision fallback |

All via Ollama. No cloud API calls.

## Current Performance

```
Pass Rate:        100% (simple tasks)
Total Loss:       ~0.11
Cache Utilization: 17%
Trend:            STABLE
```

## Documentation

- [Architecture](docs/architecture.md) - System design
- [Meta-Study](docs/meta-study.md) - Self-improvement framework
- [Design Principles](docs/design-principles.md) - Why decisions were made
- [Models](docs/models.md) - Model recommendations

## Related Projects

- [Helios](~/repos-aic/helios) - Browser automation MCP server
- [Browser Use](https://github.com/browser-use/browser-use) - Inspiration for DOM extraction
