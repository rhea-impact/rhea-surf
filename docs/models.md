# Model Recommendations

## Requirements for Browser Automation

Browser automation agents need:
1. **Tool-use ability** - Calling functions with correct parameters
2. **Planning** - Multi-step reasoning for complex workflows
3. **DOM comprehension** - Understanding page structure from text
4. **Instruction following** - Executing user intent accurately

## Recommended Models

### Primary: DeepSeek-R1-Distill (8B-14B)

```bash
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:14b  # if GPU allows
```

**Why:**
- Distilled reasoning model - outperforms larger models on complex reasoning
- Strong tool-use from training
- Good at multi-step planning
- Efficient for local inference

**Best for:** Main agent brain, deciding which actions to take

### Fallback: Llama 3 Instruct (8B)

```bash
ollama pull llama3:8b-instruct
```

**Why:**
- SOTA among open models for instruction following
- Smoother language output
- Wide ecosystem support
- Good coding/technical ability

**Best for:** When you need more conservative/reliable behavior

### Lightweight: Mistral Instruct (7B)

```bash
ollama pull mistral:7b-instruct
```

**Why:**
- Best quality-per-FLOP
- Runs on modest hardware
- Good for simpler automation tasks

**Best for:** Resource-constrained setups, many concurrent agents

## Context Length Considerations

| Model | Context | Notes |
|-------|---------|-------|
| DeepSeek-R1 8B | 32K | Plenty for DOM + history |
| Llama 3 8B | 8K | May need DOM simplification |
| Mistral 7B | 32K | Good context despite size |

## Hardware Requirements

### Minimum (7B models)
- 8GB VRAM (GPU) or 16GB RAM (CPU)
- Apple M1/M2 works well

### Recommended (8B-14B models)
- 16GB+ VRAM or 32GB RAM
- Apple M2 Pro/Max ideal

## Prompt Engineering Tips

1. **Keep DOM summaries short** - 200-400 tokens max
2. **Use strict tool schemas** - Enum values, required fields
3. **Include examples** - Show expected tool call format
4. **One action at a time** - Don't ask for multi-step plans in one call

## Example Tool Definition

```json
{
  "name": "click",
  "description": "Click an element on the page",
  "parameters": {
    "type": "object",
    "properties": {
      "element_id": {
        "type": "integer",
        "description": "Element index from read_page output, e.g. 1 for [1]"
      },
      "selector": {
        "type": "string",
        "description": "CSS selector (use element_id when available)"
      }
    },
    "required": ["element_id"]
  }
}
```
