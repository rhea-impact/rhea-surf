# Design Principles

## Core Philosophy

**Slower is fine. Smaller chunks are better.**

Local models have different tradeoffs than cloud APIs:
- ✅ No token limits or rate limits
- ✅ No cost per call
- ❌ Slower inference
- ❌ Smaller context windows
- ❌ Need more guidance

## Principles

### 1. Break Everything Into Small Steps

Don't ask the model to plan a 10-step workflow. Instead:

```
❌ "Log into bank, navigate to statements, download last 3 months"

✅ Step 1: "You're on bank.com homepage. What's the first action?"
✅ Step 2: "Login form visible. What should we fill in?"
✅ Step 3: "Logged in. Where do we go for statements?"
...
```

**Why:** Local models excel at single decisions, struggle with long-horizon planning.

### 2. Use Vision Models

Screenshots are essential, not optional:

- Local vision models (LLaVA, Qwen-VL) can interpret screenshots
- Visual verification catches DOM parsing errors
- Some UI elements are hard to describe in text

**Pattern:**
```
1. Take screenshot
2. Ask vision model: "What do you see? What elements are clickable?"
3. Execute action
4. Take new screenshot to verify
```

### 3. Multi-Tab Context

Real workflows span multiple tabs:
- Research in one tab, form in another
- Compare prices across sites
- Copy data between applications

**Requirements:**
- Track which tab is active
- Switch context cleanly
- Summarize inactive tabs (don't load full DOM)

### 4. Checkpoint Often

After each action, create a checkpoint:
- Current URL
- Screenshot
- Key page state
- What we just did
- What we're trying to accomplish

If the model gets confused, roll back to last checkpoint.

### 5. Let the Model Be Slow

Don't optimize for speed. Optimize for:
- Correctness
- Recoverability
- Clear reasoning

A 30-second workflow that works > a 5-second workflow that fails.

## Architecture Implications

### Agent Loop

```python
while not done:
    # 1. Perceive (screenshot + minimal DOM)
    screenshot = take_screenshot()
    dom_summary = get_minimal_dom()  # Just interactive elements

    # 2. Orient (what are we doing?)
    context = {
        "goal": original_task,
        "step": current_step,
        "history": last_3_actions,
        "tabs": tab_summaries,
    }

    # 3. Decide (single action)
    action = ask_model(screenshot, dom_summary, context)

    # 4. Act
    result = execute_action(action)

    # 5. Verify
    new_screenshot = take_screenshot()
    verified = ask_model("Did the action succeed?", new_screenshot)

    # 6. Checkpoint
    save_checkpoint(url, screenshot, action, result)
```

### Vision Model Integration

Need to support:
- `ollama pull llava:13b` - Good general vision
- `ollama pull qwen2-vl:7b` - Efficient, good at UI

### Tab Management

```
Tab 1: [Active] bank.com/statements - "Statement download page"
Tab 2: [Background] docs.google.com - "Expense spreadsheet"
Tab 3: [Background] mail.google.com - "Gmail inbox"
```

Keep tab summaries short. Only load full DOM for active tab.
