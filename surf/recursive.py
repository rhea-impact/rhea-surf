"""
Recursive reasoning inspired by RLM, RSA, and LADDER.

Key insights from research:
1. RLM: Don't bloat context. Delegate to sub-LLMs. Iterative answer refinement.
2. RSA: Keep population of candidates. Aggregate good reasoning from bad answers.
3. LADDER: Break hard into easy. Solve easy first. Build up curriculum.

Applied to browser automation:
- Population of action candidates (not just one)
- Extract good reasoning even from failed attempts
- Decompose complex tasks into simpler sub-tasks
- Multi-language debate for richer human understanding
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
import ollama

logger = logging.getLogger(__name__)


@dataclass
class ReasoningCandidate:
    """A candidate action with its reasoning chain."""
    action_type: str
    selector: Optional[str]
    value: Optional[str]
    reasoning: str
    language: str = "en"
    confidence: float = 0.5

    # RSA: Extract useful reasoning even from wrong answers
    useful_observations: list[str] = field(default_factory=list)


@dataclass
class AggregatedDecision:
    """Result of aggregating multiple reasoning candidates."""
    action_type: str
    selector: Optional[str]
    value: Optional[str]
    reasoning: str
    confidence: float
    candidates_considered: int
    consensus_level: float  # 0-1, how much agreement


# Multi-language prompts for richer human understanding
PROMPTS_BY_LANGUAGE = {
    "en": """Task: {task}
Current page: {url} - {title}

Interactive elements:
{elements}

Choose ONE action. Available actions: click, fill, scroll, read, done, navigate

You MUST respond with ONLY this JSON (no other text):
```json
{{"action": "click", "selector": "[1]", "value": null, "reasoning": "clicking element 1 because..."}}
```

Replace values appropriately. For "done", put the answer in "value".""",

    "zh": """任务：{task}
当前页面：{url} - {title}

交互元素：
{elements}

选择一个动作。可用动作：click, fill, scroll, read, done, navigate

只返回这个JSON格式（不要其他文字）：
```json
{{"action": "click", "selector": "[1]", "value": null, "reasoning": "点击元素1因为..."}}
```""",

    "es": """Tarea: {task}
Página actual: {url} - {title}

Elementos interactivos:
{elements}

Elige UNA acción. Acciones disponibles: click, fill, scroll, read, done, navigate

Responde SOLO con este JSON (sin otro texto):
```json
{{"action": "click", "selector": "[1]", "value": null, "reasoning": "haciendo clic en elemento 1 porque..."}}
```""",

    "de": """Aufgabe: {task}
Aktuelle Seite: {url} - {title}

Interaktive Elemente:
{elements}

Wähle EINE Aktion. Verfügbare Aktionen: click, fill, scroll, read, done, navigate

Antworte NUR mit diesem JSON (kein anderer Text):
```json
{{"action": "click", "selector": "[1]", "value": null, "reasoning": "Klicke Element 1 weil..."}}
```""",
}

# RSA-style aggregation prompt
AGGREGATION_PROMPT = """You are aggregating multiple reasoning attempts about a browser action.

Task: {task}
URL: {url}

Here are {n} candidate solutions with their reasoning:

{candidates}

Analyze ALL candidates. Even wrong answers may contain useful observations.
Extract the best action by considering:
1. Which reasoning chains identify correct elements?
2. What useful observations appear across multiple candidates?
3. Where do candidates agree vs disagree?

Respond with the BEST action as JSON:
{{
  "action": "click|fill|scroll|read|done|navigate",
  "selector": "[index] if needed",
  "value": "value if needed",
  "reasoning": "synthesized reasoning from candidates",
  "confidence": 0.0-1.0,
  "useful_observations": ["list", "of", "insights", "from", "all", "candidates"]
}}
"""


class RecursiveReasoner:
    """
    Recursive reasoning system inspired by RLM, RSA, LADDER.

    Features:
    - Population of candidates (RSA)
    - Multi-language queries for richer understanding
    - Aggregation that extracts good reasoning from bad answers
    - Iterative refinement (RLM diffusion pattern)
    """

    def __init__(
        self,
        models: list[str] = None,
        languages: list[str] = None,
        aggregator_model: str = "qwen3:14b",
    ):
        # Use available models - qwen3:14b is best, llama3.2 and llama3 as backup
        self.models = models or ["qwen3:14b", "llama3.2:latest", "llama3:latest"]
        self.languages = languages or ["en", "zh", "es"]  # Multi-language by default
        self.aggregator_model = aggregator_model
        self.client = ollama.Client()

    def reason(
        self,
        task: str,
        url: str,
        title: str,
        elements: str,
        history: list[str] = None,
    ) -> AggregatedDecision:
        """
        Generate multiple candidates and aggregate them (RSA-style).

        Uses multi-language prompts for richer human understanding.
        """
        candidates = []

        # Generate candidates across models and languages
        for model in self.models:
            for lang in self.languages:
                candidate = self._generate_candidate(
                    model=model,
                    language=lang,
                    task=task,
                    url=url,
                    title=title,
                    elements=elements,
                )
                if candidate:
                    candidates.append(candidate)
                    logger.info(f"Candidate from {model}/{lang}: {candidate.action_type}")

        if not candidates:
            return AggregatedDecision(
                action_type="scroll",
                selector=None,
                value="down",
                reasoning="No candidates generated",
                confidence=0.0,
                candidates_considered=0,
                consensus_level=0.0,
            )

        # Aggregate candidates (RSA-style)
        return self._aggregate_candidates(candidates, task, url)

    def _generate_candidate(
        self,
        model: str,
        language: str,
        task: str,
        url: str,
        title: str,
        elements: str,
    ) -> Optional[ReasoningCandidate]:
        """Generate a single candidate in a specific language."""
        prompt_template = PROMPTS_BY_LANGUAGE.get(language, PROMPTS_BY_LANGUAGE["en"])
        prompt = prompt_template.format(
            task=task,
            url=url,
            title=title,
            elements=elements[:4000],  # Truncate for smaller models
        )

        try:
            response = self.client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.4, "num_ctx": 8192},
            )

            content = response["message"]["content"]
            data = self._parse_json(content)

            return ReasoningCandidate(
                action_type=data.get("action", "scroll"),
                selector=data.get("selector"),
                value=data.get("value"),
                reasoning=data.get("reasoning", ""),
                language=language,
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception as e:
            logger.warning(f"Candidate generation failed ({model}/{language}): {e}")
            return None

    def _aggregate_candidates(
        self,
        candidates: list[ReasoningCandidate],
        task: str,
        url: str,
    ) -> AggregatedDecision:
        """
        RSA-style aggregation: synthesize best action from all candidates.

        Key insight: Even wrong answers contain useful observations.
        """
        # Format candidates for aggregation
        candidates_text = ""
        for i, c in enumerate(candidates, 1):
            candidates_text += f"""
Candidate {i} ({c.language}):
  Action: {c.action_type}
  Selector: {c.selector}
  Value: {c.value}
  Reasoning: {c.reasoning}
"""

        prompt = AGGREGATION_PROMPT.format(
            task=task,
            url=url,
            n=len(candidates),
            candidates=candidates_text,
        )

        try:
            response = self.client.chat(
                model=self.aggregator_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_ctx": 16384},
            )

            content = response["message"]["content"]
            data = self._parse_json(content)

            # Calculate consensus level
            action_counts = {}
            for c in candidates:
                action_counts[c.action_type] = action_counts.get(c.action_type, 0) + 1
            max_agreement = max(action_counts.values()) if action_counts else 0
            consensus_level = max_agreement / len(candidates) if candidates else 0

            return AggregatedDecision(
                action_type=data.get("action", "scroll"),
                selector=data.get("selector"),
                value=data.get("value"),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5)),
                candidates_considered=len(candidates),
                consensus_level=consensus_level,
            )
        except Exception as e:
            logger.warning(f"Aggregation failed: {e}")
            # Fallback: majority vote
            action_counts = {}
            for c in candidates:
                action_counts[c.action_type] = action_counts.get(c.action_type, 0) + 1
            winner = max(action_counts, key=action_counts.get)
            best = next(c for c in candidates if c.action_type == winner)

            return AggregatedDecision(
                action_type=winner,
                selector=best.selector,
                value=best.value,
                reasoning=f"Majority vote: {action_counts}",
                confidence=0.5,
                candidates_considered=len(candidates),
                consensus_level=action_counts[winner] / len(candidates),
            )

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response with robust extraction."""
        import re

        # Try code block extraction first
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]

        # Try to find JSON object pattern
        content = content.strip()
        match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

        # Clean common issues
        content = content.replace("'", '"')  # Single to double quotes

        return json.loads(content)


def decompose_task(task: str, model: str = "qwen3:14b") -> list[str]:
    """
    LADDER-style task decomposition.

    Break complex task into simpler sub-tasks.
    Solve easy ones first, build up to hard ones.
    """
    client = ollama.Client()

    prompt = f"""Break this browser task into simpler sub-tasks:

Task: {task}

Create a sequence of easier steps, from simplest to most complex.
Each step should be completable independently.

Respond as JSON:
{{
  "subtasks": [
    "simplest step first",
    "slightly harder step",
    "...",
    "final complex step"
  ]
}}
"""

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )

        content = response["message"]["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
        return data.get("subtasks", [task])
    except Exception as e:
        logger.warning(f"Task decomposition failed: {e}")
        return [task]  # Fallback to original task
