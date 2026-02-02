"""
Multi-agent debate for uncertain decisions.

When the primary model isn't confident, poll multiple cheap models
and use majority voting to decide the action.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional
from collections import Counter
import ollama

logger = logging.getLogger(__name__)


@dataclass
class Vote:
    """A single model's vote."""
    model: str
    action_type: str
    selector: Optional[str]
    value: Optional[str]
    confidence: float
    reasoning: str


@dataclass
class DebateResult:
    """Result of multi-agent debate."""
    consensus: bool
    winning_action: str
    winning_selector: Optional[str]
    winning_value: Optional[str]
    vote_count: dict[str, int]
    votes: list[Vote]
    confidence: float


DEBATE_PROMPT = """You are evaluating a browser automation decision. Given the current state, what action should be taken?

Task: {task}
URL: {url}
Title: {title}

Interactive elements:
{elements}

{history}

Respond with ONLY a JSON object:
{{
  "action": "click|fill|scroll|read|done|navigate",
  "selector": "[index] if needed",
  "value": "value if needed",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""


class Debate:
    """
    Multi-agent debate system.

    Uses multiple cheap/fast models to vote on uncertain decisions.
    Majority wins. Disagreement triggers more cautious action.
    """

    # Models to use for debate (fast/cheap ones)
    DEFAULT_MODELS = [
        "llama3.1:8b",
        "qwen3:14b",
        "gemma3n:e4b",
    ]

    def __init__(self, models: list[str] = None):
        self.models = models or self.DEFAULT_MODELS
        self.client = ollama.Client()

    def debate(
        self,
        task: str,
        url: str,
        title: str,
        elements: str,
        history: list[str] = None,
    ) -> DebateResult:
        """
        Run multi-agent debate on a decision.

        Returns DebateResult with consensus status and winning action.
        """
        votes = []

        history_text = ""
        if history:
            history_text = "Recent actions:\n" + "\n".join(history[-3:])

        prompt = DEBATE_PROMPT.format(
            task=task,
            url=url,
            title=title,
            elements=elements,
            history=history_text,
        )

        # Collect votes from each model
        for model in self.models:
            try:
                vote = self._get_vote(model, prompt)
                if vote:
                    votes.append(vote)
                    logger.info(f"Vote from {model}: {vote.action_type} (conf={vote.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Model {model} failed to vote: {e}")

        if not votes:
            # No votes - return cautious default
            return DebateResult(
                consensus=False,
                winning_action="scroll",
                winning_selector=None,
                winning_value="down",
                vote_count={},
                votes=[],
                confidence=0.0,
            )

        # Count votes by action type
        action_votes = Counter(v.action_type for v in votes)
        most_common = action_votes.most_common()

        # Check for consensus (majority)
        total = len(votes)
        winner, winner_count = most_common[0]
        consensus = winner_count > total / 2

        # Get details from winning votes
        winning_votes = [v for v in votes if v.action_type == winner]

        # Use the highest confidence vote's details
        best_vote = max(winning_votes, key=lambda v: v.confidence)

        # Calculate aggregate confidence
        avg_confidence = sum(v.confidence for v in winning_votes) / len(winning_votes)
        consensus_bonus = 0.2 if consensus else 0.0
        final_confidence = min(1.0, avg_confidence + consensus_bonus)

        return DebateResult(
            consensus=consensus,
            winning_action=winner,
            winning_selector=best_vote.selector,
            winning_value=best_vote.value,
            vote_count=dict(action_votes),
            votes=votes,
            confidence=final_confidence,
        )

    def _get_vote(self, model: str, prompt: str) -> Optional[Vote]:
        """Get a single model's vote."""
        response = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.3,
                "num_ctx": 8192,
            },
        )

        content = response["message"]["content"]

        # Parse JSON response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            return Vote(
                model=model,
                action_type=data.get("action", "scroll"),
                selector=data.get("selector"),
                value=data.get("value"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse vote from {model}: {e}")
            return None


# Singleton for easy access
_debate = None


def get_debate_system(models: list[str] = None) -> Debate:
    """Get or create debate system singleton."""
    global _debate
    if _debate is None or models:
        _debate = Debate(models)
    return _debate


def run_debate(
    task: str,
    url: str,
    title: str,
    elements: str,
    history: list[str] = None,
) -> DebateResult:
    """Run a debate and return result."""
    debate = get_debate_system()
    return debate.debate(task, url, title, elements, history)
