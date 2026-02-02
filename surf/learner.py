"""
Learning system for rhea-surf.

Analyzes run logs to extract patterns:
- What worked on specific sites
- What failed and why
- General patterns that transfer

Features:
- Embedding-based memory with recency decay
- Vector search to find relevant past experiences
- Uses Ollama for both embeddings and analysis
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import numpy as np
import ollama

logger = logging.getLogger(__name__)

PATTERNS_DIR = Path(__file__).parent.parent / "learned"
PATTERNS_DIR.mkdir(exist_ok=True)

# Recency decay parameters
HALF_LIFE_DAYS = 7  # Memory importance halves every 7 days


@dataclass
class LearnedPattern:
    """A pattern learned from experience."""
    pattern_id: str
    domain: Optional[str]  # None = general pattern
    pattern_type: str  # "success", "failure", "shortcut"
    trigger: str  # When to apply this pattern
    action: str  # What to do
    confidence: float  # 0-1
    source_runs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    times_used: int = 0
    times_succeeded: int = 0
    embedding: Optional[list[float]] = None  # Vector embedding for search

    def recency_weight(self) -> float:
        """Calculate recency weight with exponential decay."""
        created = datetime.fromisoformat(self.created_at)
        age_days = (datetime.now() - created).days
        return math.exp(-math.log(2) * age_days / HALF_LIFE_DAYS)

    def effective_score(self, similarity: float = 1.0) -> float:
        """Combined score: similarity * confidence * recency."""
        return similarity * self.confidence * self.recency_weight()


ANALYSIS_PROMPT = """Analyze this browser automation run log and extract learnings.

Run Log:
{log_json}

Extract patterns in this JSON format:
{{
  "patterns": [
    {{
      "domain": "example.com or null for general",
      "pattern_type": "success|failure|shortcut",
      "trigger": "when to apply this pattern",
      "action": "what action worked or should be avoided",
      "confidence": 0.0-1.0
    }}
  ],
  "summary": "one sentence summary"
}}

Focus on:
1. What actions led to success?
2. What actions failed and why?
3. Are there shortcuts (dumb patterns) that could skip LLM calls?
4. Site-specific quirks vs general patterns

Respond with ONLY the JSON, no other text.
"""


class Learner:
    """Learn from run logs to improve future performance."""

    def __init__(self, model: str = "llama3:latest", embed_model: str = "nomic-embed-text:latest"):
        """Use a small/fast model for analysis, embedding model for search."""
        self.model = model
        self.embed_model = embed_model
        self.client = ollama.Client()
        self._patterns: dict[str, list[LearnedPattern]] = {}  # domain -> patterns
        self._all_patterns: list[LearnedPattern] = []  # flat list for search
        self._load_patterns()

    def _embed(self, text: str) -> list[float]:
        """Get embedding for text using Ollama."""
        response = self.client.embeddings(model=self.embed_model, prompt=text)
        return response["embedding"]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _load_patterns(self):
        """Load existing patterns from disk."""
        patterns_file = PATTERNS_DIR / "patterns.json"
        if patterns_file.exists():
            with open(patterns_file) as f:
                data = json.load(f)
                for p in data.get("patterns", []):
                    pattern = LearnedPattern(**p)
                    domain = pattern.domain or "_general"
                    if domain not in self._patterns:
                        self._patterns[domain] = []
                    self._patterns[domain].append(pattern)
                    self._all_patterns.append(pattern)
            logger.info(f"Loaded {len(self._all_patterns)} patterns")

    def search_patterns(self, query: str, limit: int = 5) -> list[tuple[LearnedPattern, float]]:
        """
        Search patterns by semantic similarity with recency weighting.

        Returns: List of (pattern, effective_score) tuples
        """
        if not self._all_patterns:
            return []

        query_embedding = self._embed(query)
        results = []

        for pattern in self._all_patterns:
            # Embed pattern if not already done
            if pattern.embedding is None:
                pattern_text = f"{pattern.trigger} {pattern.action}"
                pattern.embedding = self._embed(pattern_text)

            similarity = self._cosine_similarity(query_embedding, pattern.embedding)
            score = pattern.effective_score(similarity)
            results.append((pattern, score))

        # Sort by effective score (similarity * confidence * recency)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _save_patterns(self):
        """Save patterns to disk."""
        all_patterns = []
        for patterns in self._patterns.values():
            all_patterns.extend([asdict(p) for p in patterns])

        patterns_file = PATTERNS_DIR / "patterns.json"
        with open(patterns_file, "w") as f:
            json.dump({"patterns": all_patterns}, f, indent=2)
        logger.info(f"Saved {len(all_patterns)} patterns")

    def analyze_run(self, log_path: Path) -> list[LearnedPattern]:
        """
        Analyze a run log and extract patterns.

        Args:
            log_path: Path to run_*.json log file

        Returns:
            List of learned patterns
        """
        logger.info(f"Analyzing run: {log_path}")

        with open(log_path) as f:
            log_data = json.load(f)

        # Get domain from URLs in log
        domains = set()
        for step in log_data.get("steps", []):
            url = step.get("url", "")
            if url:
                parsed = urlparse(url)
                if parsed.netloc:
                    domains.add(parsed.netloc)

        # Prepare log summary for LLM (don't send full HTML)
        summary = {
            "task": log_data.get("task"),
            "success": log_data.get("success"),
            "domains": list(domains),
            "steps": [
                {
                    "step": s.get("step_num"),
                    "url": s.get("url"),
                    "action": s.get("action"),
                    "success": s.get("success"),
                    "error": s.get("error"),
                    "read_result": s.get("read_result"),
                }
                for s in log_data.get("steps", [])
            ],
            "summary": log_data.get("summary"),
        }

        # Ask LLM to analyze
        prompt = ANALYSIS_PROMPT.format(log_json=json.dumps(summary, indent=2))

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )

        content = response["message"]["content"]

        # Parse response
        patterns = self._parse_patterns(content, log_path.name)

        # Store patterns
        for pattern in patterns:
            domain = pattern.domain or "_general"
            if domain not in self._patterns:
                self._patterns[domain] = []
            self._patterns[domain].append(pattern)

        self._save_patterns()
        return patterns

    def _parse_patterns(self, content: str, run_id: str) -> list[LearnedPattern]:
        """Parse LLM response into patterns."""
        patterns = []

        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            for p in data.get("patterns", []):
                pattern = LearnedPattern(
                    pattern_id=f"p_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(patterns)}",
                    domain=p.get("domain"),
                    pattern_type=p.get("pattern_type", "success"),
                    trigger=p.get("trigger", ""),
                    action=p.get("action", ""),
                    confidence=float(p.get("confidence", 0.5)),
                    source_runs=[run_id],
                )
                patterns.append(pattern)

            logger.info(f"Extracted {len(patterns)} patterns")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse patterns: {e}")

        return patterns

    def get_patterns_for_domain(self, url: str) -> list[LearnedPattern]:
        """Get relevant patterns for a URL."""
        parsed = urlparse(url)
        domain = parsed.netloc

        patterns = []

        # Get domain-specific patterns
        if domain in self._patterns:
            patterns.extend(self._patterns[domain])

        # Get general patterns
        if "_general" in self._patterns:
            patterns.extend(self._patterns["_general"])

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def format_patterns_for_prompt(self, patterns: list[LearnedPattern]) -> str:
        """Format patterns as context for LLM prompt."""
        if not patterns:
            return ""

        lines = ["LEARNED PATTERNS (from previous runs):"]
        for p in patterns[:5]:  # Top 5
            lines.append(f"- [{p.pattern_type}] {p.trigger} -> {p.action} (confidence: {p.confidence:.1f})")

        return "\n".join(lines)

    def record_outcome(self, pattern_id: str, success: bool):
        """Record whether a pattern worked when applied."""
        for patterns in self._patterns.values():
            for p in patterns:
                if p.pattern_id == pattern_id:
                    p.times_used += 1
                    if success:
                        p.times_succeeded += 1
                    # Update confidence based on outcomes
                    if p.times_used > 0:
                        p.confidence = p.times_succeeded / p.times_used
                    self._save_patterns()
                    return

    def store_pattern(
        self,
        pattern_id: str,
        domain: Optional[str],
        pattern_type: str,
        trigger: str,
        action: str,
        confidence: float,
        source_runs: list[str] = None,
    ):
        """
        Store a new learned pattern directly.

        Used by cache promotion to convert high-hit cached actions to patterns.
        """
        # Check for duplicate
        for patterns in self._patterns.values():
            for p in patterns:
                if p.trigger == trigger and p.action == action:
                    # Update existing
                    p.confidence = max(p.confidence, confidence)
                    p.times_used += 1
                    self._save_patterns()
                    return

        # Create new pattern
        pattern = LearnedPattern(
            pattern_id=pattern_id,
            domain=domain,
            pattern_type=pattern_type,
            trigger=trigger,
            action=action,
            confidence=confidence,
            source_runs=source_runs or [],
        )

        # Store
        domain_key = domain or "_general"
        if domain_key not in self._patterns:
            self._patterns[domain_key] = []
        self._patterns[domain_key].append(pattern)
        self._all_patterns.append(pattern)

        self._save_patterns()
        logger.info(f"Stored pattern: {pattern_id} -> {action}")


def analyze_all_logs():
    """Analyze all run logs in the logs directory."""
    logs_dir = Path(__file__).parent.parent / "logs"
    learner = Learner()

    for log_file in sorted(logs_dir.glob("run_*.json")):
        print(f"\nAnalyzing: {log_file.name}")
        patterns = learner.analyze_run(log_file)
        for p in patterns:
            print(f"  - [{p.pattern_type}] {p.trigger[:50]}...")


if __name__ == "__main__":
    analyze_all_logs()
