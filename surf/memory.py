"""
Trajectory memory for browser automation.

Stores full successful runs and retrieves them as few-shot examples.
No ML training required - just embedding search.

Schema:
- trajectories: Full run records with task, actions, outcome
- embeddings: Vector embeddings for similarity search
"""

import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

import numpy as np
import ollama

logger = logging.getLogger(__name__)


@dataclass
class ActionStep:
    """A single action in a trajectory."""
    step: int
    url: str
    action_type: str
    selector: Optional[str]
    value: Optional[str]
    success: bool
    read_result: Optional[str] = None


@dataclass
class Trajectory:
    """A complete run trajectory."""
    id: Optional[int]
    task: str
    start_url: str
    success: bool
    result: Optional[str]
    actions: List[ActionStep]
    created_at: str
    total_llm_calls: int = 0
    total_depth: int = 0

    def to_few_shot(self) -> str:
        """Format as few-shot example for prompt."""
        lines = [f"Task: {self.task}", f"URL: {self.start_url}", "Actions:"]
        for a in self.actions:
            if a.action_type == "done":
                lines.append(f"  {a.step}. done -> {a.value}")
            elif a.read_result:
                lines.append(f"  {a.step}. {a.action_type} [{a.selector}] -> read: \"{a.read_result[:50]}\"")
            else:
                lines.append(f"  {a.step}. {a.action_type} [{a.selector}] {a.value or ''}")
        lines.append(f"Result: {self.result}")
        return "\n".join(lines)


class TrajectoryMemory:
    """
    SQLite-backed trajectory memory with embedding search.

    Stores successful runs and retrieves similar ones as few-shot examples.
    """

    DEFAULT_PATH = "~/.rhea-surf/memory.db"

    def __init__(self, db_path: Optional[str] = None, embed_model: str = "nomic-embed-text:latest"):
        self.db_path = Path(os.path.expanduser(db_path or self.DEFAULT_PATH))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model
        self.client = ollama.Client()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    start_url TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    result TEXT,
                    actions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    total_llm_calls INTEGER DEFAULT 0,
                    total_depth INTEGER DEFAULT 0,
                    embedding BLOB
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traj_success ON trajectories(success)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traj_task ON trajectories(task)
            """)
            conn.commit()

    def _embed(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        response = self.client.embeddings(model=self.embed_model, prompt=text)
        return response["embedding"]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def store(self, trajectory: Trajectory) -> int:
        """
        Store a trajectory. Only stores successful runs.

        Returns trajectory ID.
        """
        if not trajectory.success:
            logger.info("Skipping failed trajectory")
            return -1

        # Create embedding from task + start_url + result
        embed_text = f"{trajectory.task} {trajectory.start_url} {trajectory.result or ''}"
        embedding = self._embed(embed_text)

        actions_json = json.dumps([asdict(a) for a in trajectory.actions])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO trajectories (
                    task, start_url, success, result, actions_json,
                    created_at, total_llm_calls, total_depth, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.task,
                trajectory.start_url,
                1 if trajectory.success else 0,
                trajectory.result,
                actions_json,
                trajectory.created_at,
                trajectory.total_llm_calls,
                trajectory.total_depth,
                np.array(embedding).tobytes()
            ))
            conn.commit()
            logger.info(f"Stored trajectory: {trajectory.task[:50]}...")
            return cursor.lastrowid

    def search(self, task: str, url: str = "", limit: int = 3) -> List[Trajectory]:
        """
        Find similar successful trajectories.

        Returns most similar trajectories sorted by similarity.
        """
        query_text = f"{task} {url}"
        query_embedding = self._embed(query_text)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM trajectories WHERE success = 1
            """)
            rows = cursor.fetchall()

        if not rows:
            return []

        # Calculate similarities
        results = []
        for row in rows:
            embedding = np.frombuffer(row['embedding'], dtype=np.float64)
            similarity = self._cosine_similarity(query_embedding, embedding.tolist())

            actions = [ActionStep(**a) for a in json.loads(row['actions_json'])]
            trajectory = Trajectory(
                id=row['id'],
                task=row['task'],
                start_url=row['start_url'],
                success=bool(row['success']),
                result=row['result'],
                actions=actions,
                created_at=row['created_at'],
                total_llm_calls=row['total_llm_calls'],
                total_depth=row['total_depth'],
            )
            results.append((trajectory, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return [t for t, _ in results[:limit]]

    def format_few_shot(self, trajectories: List[Trajectory]) -> str:
        """Format trajectories as few-shot examples for prompt."""
        if not trajectories:
            return ""

        lines = ["Here are similar successful tasks for reference:"]
        for i, t in enumerate(trajectories, 1):
            lines.append(f"\n--- Example {i} ---")
            lines.append(t.to_few_shot())

        lines.append("\n--- Your turn ---")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(total_llm_calls) as avg_llm_calls,
                    AVG(total_depth) as avg_depth
                FROM trajectories
            """)
            row = cursor.fetchone()
            return {
                'total_trajectories': row[0] or 0,
                'successful': row[1] or 0,
                'avg_llm_calls': round(row[2] or 0, 1),
                'avg_depth': round(row[3] or 0, 1),
            }

    def clear(self):
        """Clear all trajectories."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM trajectories")
            conn.commit()


# Convenience function
def get_memory(path: Optional[str] = None) -> TrajectoryMemory:
    """Get or create trajectory memory instance."""
    return TrajectoryMemory(path)


if __name__ == "__main__":
    # Test the memory
    memory = TrajectoryMemory()

    # Create a test trajectory
    traj = Trajectory(
        id=None,
        task="Find the top story on Hacker News",
        start_url="https://news.ycombinator.com",
        success=True,
        result="The top story is: Example Title",
        actions=[
            ActionStep(1, "https://news.ycombinator.com", "read", "[1]", None, True, "Example Title"),
            ActionStep(2, "https://news.ycombinator.com", "done", None, "The top story is: Example Title", True),
        ],
        created_at=datetime.now().isoformat(),
        total_llm_calls=2,
        total_depth=1,
    )

    # Store
    tid = memory.store(traj)
    print(f"Stored trajectory ID: {tid}")

    # Search
    similar = memory.search("What is the top story on HN?", "https://news.ycombinator.com")
    print(f"\nFound {len(similar)} similar trajectories")
    if similar:
        print("\nFew-shot example:")
        print(memory.format_few_shot(similar))

    # Stats
    print(f"\nStats: {memory.get_stats()}")
