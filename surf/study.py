"""
Meta-study framework for rhea-surf self-improvement.

Tracks performance over time, computes loss functions, and identifies
areas for improvement. Run regularly to monitor if the system is learning.

Key metrics (loss functions):
1. Success Rate: % of tasks completed correctly
2. Efficiency: LLM calls per successful task (lower = better)
3. Speed: Time per task (lower = better)
4. Cache Utilization: % of decisions from cache (higher = better)
5. Learning Rate: Improvement over time

Usage:
    python -m surf.study              # Run study session
    python -m surf.study --history    # Show improvement trends
    python -m surf.study --analyze    # Deep analysis
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import asyncio

logger = logging.getLogger(__name__)

# Default study tasks (varying complexity)
# Task wording should be consistent to enable cache hits on repeated runs
STUDY_TASKS = [
    {
        "name": "hn_top_story",
        "task": "what is the title of the top story?",
        "url": "https://news.ycombinator.com",
        "complexity": "simple",
        "timeout": 30,
    },
    {
        "name": "example_heading",
        "task": "what is the main heading on this page?",
        "url": "https://example.com",
        "complexity": "simple",
        "timeout": 30,
    },
    {
        "name": "github_repo",
        "task": "what is the name of this repository?",
        "url": "https://github.com/anthropics/anthropic-sdk-python",
        "complexity": "simple",
        "timeout": 45,
    },
]


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_name: str
    success: bool
    result: Optional[str]
    llm_calls: int
    cache_hits: int
    elapsed_seconds: float
    error: Optional[str] = None


@dataclass
class StudySession:
    """A complete study session with multiple tasks."""
    session_id: str
    timestamp: str
    tasks_run: int
    tasks_passed: int
    total_llm_calls: int
    total_cache_hits: int
    total_time: float
    results: List[TaskResult]

    # Computed metrics
    success_rate: float = 0.0
    avg_llm_calls: float = 0.0
    cache_utilization: float = 0.0
    avg_time: float = 0.0

    def compute_metrics(self):
        """Compute derived metrics after results are collected."""
        if self.tasks_run > 0:
            self.success_rate = self.tasks_passed / self.tasks_run
            self.avg_time = self.total_time / self.tasks_run

        if self.tasks_passed > 0:
            self.avg_llm_calls = self.total_llm_calls / self.tasks_passed

        total_decisions = self.total_llm_calls + self.total_cache_hits
        if total_decisions > 0:
            self.cache_utilization = self.total_cache_hits / total_decisions


@dataclass
class LossMetrics:
    """Loss functions for measuring system performance."""

    # Primary losses (lower is better)
    task_failure_loss: float  # 1 - success_rate
    efficiency_loss: float    # normalized LLM calls (more calls = higher loss)
    speed_loss: float         # normalized time (slower = higher loss)

    # Learning indicators
    cache_miss_loss: float    # 1 - cache_utilization

    # Combined loss
    total_loss: float

    @classmethod
    def from_session(cls, session: StudySession) -> 'LossMetrics':
        """Compute loss metrics from a study session."""
        # Task failure loss: direct measure of success
        task_failure = 1.0 - session.success_rate

        # Efficiency loss: normalize LLM calls (target: 2 calls per task)
        target_llm_calls = 2.0
        efficiency = min(1.0, max(0.0, (session.avg_llm_calls - target_llm_calls) / 10.0))

        # Speed loss: normalize time (target: 5 seconds per task)
        target_time = 5.0
        speed = min(1.0, max(0.0, (session.avg_time - target_time) / 30.0))

        # Cache miss loss
        cache_miss = 1.0 - session.cache_utilization

        # Combined loss (weighted)
        # Success is most important, then efficiency, then speed
        total = (
            0.50 * task_failure +
            0.25 * efficiency +
            0.15 * speed +
            0.10 * cache_miss
        )

        return cls(
            task_failure_loss=task_failure,
            efficiency_loss=efficiency,
            speed_loss=speed,
            cache_miss_loss=cache_miss,
            total_loss=total,
        )


class StudyDB:
    """SQLite database for storing study results."""

    DEFAULT_PATH = "~/.rhea-surf/study.db"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or self.DEFAULT_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    tasks_run INTEGER,
                    tasks_passed INTEGER,
                    total_llm_calls INTEGER,
                    total_cache_hits INTEGER,
                    total_time REAL,
                    success_rate REAL,
                    avg_llm_calls REAL,
                    cache_utilization REAL,
                    avg_time REAL,
                    -- Loss metrics
                    task_failure_loss REAL,
                    efficiency_loss REAL,
                    speed_loss REAL,
                    cache_miss_loss REAL,
                    total_loss REAL,
                    -- Raw results
                    results_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_timestamp
                ON study_sessions(timestamp)
            """)
            conn.commit()

    def store_session(self, session: StudySession, losses: LossMetrics):
        """Store a study session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO study_sessions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                session.session_id,
                session.timestamp,
                session.tasks_run,
                session.tasks_passed,
                session.total_llm_calls,
                session.total_cache_hits,
                session.total_time,
                session.success_rate,
                session.avg_llm_calls,
                session.cache_utilization,
                session.avg_time,
                losses.task_failure_loss,
                losses.efficiency_loss,
                losses.speed_loss,
                losses.cache_miss_loss,
                losses.total_loss,
                json.dumps([asdict(r) for r in session.results]),
            ))
            conn.commit()

    def get_recent_sessions(self, limit: int = 10) -> List[dict]:
        """Get recent study sessions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM study_sessions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_trend(self, metric: str, days: int = 7) -> List[tuple]:
        """Get trend for a metric over time."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT timestamp, {metric}
                FROM study_sessions
                WHERE timestamp > ?
                ORDER BY timestamp
            """, (cutoff,))
            return cursor.fetchall()

    def compute_improvement(self, metric: str = "total_loss") -> Optional[float]:
        """
        Compute improvement rate for a metric.

        Returns percentage improvement (positive = getting better).
        """
        sessions = self.get_recent_sessions(20)
        if len(sessions) < 2:
            return None

        # Compare first half to second half
        mid = len(sessions) // 2
        recent = sessions[:mid]
        older = sessions[mid:]

        recent_avg = sum(s[metric] for s in recent) / len(recent)
        older_avg = sum(s[metric] for s in older) / len(older)

        if older_avg == 0:
            return None

        # For loss metrics, lower is better, so improvement is negative change
        improvement = (older_avg - recent_avg) / older_avg * 100
        return improvement


class StudyRunner:
    """Runs study sessions and tracks improvement."""

    def __init__(self, tasks: List[dict] = None):
        self.tasks = tasks or STUDY_TASKS
        self.db = StudyDB()

    async def run_session(self, verbose: bool = True) -> StudySession:
        """Run a complete study session."""
        from surf.navigator import RecursiveNavigator

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        total_llm = 0
        total_cache = 0
        total_time = 0.0
        passed = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"STUDY SESSION: {session_id}")
            print(f"{'='*60}\n")

        for task_def in self.tasks:
            if verbose:
                print(f"Task: {task_def['name']} ({task_def['complexity']})")

            start = time.time()

            try:
                nav = RecursiveNavigator(
                    model="llama3.2:latest",
                    max_depth=5,
                    use_cache=True,
                    use_recursive=False,
                )
                nav.use_debate = False

                result = await asyncio.wait_for(
                    nav.navigate(task_def["task"], task_def["url"]),
                    timeout=task_def["timeout"]
                )

                elapsed = time.time() - start

                task_result = TaskResult(
                    task_name=task_def["name"],
                    success=result.success,
                    result=result.result,
                    llm_calls=result.llm_calls,
                    cache_hits=result.cache_hits,
                    elapsed_seconds=elapsed,
                )

                if result.success:
                    passed += 1

                total_llm += result.llm_calls
                total_cache += result.cache_hits

            except asyncio.TimeoutError:
                elapsed = task_def["timeout"]
                task_result = TaskResult(
                    task_name=task_def["name"],
                    success=False,
                    result=None,
                    llm_calls=0,
                    cache_hits=0,
                    elapsed_seconds=elapsed,
                    error="Timeout",
                )
            except Exception as e:
                elapsed = time.time() - start
                task_result = TaskResult(
                    task_name=task_def["name"],
                    success=False,
                    result=None,
                    llm_calls=0,
                    cache_hits=0,
                    elapsed_seconds=elapsed,
                    error=str(e),
                )

            results.append(task_result)
            total_time += elapsed

            if verbose:
                status = "PASS" if task_result.success else "FAIL"
                print(f"  [{status}] {elapsed:.1f}s, {task_result.llm_calls} LLM, {task_result.cache_hits} cache")
                if task_result.result:
                    print(f"  Result: {task_result.result[:60]}...")
                if task_result.error:
                    print(f"  Error: {task_result.error}")
                print()

        # Build session
        session = StudySession(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            tasks_run=len(self.tasks),
            tasks_passed=passed,
            total_llm_calls=total_llm,
            total_cache_hits=total_cache,
            total_time=total_time,
            results=results,
        )
        session.compute_metrics()

        # Compute losses
        losses = LossMetrics.from_session(session)

        # Store
        self.db.store_session(session, losses)

        if verbose:
            self._print_summary(session, losses)

        return session

    def _print_summary(self, session: StudySession, losses: LossMetrics):
        """Print session summary."""
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Success Rate: {session.success_rate:.0%} ({session.tasks_passed}/{session.tasks_run})")
        print(f"Avg LLM Calls: {session.avg_llm_calls:.1f}")
        print(f"Cache Utilization: {session.cache_utilization:.0%}")
        print(f"Avg Time: {session.avg_time:.1f}s")

        print(f"\n{'='*60}")
        print("LOSS METRICS (lower is better)")
        print(f"{'='*60}")
        print(f"Task Failure Loss: {losses.task_failure_loss:.3f}")
        print(f"Efficiency Loss:   {losses.efficiency_loss:.3f}")
        print(f"Speed Loss:        {losses.speed_loss:.3f}")
        print(f"Cache Miss Loss:   {losses.cache_miss_loss:.3f}")
        print(f"{'─'*30}")
        print(f"TOTAL LOSS:        {losses.total_loss:.3f}")

        # Show improvement trend
        improvement = self.db.compute_improvement("total_loss")
        if improvement is not None:
            direction = "improving" if improvement > 0 else "degrading"
            print(f"\nTrend: {direction} ({improvement:+.1f}% vs previous sessions)")

    def show_history(self, limit: int = 10):
        """Show study history."""
        sessions = self.db.get_recent_sessions(limit)

        if not sessions:
            print("No study sessions recorded yet.")
            return

        print(f"\n{'='*80}")
        print("STUDY HISTORY (most recent first)")
        print(f"{'='*80}")
        print(f"{'Timestamp':<20} {'Pass':<8} {'LLM':<8} {'Cache%':<8} {'Loss':<8} {'Trend':<10}")
        print(f"{'-'*80}")

        prev_loss = None
        for s in sessions:
            trend = ""
            if prev_loss is not None:
                diff = prev_loss - s['total_loss']
                if diff > 0.01:
                    trend = "↑ better"
                elif diff < -0.01:
                    trend = "↓ worse"
                else:
                    trend = "→ same"

            print(f"{s['timestamp'][:19]:<20} "
                  f"{s['tasks_passed']}/{s['tasks_run']:<6} "
                  f"{s['avg_llm_calls']:.1f}{'':>5} "
                  f"{s['cache_utilization']*100:.0f}%{'':>5} "
                  f"{s['total_loss']:.3f}{'':>4} "
                  f"{trend}")

            prev_loss = s['total_loss']

        # Overall trend
        print(f"\n{'='*80}")
        improvement = self.db.compute_improvement("total_loss")
        if improvement is not None:
            if improvement > 5:
                print(f"Overall: IMPROVING (+{improvement:.1f}% better)")
            elif improvement < -5:
                print(f"Overall: DEGRADING ({improvement:.1f}% worse)")
            else:
                print(f"Overall: STABLE ({improvement:+.1f}%)")

    def analyze(self):
        """Deep analysis of performance trends."""
        sessions = self.db.get_recent_sessions(50)

        if len(sessions) < 3:
            print("Need at least 3 study sessions for analysis.")
            return

        print(f"\n{'='*60}")
        print("PERFORMANCE ANALYSIS")
        print(f"{'='*60}")

        # Success rate trend
        success_rates = [s['success_rate'] for s in sessions]
        print(f"\nSuccess Rate:")
        print(f"  Latest: {success_rates[0]:.0%}")
        print(f"  Average: {sum(success_rates)/len(success_rates):.0%}")
        print(f"  Best: {max(success_rates):.0%}")
        print(f"  Worst: {min(success_rates):.0%}")

        # Efficiency trend
        llm_calls = [s['avg_llm_calls'] for s in sessions if s['avg_llm_calls'] > 0]
        if llm_calls:
            print(f"\nEfficiency (LLM calls per task):")
            print(f"  Latest: {llm_calls[0]:.1f}")
            print(f"  Average: {sum(llm_calls)/len(llm_calls):.1f}")
            print(f"  Best: {min(llm_calls):.1f}")
            print(f"  Worst: {max(llm_calls):.1f}")

        # Cache utilization
        cache_utils = [s['cache_utilization'] for s in sessions]
        print(f"\nCache Utilization:")
        print(f"  Latest: {cache_utils[0]:.0%}")
        print(f"  Average: {sum(cache_utils)/len(cache_utils):.0%}")
        print(f"  Best: {max(cache_utils):.0%}")

        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")

        if success_rates[0] < 0.7:
            print("- Success rate is low. Consider:")
            print("  * Improving LLM prompts")
            print("  * Adding more patterns")
            print("  * Using debate for uncertain decisions")

        if llm_calls and llm_calls[0] > 5:
            print("- High LLM call count. Consider:")
            print("  * Improving cache hit rate")
            print("  * Better loop detection")
            print("  * Clearer prompts to avoid retries")

        if cache_utils[0] < 0.3:
            print("- Low cache utilization. Consider:")
            print("  * Running more similar tasks to build cache")
            print("  * Promoting high-hit cache entries to patterns")


async def main():
    """Main entry point for study framework."""
    import sys

    runner = StudyRunner()

    if "--history" in sys.argv:
        runner.show_history()
    elif "--analyze" in sys.argv:
        runner.analyze()
    else:
        await runner.run_session(verbose=True)


if __name__ == "__main__":
    asyncio.run(main())
