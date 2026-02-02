"""
Action caching system for deterministic browser automation.

Principle: First run uses LLM. Subsequent runs replay cached actions.
This provides 100% determinism for repeated tasks on the same site structure.
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse


@dataclass
class CachedAction:
    """A cached browser action."""
    action_type: str
    selector: str
    value: Optional[str] = None
    confidence: float = 0.0
    hit_count: int = 0
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedAction':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ActionCache:
    """
    SQLite-backed action cache for deterministic replay.

    Cache key = hash(task + domain + dom_hash)
    - task: Normalized task string
    - domain: URL domain (site-level caching)
    - dom_hash: Hash of interactive elements

    Features:
    - 7-day expiry for cache entries
    - Hit count tracking
    - Confidence-based retrieval
    """

    DEFAULT_PATH = "~/.rhea-surf/cache.db"
    CACHE_TTL_DAYS = 7

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(os.path.expanduser(db_path or self.DEFAULT_PATH))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS action_cache (
                    cache_key TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    url TEXT NOT NULL,
                    dom_hash TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    selector TEXT NOT NULL,
                    value TEXT,
                    confidence REAL DEFAULT 0.0,
                    hit_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_domain
                ON action_cache(url)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_task
                ON action_cache(task)
            """)
            conn.commit()

    def cache_key(self, task: str, url: str, dom_hash: str) -> str:
        """
        Generate stable cache key.

        Uses domain (not full URL) for site-level matching.
        """
        domain = urlparse(url).netloc
        normalized = f"{task.lower().strip()}|{domain}|{dom_hash}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, task: str, url: str, dom_hash: str) -> Optional[CachedAction]:
        """
        Retrieve cached action if exists and not expired.

        Increments hit_count on retrieval.
        """
        key = self.cache_key(task, url, dom_hash)
        now = datetime.utcnow()
        cutoff = (now - timedelta(days=self.CACHE_TTL_DAYS)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM action_cache
                WHERE cache_key = ? AND created_at > ?
            """, (key, cutoff))

            row = cursor.fetchone()
            if not row:
                return None

            # Update hit count and last used
            conn.execute("""
                UPDATE action_cache
                SET hit_count = hit_count + 1, last_used_at = ?
                WHERE cache_key = ?
            """, (now.isoformat(), key))
            conn.commit()

            return CachedAction(
                action_type=row['action_type'],
                selector=row['selector'],
                value=row['value'],
                confidence=row['confidence'],
                hit_count=row['hit_count'] + 1,
                created_at=row['created_at'],
                last_used_at=now.isoformat()
            )

    def store(
        self,
        task: str,
        url: str,
        dom_hash: str,
        action_type: str,
        selector: str,
        value: Optional[str] = None,
        confidence: float = 0.8,
        success: bool = True
    ):
        """
        Store action result. Only cache successes by default.

        Updates existing entries instead of replacing to track stats.
        """
        key = self.cache_key(task, url, dom_hash)
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Check if exists
            cursor = conn.execute(
                "SELECT cache_key, success_count, failure_count FROM action_cache WHERE cache_key = ?",
                (key,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update stats
                if success:
                    conn.execute("""
                        UPDATE action_cache
                        SET success_count = success_count + 1,
                            confidence = MIN(0.99, confidence + 0.02),
                            last_used_at = ?
                        WHERE cache_key = ?
                    """, (now, key))
                else:
                    conn.execute("""
                        UPDATE action_cache
                        SET failure_count = failure_count + 1,
                            confidence = MAX(0.0, confidence - 0.1),
                            last_used_at = ?
                        WHERE cache_key = ?
                    """, (now, key))
            else:
                # Insert new entry only on success
                if success:
                    conn.execute("""
                        INSERT INTO action_cache (
                            cache_key, task, url, dom_hash,
                            action_type, selector, value, confidence,
                            hit_count, created_at, last_used_at,
                            success_count, failure_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 1, 0)
                    """, (key, task.lower().strip(), url, dom_hash,
                          action_type, selector, value, confidence, now, now))

            conn.commit()

    def record_failure(self, task: str, url: str, dom_hash: str):
        """Record a failed action attempt (even if not cached)."""
        self.store(task, url, dom_hash, "", "", success=False)

    def get_high_hit_actions(self, min_hits: int = 5) -> List[Dict[str, Any]]:
        """
        Get actions with high hit counts for pattern extraction.

        These are candidates for promotion to learned patterns.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT task, url, action_type, selector, value, hit_count, confidence
                FROM action_cache
                WHERE hit_count >= ? AND confidence > 0.8
                ORDER BY hit_count DESC
                LIMIT 100
            """, (min_hits,))

            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(hit_count) as total_hits,
                    AVG(confidence) as avg_confidence,
                    SUM(success_count) as total_successes,
                    SUM(failure_count) as total_failures
                FROM action_cache
            """)
            row = cursor.fetchone()

            return {
                'total_entries': row[0] or 0,
                'total_hits': row[1] or 0,
                'avg_confidence': round(row[2] or 0, 3),
                'total_successes': row[3] or 0,
                'total_failures': row[4] or 0,
                'hit_rate': round((row[1] or 0) / max(1, (row[3] or 0) + (row[4] or 0)), 3)
            }

    def cleanup_expired(self) -> int:
        """Remove expired cache entries. Returns count of deleted entries."""
        cutoff = (datetime.utcnow() - timedelta(days=self.CACHE_TTL_DAYS)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM action_cache WHERE created_at < ?",
                (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    def clear(self):
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM action_cache")
            conn.commit()


def hash_dom_structure(elements: List[Any]) -> str:
    """
    Hash interactive elements for cache matching.

    Only hashes the structural signature, not content,
    so minor text changes don't invalidate cache.
    """
    # Extract just the interactive elements
    interactive = [e for e in elements if getattr(e, 'is_interactive', True)]

    # Create stable representation (tag + first 50 chars of text + sorted attrs)
    structure = []
    for e in interactive:
        tag = getattr(e, 'tag', '')
        text = getattr(e, 'text', '')[:50]
        attrs = getattr(e, 'attributes', {})
        structure.append((tag, text, tuple(sorted(attrs.items()))))

    return hashlib.md5(json.dumps(structure, sort_keys=True).encode()).hexdigest()[:12]


# Convenience function
def get_cache(path: Optional[str] = None) -> ActionCache:
    """Get or create action cache instance."""
    return ActionCache(path)


if __name__ == "__main__":
    # Test the cache
    cache = ActionCache()

    # Store some test actions
    cache.store(
        task="click login button",
        url="https://example.com/login",
        dom_hash="abc123",
        action_type="click",
        selector="#login-btn",
        confidence=0.9
    )

    # Retrieve
    action = cache.get("click login button", "https://example.com/login", "abc123")
    if action:
        print(f"Retrieved: {action.action_type} on {action.selector}")
        print(f"Hit count: {action.hit_count}")
        print(f"Confidence: {action.confidence}")

    # Stats
    print(f"\nCache stats: {cache.get_stats()}")
