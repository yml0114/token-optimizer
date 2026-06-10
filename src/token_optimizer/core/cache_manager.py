"""L2: Prefix Cache Manager — Track and maximize API cache hit rates.

Core responsibilities:
1. Track prefix hashes across requests to detect changes
2. Provide cache hit rate metrics
3. Manage prefix lifecycle (detect invalidation, trigger re-cache)

DeepSeek: automatic cache from token 0 forward, 64-token granularity
MiMo: cache writes currently free, 1024-token minimum prefix
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CacheEntry:
    """One cache tracking entry."""
    prefix_hash: str
    first_seen: float
    last_hit: float
    hit_count: int = 0
    miss_count: int = 0
    prefix_tokens_est: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class CacheManager:
    """Tracks prefix cache state and hit rates.

    Uses SQLite for persistence across process restarts.
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".token_optimizer" / "cache.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prefix_cache (
                    prefix_hash TEXT PRIMARY KEY,
                    first_seen REAL NOT NULL,
                    last_hit REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    miss_count INTEGER DEFAULT 0,
                    prefix_tokens_est INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    prefix_hash TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cached_tokens INTEGER DEFAULT 0,
                    model TEXT DEFAULT '',
                    estimated_cost REAL DEFAULT 0.0
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def record_request(
        self,
        prefix_hash: str,
        cache_hit: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        model: str = "",
        estimated_cost: float = 0.0,
        prefix_tokens_est: int = 0,
    ):
        """Record one API request for cache tracking."""
        now = time.time()

        # Update prefix cache entry
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT hit_count, miss_count FROM prefix_cache WHERE prefix_hash = ?",
                (prefix_hash,),
            ).fetchone()

            if existing:
                if cache_hit:
                    conn.execute(
                        """UPDATE prefix_cache
                           SET last_hit = ?, hit_count = hit_count + 1
                           WHERE prefix_hash = ?""",
                        (now, prefix_hash),
                    )
                else:
                    conn.execute(
                        """UPDATE prefix_cache
                           SET miss_count = miss_count + 1
                           WHERE prefix_hash = ?""",
                        (prefix_hash,),
                    )
            else:
                conn.execute(
                    """INSERT INTO prefix_cache
                       (prefix_hash, first_seen, last_hit, hit_count, miss_count, prefix_tokens_est)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        prefix_hash,
                        now,
                        now,
                        1 if cache_hit else 0,
                        0 if cache_hit else 1,
                        prefix_tokens_est,
                    ),
                )

            # Log the request
            conn.execute(
                """INSERT INTO request_log
                   (timestamp, prefix_hash, cache_hit, input_tokens, output_tokens,
                    cached_tokens, model, estimated_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, prefix_hash, int(cache_hit), input_tokens, output_tokens,
                 cached_tokens, model, estimated_cost),
            )

    def get_overall_stats(self) -> dict:
        """Get overall cache statistics."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as hits,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) as misses,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(cached_tokens) as total_cached_tokens,
                    SUM(estimated_cost) as total_cost,
                    MIN(timestamp) as first_request,
                    MAX(timestamp) as last_request
                FROM request_log
            """).fetchone()

        total = row[0] or 0
        hits = row[1] or 0
        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": row[2] or 0,
            "hit_rate": round(hits / total * 100, 1) if total > 0 else 0.0,
            "total_input_tokens": row[3] or 0,
            "total_output_tokens": row[4] or 0,
            "total_cached_tokens": row[5] or 0,
            "total_cost": round(row[6] or 0, 6),
            "first_request": row[7],
            "last_request": row[8],
        }

    def get_prefix_stats(self) -> list[dict]:
        """Get per-prefix cache statistics."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT prefix_hash, first_seen, last_hit, hit_count, miss_count, prefix_tokens_est
                FROM prefix_cache
                ORDER BY hit_count DESC
            """).fetchall()

        return [
            {
                "prefix_hash": r[0],
                "first_seen": r[1],
                "last_hit": r[2],
                "hit_count": r[3],
                "miss_count": r[4],
                "hit_rate": round(r[3] / (r[3] + r[4]) * 100, 1) if (r[3] + r[4]) > 0 else 0,
                "prefix_tokens_est": r[5],
            }
            for r in rows
        ]
