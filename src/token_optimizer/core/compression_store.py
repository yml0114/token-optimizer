"""L1+: CCR Reversible Compression Store — Compression with Content Recall.

Core idea (from Headroom CCR):
  When we compress text, the original content is NOT lost. It is stored in a
  CompressionStore with a hash key, and a retrieval marker is injected at the
  end of the compressed text. The downstream LLM can retrieve the original
  content on-demand by referencing the marker.

  This enables lossy-looking compression to actually be lossless in practice,
  because the LLM can always "open" the compressed content if needed.

Storage design:
  - In-memory dict with TTL-based eviction
  - LRU eviction when capacity is exceeded
  - 12-char hex hash for compact retrieval keys
  - Default TTL: 300s, max entries: 50
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredContent:
    """A single stored original content entry."""
    hash_key: str
    original_text: str
    compressed_text: str
    stored_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = 0.0

    @property
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        return (time.time() - self.stored_at) > self.ttl

    @property
    def remaining_ttl(self) -> float:
        """Remaining TTL in seconds."""
        return max(0.0, self.ttl - (time.time() - self.stored_at))


class CompressionStore:
    """CCR (Content Compression & Recall) reversible compression store.

    Stores original content before compression, enabling the downstream LLM
    to retrieve full original text via hash-based lookup markers.

    Args:
        max_entries: Maximum number of entries before LRU eviction.
        default_ttl: Default time-to-live in seconds for each entry.
    """

    # Marker format injected into compressed text
    RETRIEVE_MARKER = "[TO:retrieve hash={hash}]"

    def __init__(self, max_entries: int = 50, default_ttl: float = 300.0):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        # OrderedDict for O(1) LRU: move_to_end on access
        self._store: OrderedDict[str, StoredContent] = OrderedDict()
        self._stats = {
            "stores": 0,
            "retrievals": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_evictions": 0,
        }

    @staticmethod
    def _generate_hash(text: str) -> str:
        """Generate a 12-character hex hash for retrieval key.

        Uses SHA-256 truncated to 12 hex chars (48 bits), giving ~2.8e14
        unique keys — more than sufficient for 50-entry in-memory store.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def store(
        self,
        original_text: str,
        compressed_text: str,
        ttl: float | None = None,
    ) -> tuple[str, str]:
        """Store original content and return (hash_key, annotated_compressed_text).

        The annotated compressed text has the retrieval marker appended:
            compressed_text + " [TO:retrieve hash=abc123def456]"

        Args:
            original_text: The text BEFORE compression.
            compressed_text: The text AFTER compression.
            ttl: Optional custom TTL in seconds. Uses default if None.

        Returns:
            Tuple of (hash_key, annotated_compressed_text).
        """
        hash_key = self._generate_hash(original_text)
        ttl = ttl if ttl is not None else self.default_ttl
        now = time.time()

        # Upsert: if same hash exists, update it
        if hash_key in self._store:
            self._store.move_to_end(hash_key)

        self._store[hash_key] = StoredContent(
            hash_key=hash_key,
            original_text=original_text,
            compressed_text=compressed_text,
            stored_at=now,
            ttl=ttl,
            access_count=0,
            last_accessed=now,
        )
        self._stats["stores"] += 1

        # Evict if over capacity
        self._evict_if_needed()

        marker = self.RETRIEVE_MARKER.format(hash=hash_key)
        annotated = f"{compressed_text} {marker}"
        return hash_key, annotated

    def retrieve(self, hash_key: str) -> str | None:
        """Retrieve original content by hash key.

        Args:
            hash_key: The 12-char hex hash from the retrieval marker.

        Returns:
            Original text if found and not expired, None otherwise.
        """
        self._stats["retrievals"] += 1

        entry = self._store.get(hash_key)
        if entry is None:
            self._stats["misses"] += 1
            return None

        if entry.is_expired:
            self._stats["misses"] += 1
            del self._store[hash_key]
            self._stats["expired_evictions"] += 1
            return None

        # LRU: move to end (most recently used)
        self._store.move_to_end(hash_key)
        entry.access_count += 1
        entry.last_accessed = time.time()
        self._stats["hits"] += 1
        return entry.original_text

    def has(self, hash_key: str) -> bool:
        """Check if a hash key exists and is not expired."""
        entry = self._store.get(hash_key)
        if entry is None:
            return False
        if entry.is_expired:
            del self._store[hash_key]
            self._stats["expired_evictions"] += 1
            return False
        return True

    def extract_hash_from_marker(self, text: str) -> str | None:
        """Extract hash key from a retrieval marker in text.

        Looks for the pattern [TO:retrieve hash=xxx] and extracts xxx.

        Args:
            text: Text that may contain a retrieval marker.

        Returns:
            Hash key if found, None otherwise.
        """
        match = re.search(r'\[TO:retrieve hash=([0-9a-f]{12})\]', text)
        return match.group(1) if match else None

    def retrieve_from_text(self, text: str) -> tuple[str | None, str]:
        """Extract marker from text, retrieve original content, and return
        (original_content_or_None, cleaned_text_without_marker).

        Args:
            text: Text that may contain a retrieval marker.

        Returns:
            Tuple of (original_content, text_with_marker_removed).
        """
        hash_key = self.extract_hash_from_marker(text)
        if hash_key is None:
            return None, text

        original = self.retrieve(hash_key)
        # Remove marker from text
        cleaned = re.sub(r'\s*\[TO:retrieve hash=[0-9a-f]{12}\]', '', text).strip()
        return original, cleaned

    def _evict_if_needed(self) -> None:
        """Evict expired entries first, then LRU if still over capacity."""
        # Phase 1: remove expired entries
        expired_keys = [
            key for key, entry in self._store.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._store[key]
            self._stats["expired_evictions"] += 1

        # Phase 2: LRU eviction if still over capacity
        while len(self._store) > self.max_entries:
            # OrderedDict: first item is least recently used
            evicted_key, _ = self._store.popitem(last=False)
            self._stats["evictions"] += 1

    def clear(self) -> None:
        """Clear all stored entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Current number of stored entries."""
        return len(self._store)

    @property
    def stats(self) -> dict[str, int]:
        """Return a copy of internal statistics."""
        return dict(self._stats)

    def get_hit_rate(self) -> float:
        """Return retrieval hit rate (0.0 to 1.0)."""
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    def cleanup_expired(self) -> int:
        """Manually remove all expired entries. Returns count removed."""
        expired_keys = [
            key for key, entry in self._store.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._store[key]
            self._stats["expired_evictions"] += 1
        return len(expired_keys)
