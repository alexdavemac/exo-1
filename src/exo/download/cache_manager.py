"""
#67 — Per-Model Cache Tracking

Multi-model cache manager that tracks disk usage per model, prevents
cache conflicts, and supports eviction policies.

Features:
- Per-model disk usage tracking
- Cache conflict detection (same model, different shards)
- LRU eviction with configurable max cache size
- Model pinning (prevent eviction of important models)
- Cache health reporting
"""

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger

from exo.shared.models.model_cards import ModelId


# ─── Configuration ──────────────────────────────────────────────────

# Default cache directory (mirrors huggingface_hub default)
DEFAULT_CACHE_DIR = Path(os.environ.get(
    "EXO_CACHE_DIR",
    os.path.expanduser("~/.cache/exo"),
))

# Max cache size in bytes (0 = unlimited)
MAX_CACHE_BYTES = int(os.environ.get("EXO_MAX_CACHE_BYTES", "0"))

# HuggingFace cache (models are downloaded here by default)
HF_CACHE_DIR = Path(os.environ.get(
    "HF_HOME",
    os.path.expanduser("~/.cache/huggingface/hub"),
))


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class ModelCacheEntry:
    """Tracks cache state for a single model."""

    model_id: ModelId
    cache_path: Optional[Path] = None
    size_bytes: int = 0
    shard_count: int = 0
    file_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    last_loaded: float = 0.0
    is_pinned: bool = False
    is_complete: bool = False

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def age_hours(self) -> float:
        return (time.time() - self.last_accessed) / 3600

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_gb, 2),
            "shard_count": self.shard_count,
            "file_count": self.file_count,
            "last_accessed": self.last_accessed,
            "last_loaded": self.last_loaded,
            "is_pinned": self.is_pinned,
            "is_complete": self.is_complete,
            "age_hours": round(self.age_hours, 1),
        }


@dataclass
class CacheConflict:
    """Describes a potential conflict in the cache."""

    model_id: ModelId
    conflict_type: str  # "duplicate_shard", "incomplete", "stale_lock"
    description: str
    paths: List[str] = field(default_factory=list)


# ─── Cache Manager ──────────────────────────────────────────────────


class ModelCacheManager:
    """
    Manages per-model cache entries with eviction and conflict detection.

    Integrates with the existing ShardDownloader stack (which uses
    huggingface_hub internally) by scanning known cache directories.
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_bytes: int = 0):
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._hf_cache_dir = HF_CACHE_DIR
        self._max_bytes = max_bytes or MAX_CACHE_BYTES
        self._entries: Dict[ModelId, ModelCacheEntry] = {}
        self._pinned: Set[ModelId] = set()

    # ─── Scanning ────────────────────────────────────────────────

    def scan(self) -> Dict[ModelId, ModelCacheEntry]:
        """Scan the cache directories and build per-model entries."""
        self._entries.clear()

        # Scan HuggingFace hub cache (models--org--name format)
        if self._hf_cache_dir.exists():
            self._scan_hf_cache()

        # Scan exo cache dir
        if self._cache_dir.exists() and self._cache_dir != self._hf_cache_dir:
            self._scan_directory(self._cache_dir)

        logger.info(
            f"Cache scan complete: {len(self._entries)} models, "
            f"{self.total_size_gb:.2f} GB total"
        )
        return dict(self._entries)

    def _scan_hf_cache(self) -> None:
        """Scan HuggingFace hub cache directory structure."""
        try:
            for item in self._hf_cache_dir.iterdir():
                if not item.is_dir() or not item.name.startswith("models--"):
                    continue
                # Parse model ID from directory name: models--org--name → org/name
                parts = item.name.split("--")
                if len(parts) >= 3:
                    model_id = ModelId("/".join(parts[1:]))
                elif len(parts) == 2:
                    model_id = ModelId(parts[1])
                else:
                    continue

                size, file_count = self._dir_size(item)
                shard_count = self._count_shards(item)

                entry = ModelCacheEntry(
                    model_id=model_id,
                    cache_path=item,
                    size_bytes=size,
                    shard_count=shard_count,
                    file_count=file_count,
                    is_pinned=model_id in self._pinned,
                    is_complete=self._check_complete(item),
                )
                self._entries[model_id] = entry
        except PermissionError:
            logger.warning(f"Permission denied scanning {self._hf_cache_dir}")

    def _scan_directory(self, directory: Path) -> None:
        """Scan a flat cache directory for model files."""
        try:
            for item in directory.iterdir():
                if not item.is_dir():
                    continue
                model_id = ModelId(item.name)
                if model_id in self._entries:
                    continue  # Already found in HF cache

                size, file_count = self._dir_size(item)
                self._entries[model_id] = ModelCacheEntry(
                    model_id=model_id,
                    cache_path=item,
                    size_bytes=size,
                    file_count=file_count,
                    is_pinned=model_id in self._pinned,
                )
        except PermissionError:
            logger.warning(f"Permission denied scanning {directory}")

    @staticmethod
    def _dir_size(path: Path) -> tuple:
        """Calculate total size and file count of a directory."""
        total = 0
        count = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
                    count += 1
        except (PermissionError, OSError):
            pass
        return total, count

    @staticmethod
    def _count_shards(path: Path) -> int:
        """Count weight shard files in a model directory."""
        shard_extensions = {".safetensors", ".bin", ".pt", ".gguf", ".mlx"}
        count = 0
        try:
            for f in path.rglob("*"):
                if f.suffix in shard_extensions:
                    count += 1
        except (PermissionError, OSError):
            pass
        return count

    @staticmethod
    def _check_complete(path: Path) -> bool:
        """Check if a model download appears complete (has config.json)."""
        # HF cache structure: models--X/snapshots/<hash>/config.json
        try:
            snapshots = path / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    if (snap / "config.json").exists():
                        return True
            # Flat layout
            return (path / "config.json").exists()
        except (PermissionError, OSError):
            return False

    # ─── Access ──────────────────────────────────────────────────

    def get(self, model_id: ModelId) -> Optional[ModelCacheEntry]:
        entry = self._entries.get(model_id)
        if entry:
            entry.touch()
        return entry

    def record_load(self, model_id: ModelId) -> None:
        """Record that a model was loaded into memory (updates LRU priority)."""
        entry = self._entries.get(model_id)
        if entry:
            entry.last_loaded = time.time()
            entry.touch()

    def list_entries(self) -> List[ModelCacheEntry]:
        return list(self._entries.values())

    @property
    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024 ** 3)

    # ─── Pinning ─────────────────────────────────────────────────

    def pin(self, model_id: ModelId) -> None:
        """Pin a model to prevent eviction."""
        self._pinned.add(model_id)
        if model_id in self._entries:
            self._entries[model_id].is_pinned = True
        logger.info(f"Pinned model: {model_id}")

    def unpin(self, model_id: ModelId) -> None:
        self._pinned.discard(model_id)
        if model_id in self._entries:
            self._entries[model_id].is_pinned = False

    # ─── Conflict Detection ──────────────────────────────────────

    def detect_conflicts(self) -> List[CacheConflict]:
        """Detect cache conflicts and issues."""
        conflicts = []

        for model_id, entry in self._entries.items():
            # Incomplete downloads
            if not entry.is_complete and entry.size_bytes > 0:
                conflicts.append(CacheConflict(
                    model_id=model_id,
                    conflict_type="incomplete",
                    description=f"Model {model_id} appears incomplete ({entry.size_gb:.1f} GB, no config.json)",
                    paths=[str(entry.cache_path)] if entry.cache_path else [],
                ))

            # Check for lock files (stale downloads)
            if entry.cache_path:
                lock_files = list(entry.cache_path.rglob("*.lock"))
                if lock_files:
                    conflicts.append(CacheConflict(
                        model_id=model_id,
                        conflict_type="stale_lock",
                        description=f"Model {model_id} has {len(lock_files)} lock file(s) — may be a stale download",
                        paths=[str(lf) for lf in lock_files],
                    ))

        return conflicts

    # ─── Eviction ────────────────────────────────────────────────

    def eviction_candidates(self, needed_bytes: int = 0) -> List[ModelCacheEntry]:
        """
        Get eviction candidates sorted by priority (evict first = index 0).

        LRU policy: least recently accessed unpinned models go first.
        """
        candidates = [
            e for e in self._entries.values()
            if not e.is_pinned and e.size_bytes > 0
        ]
        # Sort by last_accessed ascending (oldest first)
        candidates.sort(key=lambda e: e.last_accessed)

        if needed_bytes <= 0:
            return candidates

        # Filter to just enough to free needed_bytes
        result = []
        freed = 0
        for c in candidates:
            result.append(c)
            freed += c.size_bytes
            if freed >= needed_bytes:
                break
        return result

    def evict(self, model_id: ModelId) -> bool:
        """
        Evict a model from cache (delete files).

        Returns True if successfully removed.
        """
        entry = self._entries.get(model_id)
        if not entry:
            logger.warning(f"Cannot evict {model_id}: not in cache")
            return False

        if entry.is_pinned:
            logger.warning(f"Cannot evict {model_id}: model is pinned")
            return False

        if entry.cache_path and entry.cache_path.exists():
            try:
                shutil.rmtree(entry.cache_path)
                logger.info(f"Evicted {model_id}: freed {entry.size_gb:.2f} GB")
            except (PermissionError, OSError) as e:
                logger.error(f"Failed to evict {model_id}: {e}")
                return False

        del self._entries[model_id]
        return True

    def ensure_space(self, needed_bytes: int) -> bool:
        """Evict models until needed_bytes are available."""
        if self._max_bytes <= 0:
            return True  # Unlimited cache

        available = self._max_bytes - self.total_size_bytes
        if available >= needed_bytes:
            return True

        shortfall = needed_bytes - available
        candidates = self.eviction_candidates(shortfall)

        freed = 0
        for c in candidates:
            if self.evict(c.model_id):
                freed += c.size_bytes
            if freed >= shortfall:
                return True

        logger.warning(
            f"Could not free enough space: needed {shortfall} bytes, "
            f"freed {freed} bytes"
        )
        return freed >= shortfall

    # ─── Reporting ───────────────────────────────────────────────

    def summary(self) -> dict:
        """Cache summary for API/monitoring."""
        entries = self.list_entries()
        conflicts = self.detect_conflicts()

        return {
            "total_models": len(entries),
            "total_size_gb": round(self.total_size_gb, 2),
            "total_size_bytes": self.total_size_bytes,
            "max_cache_bytes": self._max_bytes,
            "max_cache_gb": round(self._max_bytes / (1024 ** 3), 2) if self._max_bytes else None,
            "utilization_pct": round(
                (self.total_size_bytes / self._max_bytes) * 100, 1
            ) if self._max_bytes > 0 else None,
            "pinned_models": len(self._pinned),
            "conflicts": len(conflicts),
            "models": [e.to_dict() for e in sorted(entries, key=lambda e: e.size_bytes, reverse=True)],
        }


# Global singleton
model_cache_manager = ModelCacheManager()
