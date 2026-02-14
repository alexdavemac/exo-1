"""
#24 — Request Timeout & Cancellation

Provides per-request timeout enforcement and cooperative cancellation
for the Exo inference cluster.

Features:
- Per-request timeout with configurable defaults and overrides
- Request cancellation via cancel token
- Request tracking with creation time, status, and metadata
- Automatic cleanup of stale/completed requests
- Integration-ready with Master command processor
"""

import asyncio
import enum
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from loguru import logger


# ─── Configuration ──────────────────────────────────────────────────

# Default timeout for inference requests (seconds)
DEFAULT_REQUEST_TIMEOUT_S = float(os.environ.get("EXO_REQUEST_TIMEOUT", "300"))
# Maximum allowed timeout (prevents abuse)
MAX_REQUEST_TIMEOUT_S = float(os.environ.get("EXO_MAX_REQUEST_TIMEOUT", "600"))
# How often to sweep stale requests
SWEEP_INTERVAL_S = 30.0
# How long completed requests are kept for status queries
COMPLETED_RETENTION_S = 120.0


# ─── Data Structures ────────────────────────────────────────────────


class RequestStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class RequestHandle:
    """Tracks a single inference request throughout its lifecycle."""

    request_id: str
    created_at: float = field(default_factory=time.time)
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    status: RequestStatus = RequestStatus.PENDING
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal cancellation machinery
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def elapsed_s(self) -> float:
        end = self.completed_at or time.time()
        return end - self.created_at

    @property
    def remaining_s(self) -> float:
        return max(0, self.timeout_s - self.elapsed_s)

    @property
    def is_expired(self) -> bool:
        return self.elapsed_s > self.timeout_s and self.status in (
            RequestStatus.PENDING,
            RequestStatus.RUNNING,
        )

    @property
    def is_active(self) -> bool:
        return self.status in (RequestStatus.PENDING, RequestStatus.RUNNING)

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Signal cancellation. Cooperating code should poll is_cancelled."""
        self._cancel_event.set()
        if self.is_active:
            self.status = RequestStatus.CANCELLED
            self.completed_at = time.time()

    def mark_running(self) -> None:
        if self.status == RequestStatus.PENDING:
            self.status = RequestStatus.RUNNING

    def mark_completed(self) -> None:
        self.status = RequestStatus.COMPLETED
        self.completed_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = RequestStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def mark_timed_out(self) -> None:
        self.status = RequestStatus.TIMED_OUT
        self.completed_at = time.time()
        self._cancel_event.set()

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "elapsed_s": round(self.elapsed_s, 2),
            "timeout_s": self.timeout_s,
            "remaining_s": round(self.remaining_s, 2),
            "error": self.error,
            "metadata": self.metadata,
        }


# ─── Request Tracker ────────────────────────────────────────────────


class RequestTracker:
    """
    Centrally tracks all active inference requests with timeout enforcement.

    Usage:
        tracker = RequestTracker()
        handle = tracker.create("cmd_abc123", timeout_s=120)

        # In the request processing coroutine:
        result = await tracker.run_with_timeout(
            handle,
            some_async_work(handle),
        )

        # Or check manually:
        if handle.is_cancelled:
            return  # abort

        # Cancel externally:
        tracker.cancel("cmd_abc123")
    """

    def __init__(self):
        self._requests: Dict[str, RequestHandle] = {}
        self._sweep_task: Optional[asyncio.Task] = None

    def create(
        self,
        request_id: str,
        timeout_s: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RequestHandle:
        """Create and register a new request handle."""
        timeout = min(
            timeout_s or DEFAULT_REQUEST_TIMEOUT_S,
            MAX_REQUEST_TIMEOUT_S,
        )
        handle = RequestHandle(
            request_id=request_id,
            timeout_s=timeout,
            metadata=metadata or {},
        )
        self._requests[request_id] = handle
        logger.debug(f"Request created: {request_id} timeout={timeout}s")
        return handle

    def get(self, request_id: str) -> Optional[RequestHandle]:
        return self._requests.get(request_id)

    def cancel(self, request_id: str) -> bool:
        """Cancel a request by ID. Returns True if found and cancelled."""
        handle = self._requests.get(request_id)
        if handle and handle.is_active:
            handle.cancel()
            logger.info(f"Request cancelled: {request_id} after {handle.elapsed_s:.1f}s")
            return True
        return False

    async def run_with_timeout(
        self,
        handle: RequestHandle,
        coro: Coroutine,
    ) -> Any:
        """
        Run a coroutine with timeout enforcement and cancellation support.

        Raises:
            asyncio.TimeoutError: If the request times out.
            asyncio.CancelledError: If the request is cancelled.
        """
        handle.mark_running()

        try:
            result = await asyncio.wait_for(coro, timeout=handle.timeout_s)
            handle.mark_completed()
            return result
        except asyncio.TimeoutError:
            handle.mark_timed_out()
            logger.warning(
                f"Request timed out: {handle.request_id} "
                f"after {handle.elapsed_s:.1f}s (limit={handle.timeout_s}s)"
            )
            raise
        except asyncio.CancelledError:
            handle.cancel()
            raise
        except Exception as e:
            handle.mark_failed(str(e))
            raise

    def list_active(self) -> List[RequestHandle]:
        """Get all active (pending/running) requests."""
        return [h for h in self._requests.values() if h.is_active]

    def list_all(self) -> List[RequestHandle]:
        return list(self._requests.values())

    def stats(self) -> dict:
        """Get request tracking statistics."""
        all_handles = list(self._requests.values())
        by_status = {}
        for h in all_handles:
            by_status[h.status.value] = by_status.get(h.status.value, 0) + 1

        active = [h for h in all_handles if h.is_active]
        return {
            "total_tracked": len(all_handles),
            "active": len(active),
            "by_status": by_status,
            "oldest_active_s": round(max((h.elapsed_s for h in active), default=0), 1),
        }

    def sweep(self) -> int:
        """Remove completed requests older than retention period."""
        now = time.time()
        to_remove = []
        for rid, h in self._requests.items():
            if not h.is_active and h.completed_at:
                if (now - h.completed_at) > COMPLETED_RETENTION_S:
                    to_remove.append(rid)
            # Also expire timed-out active requests that weren't caught
            if h.is_active and h.is_expired:
                h.mark_timed_out()

        for rid in to_remove:
            del self._requests[rid]

        if to_remove:
            logger.debug(f"Swept {len(to_remove)} stale request records")
        return len(to_remove)

    async def start_sweep_loop(self) -> None:
        """Start background sweep task (call once)."""
        if self._sweep_task is not None:
            return

        async def _loop():
            while True:
                await asyncio.sleep(SWEEP_INTERVAL_S)
                self.sweep()

        self._sweep_task = asyncio.create_task(_loop())

    def stop(self) -> None:
        if self._sweep_task:
            self._sweep_task.cancel()
            self._sweep_task = None


# Global singleton
request_tracker = RequestTracker()
