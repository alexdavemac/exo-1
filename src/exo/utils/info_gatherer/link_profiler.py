"""
#25 — Network Link Profiling

Measures link quality between nodes for smart task placement decisions.
Captures latency, throughput estimate, jitter, and packet loss in a
non-intrusive way (small HTTP probes + TCP RTT measurement).

The profiler runs periodically and publishes LinkProfile events so the
master's placement algorithm can factor in link quality.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
from loguru import logger

from exo.shared.types.common import NodeId


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class LinkProfile:
    """Quality profile for a single link between two nodes."""

    source: NodeId
    target: NodeId
    target_ip: str

    # Latency
    rtt_ms: float = 0.0  # median round-trip time
    rtt_min_ms: float = 0.0
    rtt_max_ms: float = 0.0
    jitter_ms: float = 0.0  # stddev of RTT samples

    # Reliability
    loss_pct: float = 0.0  # percentage of failed probes
    probes_sent: int = 0
    probes_received: int = 0

    # Throughput estimate (bytes/sec, from payload probe)
    throughput_bps: float = 0.0

    # Classification
    quality: str = "unknown"  # excellent / good / fair / poor / down

    # Metadata
    interface_type: str = "unknown"  # wifi, ethernet, thunderbolt
    measured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "source": str(self.source),
            "target": str(self.target),
            "target_ip": self.target_ip,
            "rtt_ms": round(self.rtt_ms, 2),
            "rtt_min_ms": round(self.rtt_min_ms, 2),
            "rtt_max_ms": round(self.rtt_max_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "loss_pct": round(self.loss_pct, 1),
            "throughput_bps": round(self.throughput_bps),
            "quality": self.quality,
            "interface_type": self.interface_type,
            "measured_at": self.measured_at,
        }


@dataclass
class ClusterLinkMap:
    """All measured links in the cluster."""

    profiles: Dict[Tuple[NodeId, NodeId], LinkProfile] = field(default_factory=dict)

    def get(self, source: NodeId, target: NodeId) -> Optional[LinkProfile]:
        return self.profiles.get((source, target))

    def set(self, profile: LinkProfile) -> None:
        self.profiles[(profile.source, profile.target)] = profile

    def best_link_to(self, target: NodeId) -> Optional[LinkProfile]:
        """Get the best quality link to a target across all sources."""
        candidates = [p for (_, t), p in self.profiles.items() if t == target]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.rtt_ms)

    def all_profiles(self) -> List[LinkProfile]:
        return list(self.profiles.values())

    def summary(self) -> dict:
        """Cluster-wide link quality summary."""
        all_p = self.all_profiles()
        if not all_p:
            return {"total_links": 0}

        quality_counts = {}
        for p in all_p:
            quality_counts[p.quality] = quality_counts.get(p.quality, 0) + 1

        return {
            "total_links": len(all_p),
            "avg_rtt_ms": round(statistics.mean(p.rtt_ms for p in all_p), 2),
            "avg_jitter_ms": round(statistics.mean(p.jitter_ms for p in all_p), 2),
            "avg_loss_pct": round(statistics.mean(p.loss_pct for p in all_p), 1),
            "quality_distribution": quality_counts,
        }


# ─── Profiling Functions ────────────────────────────────────────────

PROBE_COUNT = 5
PROBE_TIMEOUT_S = 3.0
THROUGHPUT_PAYLOAD_KB = 64  # Small payload for bandwidth estimation


def _classify_quality(rtt_ms: float, jitter_ms: float, loss_pct: float) -> str:
    """Classify link quality based on measured parameters."""
    if loss_pct >= 50:
        return "down"
    if loss_pct >= 20 or rtt_ms > 500:
        return "poor"
    if loss_pct >= 5 or rtt_ms > 100 or jitter_ms > 50:
        return "fair"
    if rtt_ms > 20 or jitter_ms > 10:
        return "good"
    return "excellent"


async def probe_link(
    source_id: NodeId,
    target_id: NodeId,
    target_ip: str,
    port: int = 52415,
    interface_type: str = "unknown",
) -> LinkProfile:
    """
    Measure link quality to a specific target node/IP.

    Uses HTTP GET /node_id as the probe endpoint (lightweight, always available).
    """
    rtt_samples: List[float] = []
    failures = 0

    timeout = httpx.Timeout(timeout=PROBE_TIMEOUT_S)

    if ":" in target_ip:
        base_url = f"http://[{target_ip}]:{port}"
    else:
        base_url = f"http://{target_ip}:{port}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        # --- Latency / loss probes ---
        for _ in range(PROBE_COUNT):
            t0 = time.monotonic()
            try:
                r = await client.get(f"{base_url}/node_id")
                if r.status_code == 200:
                    rtt_samples.append((time.monotonic() - t0) * 1000)
                else:
                    failures += 1
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError):
                failures += 1
            # Small inter-probe gap to avoid burst
            await asyncio.sleep(0.05)

        # --- Throughput estimate ---
        throughput_bps = 0.0
        if rtt_samples:
            try:
                # Request the /health endpoint which returns a small JSON payload
                # Measure how fast a slightly larger exchange completes
                t0 = time.monotonic()
                r = await client.get(f"{base_url}/health")
                elapsed_s = time.monotonic() - t0
                if r.status_code == 200 and elapsed_s > 0:
                    payload_bytes = len(r.content)
                    throughput_bps = payload_bytes / elapsed_s
            except Exception:
                pass

    probes_sent = PROBE_COUNT
    probes_received = len(rtt_samples)
    loss_pct = (failures / probes_sent) * 100 if probes_sent > 0 else 100

    if rtt_samples:
        rtt_median = statistics.median(rtt_samples)
        rtt_min = min(rtt_samples)
        rtt_max = max(rtt_samples)
        jitter = statistics.stdev(rtt_samples) if len(rtt_samples) >= 2 else 0.0
    else:
        rtt_median = rtt_min = rtt_max = jitter = 0.0

    quality = _classify_quality(rtt_median, jitter, loss_pct)

    profile = LinkProfile(
        source=source_id,
        target=target_id,
        target_ip=target_ip,
        rtt_ms=rtt_median,
        rtt_min_ms=rtt_min,
        rtt_max_ms=rtt_max,
        jitter_ms=jitter,
        loss_pct=loss_pct,
        probes_sent=probes_sent,
        probes_received=probes_received,
        throughput_bps=throughput_bps,
        quality=quality,
        interface_type=interface_type,
        measured_at=time.time(),
    )

    logger.debug(
        f"Link {source_id} → {target_id} ({target_ip}): "
        f"rtt={rtt_median:.1f}ms jitter={jitter:.1f}ms loss={loss_pct:.0f}% "
        f"quality={quality}"
    )
    return profile


async def profile_all_links(
    self_node_id: NodeId,
    reachable_nodes: Dict[NodeId, set],
    interface_types: Optional[Dict[str, str]] = None,
) -> ClusterLinkMap:
    """
    Profile all reachable links from this node.

    Args:
        self_node_id: This node's ID.
        reachable_nodes: Map of node_id → set of reachable IPs (from check_reachable).
        interface_types: Optional map of IP → interface type.
    """
    link_map = ClusterLinkMap()
    tasks = []

    for target_id, ips in reachable_nodes.items():
        if target_id == self_node_id:
            continue
        for ip in ips:
            itype = (interface_types or {}).get(ip, "unknown")
            tasks.append(probe_link(self_node_id, target_id, ip, interface_type=itype))

    if not tasks:
        return link_map

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, LinkProfile):
            link_map.set(result)
        elif isinstance(result, Exception):
            logger.warning(f"Link probe failed: {result}")

    logger.info(
        f"Profiled {len(link_map.all_profiles())} links from {self_node_id}: "
        f"{link_map.summary()}"
    )
    return link_map
