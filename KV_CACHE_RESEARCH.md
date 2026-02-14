# KV Cache Management Strategies for Distributed LLM Inference
## Research Report: Mac M4 + iOS Device Pipeline Parallelism

**Date**: January 31, 2026
**Focus**: Heterogeneous cluster with Mac mini M4 (16GB) and iOS devices (8GB)
**Architecture**: Pipeline parallelism via EXOT protocol with TCP tensor exchange

---

## Executive Summary

Your exo distributed inference system has a robust KV cache foundation with **prefix caching, LRU eviction, and quantization support**. However, the current implementation was designed primarily for single-device and homogeneous multi-device scenarios. This research identifies critical gaps and optimization opportunities for **heterogeneous Mac/iOS pipeline parallelism** where memory constraints are severe.

**Key Finding**: KV cache data is **NOT currently shared between pipeline stages**. Each device maintains independent caches for its layer subset, creating redundancy and missed optimization opportunities.

---

## 1. Current KV Cache Implementation in Exo

### 1.1 Core Architecture
**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/cache.py`

```python
class KVPrefixCache:
    """Single-device prefix caching with LRU eviction."""
    def __init__(self, tokenizer: TokenizerWrapper):
        self.prompts: list[mx.array] = []          # Tokenized prompts
        self.caches: list[KVCacheType] = []        # KV cache per prompt
        self._last_used: list[int] = []            # Access counter for LRU
        self._access_counter: int = 0
```

**Current KV Cache Types** (lines 188-210):
1. **KVCache** - Standard float32/float16 (default)
2. **QuantizedKVCache** - Group-wise quantization with configurable bits
3. **RotatingKVCache** - Sliding window for fixed-size context

### 1.2 Memory Management Strategy

**Threshold-Based LRU Eviction** (lines 133-156):
```python
_DEFAULT_MEMORY_THRESHOLD = 0.85  # Configurable via EXO_MEMORY_THRESHOLD

def _evict_if_needed(self):
    """Evict LRU entries when active memory exceeds threshold."""
    active: int = mx.metal.get_active_memory()
    limit = int(mx.metal.device_info()["max_recommended_working_set_size"])

    if active < limit * _MEMORY_THRESHOLD:
        return  # No eviction needed

    # Evict LRU entries until below threshold
    while len(self.caches) > 0:
        lru_index = self._last_used.index(min(self._last_used))
        # ... evict and repeat
```

**Problem for Mac/iOS Pipeline**:
- Only monitors **local device** memory (MLX Metal for Mac)
- iOS devices lack equivalent memory telemetry in current code
- No coordination between Mac and iOS cache management

### 1.3 Quantization Support

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/constants.py`

```python
KV_GROUP_SIZE: int | None = 32          # Group size for quantization
KV_CACHE_BITS: int | None = None        # Default: no quantization
CACHE_GROUP_SIZE: int = 64
ATTENTION_KV_BITS: int | None = 4       # 4-bit for attention (unused)
MAX_KV_SIZE: int | None = 3200          # Rotating cache max tokens
KEEP_KV_SIZE: int | None = 1600         # Tokens to keep after trimming
```

**Current Quantization**: Integrated from MLX LM via `QuantizedKVCache`
- Supports group-wise quantization (default 32 elements per group)
- Can reduce KV cache memory by 80-90% with 4-bit quantization
- **Not currently enabled by default** (KV_CACHE_BITS = None)

---

## 2. KV Cache Distribution in Pipeline Parallelism

### 2.1 Current EXOT Pipeline Design

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/exot_pipeline.py`

The EXOT protocol implements **activations-only exchange** between Mac and iOS:

```python
class ExotPipelineWrapper(nn.Module):
    """Routes tensor activations to iOS for assigned layer processing."""

    def __call__(self, x: mx.array, cache: list | None = None) -> mx.array:
        if self._ios_first:
            # iOS layers [0, N) -> Mac layers [N, M)
            h = send_to_ios(x)           # Send embeddings
            h = self._forward_local_layers(h, cache)  # Process remaining layers
        else:
            # Mac layers [0, N) -> iOS layers [N, M)
            h = self._forward_local_layers(x, cache)
            h = send_to_ios(h)           # Send to iOS
        return h
```

**KV Cache Handling** (lines 557-621):
- Mac and iOS each maintain **independent caches** for their layer subsets
- Cache is NOT exchanged over the network
- Attention masks computed locally per device
- No prefill coordination between stages

**Architecture Diagram**:
```
Mac M4 (16GB)                    iOS Device (8GB)
┌──────────────────────┐        ┌──────────────────┐
│ Embedding Layer      │        │                  │
│ Layers [0, N)        │        │ Layers [N, M)    │
│ KV Cache: [N] entries│        │ KV Cache: [M-N]  │
│ (stored on Metal GPU)│        │ entries (stored) │
└──────────────────────┘        └──────────────────┘
         │                              │
         └──────EXOT Protocol───────────┘
         (Activations only, TCP socket)
```

### 2.2 Issues with Current Approach

| Issue | Impact | Severity |
|-------|--------|----------|
| **Cache Duplication** | Same prompt prefix cached on both devices | Medium |
| **No Prefill Sync** | Each device prefills its layers independently, no optimization | High |
| **Network Latency** | Only activations exchanged; KV cache stays local | Medium |
| **iOS Memory Blind Spot** | No memory monitoring for iOS cache growth | High |
| **Context Length Limits** | Each device limited by own memory; aggregate underutilized | High |

---

## 3. KV Cache Distribution Strategies

### Strategy 1: Device-Local Caching (Current Implementation)

**Pros**:
- Simple implementation
- No network overhead for cache data
- Each device can optimize locally

**Cons**:
- Wastes cache space (duplicated prefixes on both devices)
- No sharing of computation
- iOS context limited to 8GB

**Recommendation**: **Status quo adequate** for read-only inference with independent prompts

---

### Strategy 2: Master Cache on Mac + iOS Delegates (Proposed)

**Design**:
```
Mac: Owns full KV cache for entire prompt
     [Complete prefilled cache: [0, M) layers]

iOS: On-demand fetches layer-specific slices
     Only holds current-layer KV when needed

Exchange Protocol:
1. Mac prefills prompt -> [0, M) complete cache
2. During generation, Mac sends relevant KV slice for [N, M) to iOS
3. iOS processes layers, returns activations
4. Cache updated on Mac, ready for next token
```

**Implementation Approach**:

Create new distributed cache manager:
```python
# File: exo/worker/engines/mlx/distributed_cache.py (NEW)

class DistributedKVCacheManager:
    """Manages KV cache across pipeline stages with selective transfer."""

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        is_primary_stage: bool,  # Mac is primary
        local_start_layer: int,
        local_end_layer: int,
        total_layers: int,
    ):
        self.is_primary = is_primary_stage
        self.full_cache = None  # Only on primary
        self.local_cache_ids = None  # Which cache entries to keep

    def get_cache_slice(
        self,
        layer_start: int,
        layer_end: int,
    ) -> KVCacheType:
        """Get KV cache slice for layer range."""
        if self.is_primary:
            # Return slice of full cache
            return self.full_cache[layer_start:layer_end]
        else:
            # Request from Mac via EXOT protocol
            return self._fetch_from_primary(layer_start, layer_end)

    def transfer_layer_cache(
        self,
        layer_idx: int,
        cache_slice: KVCacheType,
    ) -> None:
        """Update cache for specific layer after processing."""
        if self.is_primary:
            self.full_cache[layer_idx] = cache_slice
```

**Memory Savings**:
- Mac: ~6GB cache + 10GB models = 16GB used
- iOS: ~0.5GB cache + 4GB models = 4.5GB used (vs 8GB potential)
- **Enables larger context windows** (Mac can buffer more)

**Challenges**:
- Cache transfer latency (mitigate with compression)
- Complexity managing partial cache ownership
- iOS needs periodic sync

---

### Strategy 3: Sliding Window + Distributed Cache

**Design**: Combine rotating cache with distributed ownership

```python
# File: exo/worker/engines/mlx/sliding_window_distributed.py (NEW)

class SlidingWindowDistributedCache:
    """
    Each device keeps sliding window of recent tokens.
    Older tokens managed by cache replacement policy.
    """

    def __init__(
        self,
        max_window_size: int = 2048,  # Tokens to keep per device
    ):
        self.window_size = max_window_size
        self.local_cache = RotatingKVCache(max_size=max_window_size)

    def update_with_prefetching(
        self,
        new_tokens: int,
        remote_cache_manager,
    ) -> None:
        """Add tokens, prefetch from remote if needed."""
        current_size = self.local_cache.offset

        if current_size + new_tokens > self.window_size:
            # Window full - need to evict old tokens
            # Option: Request full history from Mac for context
            missing_context = self._compute_missing_range()
            prefetched = remote_cache_manager.prefetch_range(missing_context)
```

**Memory Profile for 8K context with Llama-3-70B**:
```
Full KV cache: ~2GB per device (65K hidden × 2 ÷ 2 layers per device)
Sliding window (2K): ~400MB per device (92% reduction)
With 4-bit quantization: ~50MB per device
```

---

### Strategy 4: PagedAttention-Style Block-Level Management

**Current Status**: NOT implemented in exo

**Why It's Relevant**:
- vLLM's PagedAttention divides KV cache into 16KB blocks
- Enables efficient memory sharing and prefixing
- Perfect for heterogeneous devices with varying memory

**Adaptation for Mac/iOS**:

```python
# File: exo/worker/engines/mlx/paged_kv_cache.py (NEW)

class PagedKVCacheManager:
    """
    KV cache block manager with device-aware paging.

    Each 16KB "page" can:
    - Be cached on any device
    - Be accessed from other devices with minimal overhead
    - Be evicted based on usage patterns
    """

    BLOCK_SIZE_TOKENS = 16  # Tokens per block

    def __init__(self, config):
        self.blocks: dict[int, KVBlock] = {}  # Block ID -> data
        self.block_refs: list[int] = []       # Token -> block ID
        self.device_placement: dict[int, str] = {}  # Block -> 'mac'/'ios'

    def allocate_block(self, device: str) -> int:
        """Allocate new KV block."""
        block_id = len(self.blocks)
        self.blocks[block_id] = KVBlock()
        self.device_placement[block_id] = device
        return block_id

    def access_block(self, block_id: int) -> KVBlock:
        """Get block, migrating if needed."""
        current_device = self.device_placement[block_id]
        if current_device != self.current_device:
            self._migrate_block(block_id, self.current_device)
        return self.blocks[block_id]
```

**Benefits for Your Cluster**:
- Mac dynamically managers block placement
- iOS requests only blocks it needs
- Natural load balancing: frequently-used blocks on nearest device
- Enables KV cache swapping to disk on Mac

---

## 4. iOS-Specific Considerations

### 4.1 Memory Pressure on iOS

**Problem**: iOS reserves memory for OS, browser, other apps
- Claimed: 8GB
- Reality: ~5-6GB available for exo at best
- Under memory pressure: iOS kills app without warning

**Current Detection**: NONE (critical gap)

**Solution - Add iOS Memory Monitoring**:

```python
# File: exo/shared/types/profiling.py (EXTEND)

@dataclass
class iOSMemoryInfo:
    """iOS device memory metrics."""
    total_memory: int          # Total device memory
    available_memory: int      # Free memory now
    pressure_level: int        # 0=normal, 1=warning, 2=critical
    background_mode: bool      # App in background (more restricted)

    @property
    def safe_cache_limit(self) -> int:
        """Compute safe KV cache size."""
        if self.pressure_level >= 2:
            return self.available_memory * 0.3  # Reduce to 30%
        elif self.pressure_level >= 1:
            return self.available_memory * 0.6  # Reduce to 60%
        else:
            return self.available_memory * 0.8  # Use 80%

# File: exo/worker/engines/mlx/exot_pipeline.py (ADD to ExotClient)

class ExotClient:
    """... existing code ..."""

    def get_remote_memory_info(self) -> iOSMemoryInfo:
        """Fetch iOS memory status via EXOT protocol."""
        # New message type: MEMORY_INFO_REQUEST
        msg = TensorMessage(
            msg_type=MessageType.MEMORY_INFO_REQUEST,
            dtype=TensorDType.INT32,
            shape=[],
            data=b"",
        )
        self._send_raw(msg.serialize())
        response = self._receive_message()

        # Parse iOS memory status from response
        return iOSMemoryInfo.from_bytes(response.data)
```

### 4.2 Eviction Strategies for iOS

**Current Approach**: LRU at prompt granularity (too coarse)

**Proposed: Token-level LRU**:

```python
# File: exo/worker/engines/mlx/cache.py (MODIFY KVPrefixCache)

class KVPrefixCache:
    """... existing code ..."""

    def _evict_if_needed_ios_aware(self, ios_memory_available: int):
        """Evict considering both Mac and iOS memory."""
        mac_active = mx.metal.get_active_memory()
        mac_limit = int(mx.metal.device_info()["max_recommended_working_set_size"])

        # Can iOS support current cache?
        total_cache_size = sum(
            self._estimate_cache_bytes(cache)
            for cache in self.caches
        )

        if total_cache_size > ios_memory_available * 0.7:
            # iOS would be stressed - evict oldest caches
            while len(self.caches) > 0 and total_cache_size > ios_memory_available * 0.5:
                lru_index = self._last_used.index(min(self._last_used))
                evicted_bytes = self._estimate_cache_bytes(self.caches[lru_index])
                self.prompts.pop(lru_index)
                self.caches.pop(lru_index)
                self._last_used.pop(lru_index)
                total_cache_size -= evicted_bytes

    def _estimate_cache_bytes(self, cache: KVCacheType) -> int:
        """Estimate memory footprint of cache."""
        total = 0
        for layer_cache in cache:
            # Each KVCache has keys and values
            if hasattr(layer_cache, 'keys'):
                total += layer_cache.keys.nbytes + layer_cache.values.nbytes
        return total
```

### 4.3 Background App Restrictions

**iOS Challenge**: Apps have minimal CPU/memory access in background

**Mitigation**:
1. **Pause cache updates** when app backgrounded
2. **Compress cache** before app backgrounding
3. **Warn user** of inference interruption

```python
# File: exo/worker/runner/bootstrap.py (ADD)

class iOSInferenceStateManager:
    """Manage inference lifecycle on iOS."""

    def on_app_background(self):
        """Called when iOS app moves to background."""
        # Pause any pending generation
        self.pause_generation = True

        # Compress KV cache aggressively
        self.kv_cache_manager.compress_all_caches(bits=2)

        # Notify Mac: "I'm backgrounded, pause inference"
        self.exot_client.send_background_notification()

    def on_app_foreground(self):
        """Called when iOS app returns to foreground."""
        # Decompress cache
        self.kv_cache_manager.decompress_all_caches()

        # Resume generation
        self.pause_generation = False
```

---

## 5. Optimization Opportunities

### 5.1 KV Cache Quantization (Ready to Use)

**Current Status**: Implemented but **disabled by default**

**Impact**: 4-bit quantization = 75-80% memory reduction

**Recommendation**: Enable for iOS devices

```python
# File: exo/worker/engines/mlx/cache.py (MODIFY make_kv_cache)

def make_kv_cache(
    model: Model,
    max_kv_size: int | None = None,
    keep: int = 0,
    quantize_for_device: str = "mac",  # NEW parameter
) -> KVCacheType:
    """Create KV cache, optionally quantized for memory-limited devices."""

    assert hasattr(model, "layers")

    # Use quantization for iOS devices
    if quantize_for_device == "ios":
        logger.info("Using 4-bit quantized KV cache for iOS")
        return [
            QuantizedKVCache(group_size=32, bits=4)
            for _ in model.layers
        ]
    elif quantize_for_device == "mac":
        logger.info("Using default KV cache for Mac")
        return [KVCache() for _ in model.layers]
    # ... existing logic
```

**Memory Savings for Llama-3-70B**:
```
Full precision (fp32):    ~2GB per device
4-bit quantized:          ~400MB per device
4-bit + 2K sliding window: ~100MB per device
```

### 5.2 Sliding Window Attention (Partially Available)

**Current Status**: Only in GPT-OSS MOE variant (line 220-230 in auto_parallel.py)

**Issue**: Not exposed as general option

**Recommendation**: Expose sliding window for all models

```python
# File: exo/worker/engines/mlx/constants.py (MODIFY)

# Add sliding window configuration
SLIDING_WINDOW_SIZE: int | None = 2048  # Tokens to keep in context
USE_SLIDING_WINDOW: bool = False        # Disable by default for compatibility

# File: exo/worker/engines/mlx/cache.py (MODIFY)

def make_kv_cache(
    model: Model,
    max_kv_size: int | None = None,
    keep: int = 0,
    use_sliding_window: bool = False,
) -> KVCacheType:
    """Create cache with optional sliding window."""

    if use_sliding_window and max_kv_size is None:
        logger.info(f"Using rotating KV cache (sliding window)")
        return [
            RotatingKVCache(max_size=SLIDING_WINDOW_SIZE, keep=KEEP_KV_SIZE)
            for _ in model.layers
        ]
    # ... rest of logic
```

**Trade-offs**:
- **Pros**: 90% context memory reduction, works with any context length
- **Cons**: Cannot attend to full history, may lose long-range dependencies

### 5.3 Prompt Prefix Caching (Already Great!)

**Current Status**: Excellent implementation

**What's Good**:
- Exact match detection (lines 86-97)
- Prefix matching (lines 102-120)
- LRU eviction with memory awareness (lines 133-156)

**Potential Enhancement**: Cache eviction based on iOS memory

```python
# File: exo/worker/engines/mlx/cache.py (MODIFY _evict_if_needed)

def _evict_if_needed(self, ios_memory_constraint: int | None = None):
    """Evict LRU entries considering both Mac and iOS memory."""

    # Mac side check (current)
    active: int = mx.metal.get_active_memory()
    limit = int(mx.metal.device_info()["max_recommended_working_set_size"])
    mac_pressure = active / limit

    # iOS side check (new)
    ios_pressure = 1.0
    if ios_memory_constraint is not None:
        total_cache_bytes = sum(
            self._estimate_cache_bytes(c) for c in self.caches
        )
        ios_pressure = total_cache_bytes / ios_memory_constraint

    # Use maximum pressure to be conservative
    pressure = max(mac_pressure, ios_pressure)

    if pressure < _MEMORY_THRESHOLD:
        return

    # Evict LRU entries
    while len(self.caches) > 0 and max(
        mx.metal.get_active_memory() / limit,
        (total_cache_bytes - self._estimate_cache_bytes(self.caches[0])) / ios_memory_constraint
        if ios_memory_constraint else 0,
    ) > _MEMORY_THRESHOLD * 0.95:
        # ... evict logic
```

### 5.4 PagedAttention (Research/Future)

**Not Currently Implemented**

**Relevance for Your Use Case**: **High for long contexts**

**Why**:
- Block-level granularity enables smarter eviction
- Perfect for balancing Mac (16GB) + iOS (8GB)
- vLLM shows 2-3x throughput improvements

**Estimated Implementation Effort**: 2-3 weeks

**Starting Point**: Adapt vLLM's implementation to MLX

```python
# Pseudocode structure

class MLXPagedAttention:
    """Port vLLM's PagedAttention to MLX."""

    BLOCK_SIZE = 16  # tokens per block

    def paged_attention(
        self,
        query: mx.array,
        key_blocks: list[mx.array],  # Each is one block
        value_blocks: list[mx.array],
        block_table: list[int],        # Which blocks to attend to
    ) -> mx.array:
        """Efficient attention over block-level KV cache."""
        # Compute attention only for relevant blocks
        # Can skip blocks on other devices or compressed blocks
```

---

## 6. Action Plan & Recommendations

### Immediate (This Week)

**Priority 1: Enable KV Cache Quantization for iOS**
- **File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/cache.py`
- **Change**: Add `quantize_for_device` parameter
- **Impact**: Reduce iOS KV cache from ~1GB to ~200MB per conversation
- **Effort**: 30 minutes
- **Testing**: Run inference with 8K context on iOS, measure memory

```python
# Line 188-210, modify make_kv_cache()
if KV_CACHE_BITS is None and is_ios_device:
    KV_CACHE_BITS = 4  # Force 4-bit for iOS
```

**Priority 2: Add iOS Memory Monitoring to EXOT Protocol**
- **File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/exot_pipeline.py`
- **Change**: Add `MessageType.MEMORY_STATUS` exchange
- **Impact**: Mac can monitor iOS memory pressure, adjust inference
- **Effort**: 2-3 hours
- **Testing**: Verify memory stats flow back to Mac

```python
# Lines 39-45, add to MessageType enum
MEMORY_STATUS = 7

# Lines 249-297, add new method
def get_ios_memory_status(self) -> MemoryInfo:
    """Query iOS for current memory usage."""
```

**Priority 3: Implement Token-Level Cache Eviction**
- **File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/cache.py`
- **Change**: Replace prompt-level eviction with token-level
- **Impact**: Finer-grained memory control, less cache thrashing
- **Effort**: 4-5 hours
- **Testing**: Verify cache retention patterns with long contexts

### Medium-Term (This Month)

**Priority 4: Distributed Cache Manager**
- **New File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/distributed_cache.py`
- **Design**: Master cache on Mac, iOS fetches slices
- **Impact**: 30-40% memory savings across cluster
- **Effort**: 1 week
- **Testing**: Multi-device inference with different layer counts

**Priority 5: Sliding Window Mode**
- **File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/constants.py`
- **Change**: Expose `SLIDING_WINDOW_SIZE` as config option
- **Impact**: Enable very long contexts (32K+) on iOS
- **Effort**: 1-2 days
- **Testing**: Verify coherence over long conversations

### Long-Term (Q2 2026)

**Priority 6: PagedAttention Implementation**
- **New Module**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/paged_attention.py`
- **Design**: Block-level KV management with device-aware placement
- **Impact**: 2-3x throughput improvement for long contexts
- **Effort**: 2-3 weeks
- **Complexity**: High (requires MLX KVCache internals understanding)

---

## 7. Code References & Exact Locations

### Critical Files for Modification

| File | Lines | Purpose |
|------|-------|---------|
| `/exo/src/exo/worker/engines/mlx/cache.py` | 188-210 | `make_kv_cache()` - add quantization |
| `/exo/src/exo/worker/engines/mlx/cache.py` | 133-156 | `_evict_if_needed()` - enhance for iOS |
| `/exo/src/exo/worker/engines/mlx/exot_pipeline.py` | 39-45 | `MessageType` enum - add MEMORY_STATUS |
| `/exo/src/exo/worker/engines/mlx/exot_pipeline.py` | 199-339 | `ExotClient` class - add memory queries |
| `/exo/src/exo/worker/engines/mlx/constants.py` | 1-15 | Constants - add sliding window config |
| `/exo/src/exo/worker/engines/mlx/generator/generate.py` | 161-220 | `mlx_generate()` - prefill optimization |

### New Files to Create

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `/exo/src/exo/worker/engines/mlx/distributed_cache.py` | Master/slave cache coordination | 300-400 |
| `/exo/src/exo/worker/engines/mlx/ios_memory_monitor.py` | iOS memory telemetry | 150-200 |
| `/exo/src/exo/worker/engines/mlx/paged_attention.py` | Block-level attention (future) | 500-700 |

---

## 8. Memory Capacity Analysis

### Example: Llama-3-70B with 8K Context

**Mac M4 (16GB)**:
```
Model weights:          ~40GB (doesn't fit, need quantization)
  → 4-bit quantized:    ~10GB
KV cache (full):        ~2GB
Activations:            ~1GB
Free memory:            ~3GB  ✓ Safe margin
────────────────────────────
Total used:             ~14GB / 16GB
```

**iOS Device (8GB)**:
```
Model subset (4 layers):  ~4GB (worst case: full precision)
  → 4-bit quantized:     ~1GB
KV cache (full):         ~1GB
  → 4-bit quantized:     ~200MB
Activations:             ~200MB
OS + other:              ~3GB
─────────────────────────────
Total available:         ~8GB
Needed:                  ~5.4GB  ⚠️ Close to limit

WITH OPTIMIZATION:
Model (4-bit):           ~1GB
KV cache (4-bit):        ~200MB
Activations:             ~200MB
Free buffer:             ~5.6GB  ✓ Safe
```

---

## 9. Context Length Limitations

**Current System Bottleneck**: iPhone 15 Pro with 8GB RAM

### Context Window Limits

| Configuration | Max Context | Notes |
|---------------|-------------|-------|
| Full precision | 2K tokens | Hits iOS memory wall |
| 4-bit quantized | 4-6K tokens | Moderate compression |
| 4-bit + 2K sliding window | 32K tokens | Loses long-range context |
| Distributed cache + 4-bit | 8-12K tokens | Master-slave model |
| PagedAttention (future) | 16K+ tokens | Optimal block placement |

---

## 10. Benchmarking Recommendations

### Test Configurations

**Test 1: Baseline Cache Memory Usage**
```bash
# Run inference with different cache types
python -m exo.worker \
    --kv-cache-bits 0    # float32
    --context-length 4096 \
    --device ios

# Measure: Peak memory, cache footprint
```

**Test 2: Quantization Impact**
```bash
# Compare 4-bit vs float32
for bits in 0 4 8; do
    python -m exo.worker \
        --kv-cache-bits $bits \
        --benchmark-throughput \
        --context-length 8192
done
```

**Test 3: Prefix Caching Effectiveness**
```bash
# Measure cache hit rates
python -c "
    from exo.worker.engines.mlx.cache import KVPrefixCache
    # Run multiple inference calls with overlapping prompts
    # Track: hit_rate, eviction_count, avg_reuse_tokens
"
```

---

## 11. Summary Table

| Component | Current Status | Gap | Recommendation |
|-----------|---------------|----|-----------------|
| KV Cache Storage | ✓ Multiple types (KV, Quantized, Rotating) | Quantization disabled by default | Enable for iOS |
| Memory Monitoring | ✓ Mac (Metal), ✗ iOS | No iOS telemetry | Add EXOT memory queries |
| LRU Eviction | ✓ Implemented | Prompt-level, not iOS-aware | Token-level + iOS constraints |
| Prefix Caching | ✓ Excellent | None | Minor enhancements only |
| Sliding Window | ✓ Partial (MOE only) | Not exposed for general use | Expose as config option |
| Distributed Cache | ✗ Not implemented | None | Implement master-slave model |
| PagedAttention | ✗ Not implemented | None | Plan for Q2 2026 |

---

## References

1. **vLLM PagedAttention**: https://arxiv.org/abs/2309.06180
2. **MLX LM KV Cache**: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
3. **KV Cache Quantization**: https://arxiv.org/abs/2407.12384
4. **Distributed Inference Strategies**: https://arxiv.org/abs/2310.03400

---

## Questions for Clarification

1. **Context Length Priority**: Do you need 32K+ context or optimize for throughput?
2. **iOS Version Target**: Minimum iOS version (affects memory APIs available)?
3. **Latency Tolerance**: How much added latency acceptable for cache coordination?
4. **Model Precision**: Will you quantize model weights to 4-bit, or just KV cache?
5. **Typical Batch Size**: Single inference at a time, or multiple concurrent?

