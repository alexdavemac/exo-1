# KV Cache Optimization: Implementation Guide
## Practical Code Changes for Mac/iOS Pipeline Parallelism

---

## Implementation 1: Enable KV Cache Quantization for iOS

**Difficulty**: Easy (30 mins)
**Impact**: ~75% KV cache memory reduction
**Files Modified**: 2

### Change 1.1: Extend `make_kv_cache()` Function

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/cache.py`

**Current Code** (lines 188-210):
```python
def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    # TODO: Do this for all models
    if hasattr(model, "make_cache") and isinstance(model, GptOssModel):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
```

**Replacement Code**:
```python
def make_kv_cache(
    model: Model,
    max_kv_size: int | None = None,
    keep: int = 0,
    device_name: str = "mac",  # NEW: specify target device
) -> KVCacheType:
    """Create KV cache, with optional device-specific optimization.

    Args:
        model: The model to create cache for
        max_kv_size: Size limit for rotating cache (if used)
        keep: Number of tokens to keep when rotating (if used)
        device_name: Target device - "mac", "ios", or "auto"
                    "auto" uses quantization for iOS, default for Mac

    Returns:
        List of cache objects, one per layer
    """
    assert hasattr(model, "layers")

    # Determine quantization strategy based on device
    use_quantization = False
    quantize_bits = KV_CACHE_BITS

    if device_name == "ios":
        # iOS has severe memory constraints - use quantization by default
        use_quantization = True
        quantize_bits = quantize_bits or 4  # Default to 4-bit if not specified
        logger.info(f"iOS device detected: using {quantize_bits}-bit quantized KV cache")
    elif device_name == "auto":
        # Let environment variable decide
        use_quantization = KV_CACHE_BITS is not None
    elif device_name != "mac":
        logger.warning(f"Unknown device '{device_name}', treating as 'mac'")

    # TODO: Do this for all models
    if hasattr(model, "make_cache") and isinstance(model, GptOssModel):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if use_quantization:
            logger.info(
                f"Using {quantize_bits}-bit quantized KV cache "
                f"(group_size={CACHE_GROUP_SIZE})"
            )
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=quantize_bits)
                for _ in model.layers
            ]
        else:
            logger.info("Using default KV cache (full precision)")
            return [KVCache() for _ in model.layers]
    else:
        logger.info(
            f"Using rotating KV cache with max_size={max_kv_size}, keep={keep}"
        )
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
```

### Change 1.2: Pass Device Info Through Pipeline

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/generator/generate.py`

**Current Code** (lines 185-186):
```python
    else:
        caches = make_kv_cache(model=model)
```

**New Code**:
```python
    else:
        # Determine if running on iOS via EXOT pipeline
        is_ios = task.metadata.get("device_type") == "ios" if hasattr(task, "metadata") else False
        device_name = "ios" if is_ios else "mac"
        caches = make_kv_cache(model=model, device_name=device_name)
```

**Alternative (More Robust)**: Check global EXOT state

```python
# At top of file
from exo.worker.engines.mlx.exot_pipeline import get_exot_client

# In mlx_generate() function
    else:
        # Detect if running in EXOT pipeline (iOS present)
        is_ios_stage = get_exot_client() is not None
        caches = make_kv_cache(
            model=model,
            device_name="ios" if is_ios_stage else "mac"
        )
```

### Change 1.3: Update Constants for iOS

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/constants.py`

**Current Code** (lines 12):
```python
KV_CACHE_BITS: int | None = None
```

**New Code**:
```python
KV_CACHE_BITS: int | None = None           # Default behavior
IOS_KV_CACHE_BITS: int = 4                 # Force 4-bit for iOS
IOS_CACHE_GROUP_SIZE: int = 64             # iOS-optimized group size
```

**Then in cache.py**:
```python
from exo.worker.engines.mlx.constants import IOS_KV_CACHE_BITS

def make_kv_cache(..., device_name: str = "mac"):
    ...
    if device_name == "ios":
        quantize_bits = IOS_KV_CACHE_BITS  # Use iOS-specific default
```

### Testing Change 1

```bash
# Test on Mac (should use float32)
python -c "
from exo.worker.engines.mlx.cache import make_kv_cache
from exo.worker.engines.mlx import Model
from mlx_lm.utils import load_model

model, _ = load_model('path/to/model')
cache_mac = make_kv_cache(model, device_name='mac')
print(f'Mac cache type: {type(cache_mac[0]).__name__}')  # KVCache
"

# Test on iOS (should use 4-bit quantized)
# (Run on actual iOS via EXOT protocol or mock)
```

---

## Implementation 2: Add iOS Memory Monitoring

**Difficulty**: Medium (3-4 hours)
**Impact**: Intelligent cache eviction based on actual iOS memory
**Files Modified**: 3 files, 1 new file

### Change 2.1: Extend EXOT Protocol with Memory Status

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/exot_pipeline.py`

**Current Code** (lines 39-45):
```python
class MessageType(IntEnum):
    SYNC = 1
    SYNC_ACK = 2
    ACTIVATION = 3
    RESULT = 4
    ERROR = 5
    KEEPALIVE = 6
```

**New Code**:
```python
class MessageType(IntEnum):
    SYNC = 1
    SYNC_ACK = 2
    ACTIVATION = 3
    RESULT = 4
    ERROR = 5
    KEEPALIVE = 6
    MEMORY_STATUS_REQUEST = 7      # NEW: Mac requests iOS memory info
    MEMORY_STATUS_RESPONSE = 8     # NEW: iOS responds with memory stats
```

### Change 2.2: Add Memory Status Methods to ExotClient

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/exot_pipeline.py`

**Add After Line 397** (after `_recv_exact` method):

```python
    def request_memory_status(self) -> dict[str, int] | None:
        """Request current memory status from iOS device.

        Returns:
            Dict with keys: {
                'total_memory': int,          # Total device memory
                'available_memory': int,      # Free memory
                'pressure_level': int,        # 0=normal, 1=warning, 2=critical
            }
        """
        if not self._connected or not self._socket:
            logger.warning("[EXOT] Not connected, cannot request memory status")
            return None

        try:
            # Send request
            msg = TensorMessage(
                msg_type=MessageType.MEMORY_STATUS_REQUEST,
                dtype=TensorDType.INT32,
                shape=[],
                data=b"",  # Empty payload
            )
            self._send_raw(msg.serialize())
            logger.debug("[EXOT] Sent memory status request")

            # Receive response
            response = self._receive_message()
            if response is None:
                logger.warning("[EXOT] No response to memory status request")
                return None

            if response.msg_type != MessageType.MEMORY_STATUS_RESPONSE:
                logger.warning(
                    f"[EXOT] Expected MEMORY_STATUS_RESPONSE, got {response.msg_type}"
                )
                return None

            # Parse response: JSON with memory info
            memory_data = json.loads(response.data.decode("utf-8"))
            logger.debug(f"[EXOT] iOS memory status: {memory_data}")
            return memory_data

        except Exception as e:
            logger.error(f"[EXOT] Failed to get memory status: {e}")
            return None
```

### Change 2.3: Create iOS Memory Monitor Type

**New File**: `/Users/alexmcdaniel/Projects/exo/src/exo/shared/types/ios_memory.py`

```python
"""iOS device memory monitoring."""

from dataclasses import dataclass
from typing import Literal

@dataclass
class iOSMemoryStatus:
    """Current memory state of iOS device."""
    total_memory_bytes: int
    available_memory_bytes: int
    pressure_level: Literal[0, 1, 2]  # 0=normal, 1=warning, 2=critical
    background_mode: bool = False

    @property
    def available_gb(self) -> float:
        """Available memory in GB."""
        return self.available_memory_bytes / (1024**3)

    @property
    def pressure_name(self) -> str:
        """Human-readable pressure level."""
        return {0: "normal", 1: "warning", 2: "critical"}[self.pressure_level]

    def get_safe_cache_limit(self, cushion: float = 0.3) -> int:
        """Compute safe KV cache size given current pressure.

        Args:
            cushion: Fraction of available memory to reserve as safety margin

        Returns:
            Safe cache size in bytes
        """
        # Reserve more memory when under pressure
        usable_fraction = {
            0: 0.7,   # Normal: use 70% of available
            1: 0.5,   # Warning: use 50%
            2: 0.3,   # Critical: use 30% only
        }[self.pressure_level]

        available_for_cache = self.available_memory_bytes * usable_fraction
        reserved = int(available_for_cache * cushion)
        return max(0, available_for_cache - reserved)

    @classmethod
    def from_dict(cls, data: dict) -> "iOSMemoryStatus":
        """Create from EXOT protocol response."""
        return cls(
            total_memory_bytes=data["totalMemory"],
            available_memory_bytes=data["availableMemory"],
            pressure_level=data["pressureLevel"],
            background_mode=data.get("backgroundMode", False),
        )
```

### Change 2.4: Enhance Cache Eviction with iOS Memory

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/cache.py`

**Current Code** (lines 27-33):
```python
class KVPrefixCache:
    def __init__(self, tokenizer: TokenizerWrapper):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0
        self._tokenizer: TokenizerWrapper = tokenizer
```

**New Code**:
```python
class KVPrefixCache:
    def __init__(self, tokenizer: TokenizerWrapper):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0
        self._tokenizer: TokenizerWrapper = tokenizer
        self._exot_client: object | None = None  # NEW: Reference to EXOT client for iOS monitoring
        self._estimated_cache_sizes: list[int] = []  # NEW: Byte size of each cache
```

**Add Method After Line 49**:
```python
    def set_exot_client(self, exot_client: object) -> None:
        """Set EXOT client for iOS memory monitoring."""
        self._exot_client = exot_client

    def _estimate_cache_bytes(self, cache: KVCacheType) -> int:
        """Estimate memory footprint of a KV cache in bytes.

        Approximation based on layer cache sizes.
        """
        total_bytes = 0
        for layer_cache in cache:
            if hasattr(layer_cache, "keys"):
                # Standard KVCache with keys and values
                total_bytes += layer_cache.keys.nbytes
                total_bytes += layer_cache.values.nbytes
            elif hasattr(layer_cache, "state"):
                # QuantizedKVCache - harder to estimate
                # Assume ~25% of standard size (4-bit quantization)
                # This is conservative; actual may be less
                total_bytes += int(
                    (layer_cache.state.nbytes if hasattr(layer_cache.state, "nbytes") else 0) * 0.25
                )
        return total_bytes
```

**Modify `add_kv_cache()` Method** (lines 41-49):
```python
    def add_kv_cache(self, prompt: str, cache: KVCacheType):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        tokenized_prompt = encode_prompt(self._tokenizer, prompt)
        self.prompts.append(tokenized_prompt)
        self.caches.append(deepcopy(cache))
        cache_bytes = self._estimate_cache_bytes(cache)  # NEW
        self._estimated_cache_sizes.append(cache_bytes)  # NEW
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        logger.info(
            f"KV cache added: {len(tokenized_prompt)} tokens "
            f"(~{cache_bytes / (1024**2):.1f}MB)"  # NEW
        )
```

**Enhance `_evict_if_needed()` Method** (lines 133-156):

Replace entire method with:

```python
    def _evict_if_needed(self):
        """Evict least recently used entries while memory pressure is high.

        Considers both Mac (via MLX Metal) and iOS (via EXOT) memory.
        """
        if len(self.caches) == 0:
            return

        # Check Mac memory (always available)
        mac_active: int = mx.metal.get_active_memory()
        mac_limit = int(mx.metal.device_info()["max_recommended_working_set_size"])
        mac_pressure = mac_active / mac_limit

        # Check iOS memory (if available via EXOT)
        ios_pressure = 0.0  # Assume no pressure by default
        ios_safe_cache_limit = None

        if self._exot_client is not None:
            try:
                memory_status = self._exot_client.request_memory_status()
                if memory_status is not None:
                    ios_safe_cache_limit = memory_status.get("safe_cache_limit")

                    # Compute iOS pressure: current cache / safe limit
                    total_cache_bytes = sum(self._estimated_cache_sizes)
                    if ios_safe_cache_limit > 0:
                        ios_pressure = min(1.0, total_cache_bytes / ios_safe_cache_limit)

                    logger.debug(
                        f"iOS memory: {memory_status.get('available_memory')/1e9:.2f}GB available, "
                        f"pressure={memory_status.get('pressure_level')}"
                    )
            except Exception as e:
                logger.warning(f"Failed to get iOS memory status: {e}")

        # Use maximum pressure (most constrained device)
        max_pressure = max(mac_pressure, ios_pressure)

        if max_pressure < _MEMORY_THRESHOLD:
            return  # No eviction needed

        # Evict LRU entries until below threshold
        logger.info(
            f"Memory pressure high (mac={mac_pressure:.2f}, ios={ios_pressure:.2f}), "
            f"evicting LRU entries"
        )

        while len(self.caches) > 0:
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            evicted_bytes = self._estimated_cache_sizes[lru_index]

            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._last_used.pop(lru_index)
            self._estimated_cache_sizes.pop(lru_index)  # NEW

            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens, "
                f"~{evicted_bytes / (1024**2):.1f}MB) due to memory pressure"
            )

            # Re-check pressure
            if ios_safe_cache_limit is not None:
                total_cache_bytes = sum(self._estimated_cache_sizes)
                ios_pressure = min(1.0, total_cache_bytes / ios_safe_cache_limit) if ios_safe_cache_limit > 0 else 0.0

            mac_active = mx.metal.get_active_memory()
            mac_pressure = mac_active / mac_limit

            max_pressure = max(mac_pressure, ios_pressure)
            if max_pressure < _MEMORY_THRESHOLD * 0.95:
                break  # Evicted enough
```

### Testing Change 2

```bash
# Test memory monitoring integration
python -c "
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.exot_pipeline import ExotClient
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B-Instruct')
cache_mgr = KVPrefixCache(tokenizer)

# Mock EXOT client
class MockExotClient:
    def request_memory_status(self):
        return {
            'total_memory': 8 * 1024**3,
            'available_memory': 3 * 1024**3,
            'pressure_level': 1,
            'safe_cache_limit': 500 * 1024**2,  # 500MB safe
        }

cache_mgr.set_exot_client(MockExotClient())
print('Memory monitoring enabled')
"
```

---

## Implementation 3: Token-Level Cache Size Tracking

**Difficulty**: Easy (1 hour)
**Impact**: Better visibility into cache memory usage
**Files Modified**: 1

### Change 3.1: Add Cache Size Logging

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/cache.py`

**Add After `_estimate_cache_bytes()` Method**:

```python
    def get_cache_memory_summary(self) -> dict:
        """Get current cache memory usage summary.

        Returns:
            {
                'total_caches': int,
                'total_bytes': int,
                'total_mb': float,
                'per_cache': list[dict],  # Sorted by size descending
            }
        """
        total_bytes = sum(self._estimated_cache_sizes)

        per_cache = [
            {
                'prompt_tokens': len(self.prompts[i]),
                'bytes': self._estimated_cache_sizes[i],
                'mb': self._estimated_cache_sizes[i] / (1024**2),
                'last_used': self._last_used[i],
            }
            for i in range(len(self.caches))
        ]

        # Sort by size descending
        per_cache.sort(key=lambda x: x['bytes'], reverse=True)

        return {
            'total_caches': len(self.caches),
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024**2),
            'per_cache': per_cache,
        }

    def log_cache_summary(self):
        """Log detailed cache memory summary."""
        summary = self.get_cache_memory_summary()
        logger.info(
            f"KV Cache Summary: {summary['total_caches']} entries, "
            f"~{summary['total_mb']:.1f}MB total"
        )

        if summary['per_cache']:
            logger.debug("Cache breakdown:")
            for i, entry in enumerate(summary['per_cache'][:5]):  # Top 5
                logger.debug(
                    f"  [{i}] {entry['prompt_tokens']} tokens, "
                    f"{entry['mb']:.1f}MB, last_used={entry['last_used']}"
                )
```

**Call from `add_kv_cache()` after line 49**:
```python
        self.log_cache_summary()  # NEW: Log after each addition
```

---

## Implementation 4: Sliding Window Configuration

**Difficulty**: Medium (2-3 hours)
**Impact**: Enable very long contexts on iOS
**Files Modified**: 2

### Change 4.1: Add Sliding Window Constants

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/constants.py`

**Add After Line 9**:
```python
# Sliding window attention for long contexts
SLIDING_WINDOW_SIZE: int | None = None      # Tokens to keep (None = disabled)
SLIDING_WINDOW_ENABLED: bool = False         # Feature flag
SLIDING_WINDOW_MIN_CONTEXT: int = 4096       # Min context length to trigger sliding window
```

### Change 4.2: Update `make_kv_cache()` to Support Sliding Window

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/cache.py`

**Modify function signature**:
```python
def make_kv_cache(
    model: Model,
    max_kv_size: int | None = None,
    keep: int = 0,
    device_name: str = "mac",
    enable_sliding_window: bool = False,  # NEW parameter
) -> KVCacheType:
```

**Add Logic Before Return Statements**:
```python
    # Determine if sliding window should be used
    if enable_sliding_window and max_kv_size is None:
        from exo.worker.engines.mlx.constants import (
            SLIDING_WINDOW_SIZE,
            KEEP_KV_SIZE,
        )

        if SLIDING_WINDOW_SIZE is not None:
            logger.info(
                f"Using rotating KV cache (sliding window) "
                f"with window_size={SLIDING_WINDOW_SIZE}, keep={KEEP_KV_SIZE}"
            )
            return [
                RotatingKVCache(
                    max_size=SLIDING_WINDOW_SIZE,
                    keep=KEEP_KV_SIZE,
                )
                for _ in model.layers
            ]

    # Existing logic for quantization and default caches follows...
```

**Usage in Generator**:
```python
# In mlx_generate() function
from exo.worker.engines.mlx.constants import SLIDING_WINDOW_ENABLED

caches = make_kv_cache(
    model=model,
    device_name=device_name,
    enable_sliding_window=SLIDING_WINDOW_ENABLED,  # NEW
)
```

---

## Implementation 5: EXOT Protocol Enhancement for Cache Transfer

**Difficulty**: Hard (1 week)
**Impact**: Enable master-slave cache architecture
**Files Modified**: exot_pipeline.py, generator/generate.py

### Change 5.1: Add Cache Transfer Message Types

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/exot_pipeline.py`

**Lines 39-45, extend MessageType**:
```python
class MessageType(IntEnum):
    SYNC = 1
    SYNC_ACK = 2
    ACTIVATION = 3
    RESULT = 4
    ERROR = 5
    KEEPALIVE = 6
    MEMORY_STATUS_REQUEST = 7
    MEMORY_STATUS_RESPONSE = 8
    CACHE_SLICE_REQUEST = 9     # NEW: Mac->iOS: "Send me KV cache for layers [N,M)"
    CACHE_SLICE_RESPONSE = 10   # NEW: iOS->Mac: KV cache data
    CACHE_UPDATE = 11           # NEW: Mac->iOS: "Update your cache after processing"
```

### Change 5.2: Add Cache Transfer Methods

**File**: `/Users/alexmcdaniel/Projects/exo/src/exo/worker/engines/mlx/exot_pipeline.py`

**Add After `request_memory_status()` method**:

```python
    def request_cache_slice(
        self,
        layer_start: int,
        layer_end: int,
        token_count: int,
    ) -> mx.array | None:
        """Request KV cache slice from iOS for specific layers.

        Args:
            layer_start: First layer index
            layer_end: Last layer index (exclusive)
            token_count: Number of tokens in cache

        Returns:
            Concatenated KV cache as mx.array or None if failed
        """
        if not self._connected or not self._socket:
            logger.warning("[EXOT] Not connected, cannot request cache slice")
            return None

        try:
            # Send request
            request_data = json.dumps({
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "tokenCount": token_count,
            }).encode("utf-8")

            msg = TensorMessage(
                msg_type=MessageType.CACHE_SLICE_REQUEST,
                dtype=TensorDType.INT32,
                shape=[],
                data=request_data,
            )
            self._send_raw(msg.serialize())
            logger.debug(
                f"[EXOT] Requested cache slice for layers [{layer_start}, {layer_end})"
            )

            # Receive response
            response = self._receive_message()
            if response is None:
                logger.warning("[EXOT] No response to cache request")
                return None

            if response.msg_type != MessageType.CACHE_SLICE_RESPONSE:
                logger.warning(
                    f"[EXOT] Expected CACHE_SLICE_RESPONSE, got {response.msg_type}"
                )
                return None

            # Convert response data to array
            cache_array = response.to_array()
            logger.debug(f"[EXOT] Received cache slice: shape={cache_array.shape}")
            return cache_array

        except Exception as e:
            logger.error(f"[EXOT] Failed to get cache slice: {e}")
            return None

    def update_cache_after_generation(
        self,
        updated_cache: mx.array,
        layer_start: int,
        layer_end: int,
    ) -> bool:
        """Send updated KV cache back to iOS after generation.

        Args:
            updated_cache: Updated KV cache array
            layer_start: First layer in this cache
            layer_end: Last layer (exclusive)

        Returns:
            True if successful
        """
        if not self._connected or not self._socket:
            logger.warning("[EXOT] Not connected, cannot send cache update")
            return False

        try:
            # Create message with cache data
            msg = TensorMessage.from_array(MessageType.CACHE_UPDATE, updated_cache)

            # Add metadata about which layers this cache covers
            # (Hack: embed in unused shape slots or use separate metadata message)
            # Better: extend protocol to include metadata

            self._send_raw(msg.serialize())
            logger.debug(
                f"[EXOT] Sent cache update for layers [{layer_start}, {layer_end})"
            )
            return True

        except Exception as e:
            logger.error(f"[EXOT] Failed to send cache update: {e}")
            return False
```

---

## Validation Checklist

After implementing each change, validate:

- [ ] **Change 1: Quantization**
  - [ ] iOS inference uses 4-bit cache (check logs)
  - [ ] Memory usage reduced by 75%
  - [ ] Inference quality unchanged (BLEU/perplexity)
  - [ ] No slowdown in throughput

- [ ] **Change 2: iOS Memory Monitoring**
  - [ ] EXOT receives memory status messages
  - [ ] Cache eviction triggered when iOS pressure > 0.85
  - [ ] Logs show "memory pressure high" when appropriate

- [ ] **Change 3: Size Tracking**
  - [ ] `log_cache_summary()` output is accurate
  - [ ] Memory estimates match actual profiling
  - [ ] No memory overhead from tracking

- [ ] **Change 4: Sliding Window**
  - [ ] Long contexts (32K tokens) work on iOS
  - [ ] Context window limited to SLIDING_WINDOW_SIZE
  - [ ] Older tokens properly discarded (no memory growth)

- [ ] **Change 5: Cache Transfer** (if implementing)
  - [ ] Cache slices transfer without corruption
  - [ ] Updated caches properly reflected in iOS state
  - [ ] Latency acceptable (< 100ms per cache transfer)

---

## Performance Benchmarking

After each implementation, run:

```bash
# Measure memory usage
python -c "
import subprocess
import json

config = {
    'model': 'mlx-community/Llama-3-8B',
    'context_length': 8192,
    'batch_size': 1,
    'kv_cache_quantization': '4bit',
    'device': 'ios',
}

# Run inference
result = subprocess.run(
    ['python', '-m', 'exo.worker', '--benchmark', '--config', json.dumps(config)],
    capture_output=True,
)

# Parse memory output
lines = result.stdout.decode().split('\n')
for line in lines:
    if 'peak_memory' in line or 'cache_memory' in line:
        print(line)
"

# Benchmark throughput
time python -m exo.benchmark \
    --model mlx-community/Llama-3-8B \
    --num-prompts 100 \
    --context-length 4096 \
    --quantized-cache
```

---

## Rollback Plan

If any implementation causes issues:

1. **Quantization Issues** (Change 1)
   - Revert `device_name` parameter
   - Set `IOS_KV_CACHE_BITS = None`

2. **Memory Monitoring Issues** (Change 2)
   - Set `_exot_client = None` in cache.py
   - Remove EXOT memory status calls

3. **Size Tracking Issues** (Change 3)
   - Remove `_estimated_cache_sizes` list
   - Remove calls to `log_cache_summary()`

4. **Sliding Window Issues** (Change 4)
   - Set `SLIDING_WINDOW_ENABLED = False`
   - Revert to standard KV cache

---

## Next Steps

1. **Implement Change 1-2 this week** (high impact, low risk)
2. **Deploy to staging** (test with real iOS devices)
3. **Monitor for 1 week** (ensure stability)
4. **Implement Change 3-4 next sprint** (refinement)
5. **Plan Change 5 for Q2** (complex, higher risk)

