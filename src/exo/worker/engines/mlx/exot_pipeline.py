"""
EXOT Pipeline - TCP tensor exchange protocol for Mac <-> iOS pipeline parallelism.

This module implements the Mac side of the EXOT (EXO Tensor) protocol, enabling
distributed inference where the Mac sends activations to iOS devices running
the TensorServer, and receives processed results back.

Protocol Wire Format:
- Magic: 0x45584F54 ("EXOT") - 4 bytes, big-endian
- Version: 1 - 4 bytes, big-endian
- Message Type: 4 bytes, big-endian
- DType: 4 bytes, big-endian
- NDim: 4 bytes, big-endian
- Shape: NDim * 4 bytes, big-endian
- DataLen: 8 bytes, big-endian
- Data: DataLen bytes
"""

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

# Protocol constants - must match TensorProtocol.swift
EXOT_MAGIC = 0x45584F54  # "EXOT" in ASCII
EXOT_VERSION = 1
EXOT_DEFAULT_PORT = 52414
EXOT_HEADER_MIN_SIZE = 24


class MessageType(IntEnum):
    SYNC = 1
    SYNC_ACK = 2
    ACTIVATION = 3
    RESULT = 4
    ERROR = 5
    KEEPALIVE = 6


class TensorDType(IntEnum):
    FLOAT16 = 0
    FLOAT32 = 1
    BFLOAT16 = 2
    INT32 = 3
    INT64 = 4

    @classmethod
    def from_mx_dtype(cls, dtype: mx.Dtype) -> "TensorDType":
        mapping = {
            mx.float16: cls.FLOAT16,
            mx.float32: cls.FLOAT32,
            mx.bfloat16: cls.BFLOAT16,
            mx.int32: cls.INT32,
            mx.int64: cls.INT64,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    def to_mx_dtype(self) -> mx.Dtype:
        mapping = {
            self.FLOAT16: mx.float16,
            self.FLOAT32: mx.float32,
            self.BFLOAT16: mx.bfloat16,
            self.INT32: mx.int32,
            self.INT64: mx.int64,
        }
        return mapping[self]


@dataclass
class SyncMessage:
    """Sync message sent at connection start."""
    session_id: str
    model_id: str
    start_layer: int
    end_layer: int
    world_size: int
    rank: int
    total_layers: int
    is_first_stage: bool
    is_last_stage: bool
    sequence_id: str | None = None


@dataclass
class TensorMessage:
    """A tensor message in EXOT protocol format."""
    msg_type: MessageType
    dtype: TensorDType
    shape: list[int]
    data: bytes

    @classmethod
    def from_array(cls, msg_type: MessageType, array: mx.array) -> "TensorMessage":
        """Create a TensorMessage from an MLX array."""
        dtype = TensorDType.from_mx_dtype(array.dtype)
        shape = list(array.shape)
        # Ensure array is contiguous and get bytes
        array = mx.array(array)  # Make contiguous copy
        mx.eval(array)
        data = bytes(memoryview(array))
        return cls(msg_type=msg_type, dtype=dtype, shape=shape, data=data)

    def to_array(self) -> mx.array:
        """Convert to MLX array."""
        mx_dtype = self.dtype.to_mx_dtype()
        return mx.array(memoryview(self.data)).reshape(self.shape).astype(mx_dtype)

    def serialize(self) -> bytes:
        """Serialize to wire format."""
        result = bytearray()

        # Magic (4 bytes, big-endian)
        result.extend(struct.pack(">I", EXOT_MAGIC))

        # Version (4 bytes, big-endian)
        result.extend(struct.pack(">I", EXOT_VERSION))

        # Message type (4 bytes, big-endian)
        result.extend(struct.pack(">I", self.msg_type))

        # DType (4 bytes, big-endian)
        result.extend(struct.pack(">I", self.dtype))

        # NDim (4 bytes, big-endian)
        result.extend(struct.pack(">I", len(self.shape)))

        # Shape (NDim * 4 bytes, big-endian)
        for dim in self.shape:
            result.extend(struct.pack(">I", dim))

        # Data length (8 bytes, big-endian)
        result.extend(struct.pack(">Q", len(self.data)))

        # Data
        result.extend(self.data)

        return bytes(result)

    @classmethod
    def deserialize(cls, data: bytes) -> "TensorMessage":
        """Deserialize from wire format."""
        if len(data) < EXOT_HEADER_MIN_SIZE:
            raise ValueError("Message too short")

        offset = 0

        # Magic
        magic = struct.unpack_from(">I", data, offset)[0]
        if magic != EXOT_MAGIC:
            raise ValueError(f"Invalid magic: {magic:#x}")
        offset += 4

        # Version
        version = struct.unpack_from(">I", data, offset)[0]
        if version != EXOT_VERSION:
            raise ValueError(f"Unsupported version: {version}")
        offset += 4

        # Message type
        msg_type = MessageType(struct.unpack_from(">I", data, offset)[0])
        offset += 4

        # DType
        dtype = TensorDType(struct.unpack_from(">I", data, offset)[0])
        offset += 4

        # NDim
        ndim = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        # Shape
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from(">I", data, offset)[0])
            offset += 4

        # Data length
        data_len = struct.unpack_from(">Q", data, offset)[0]
        offset += 8

        # Data
        if len(data) < offset + data_len:
            raise ValueError("Incomplete data")
        tensor_data = data[offset:offset + data_len]

        return cls(msg_type=msg_type, dtype=dtype, shape=shape, data=tensor_data)


class ExotClient:
    """
    TCP client for EXOT protocol communication with iOS TensorServer.

    Handles connection management, sync handshake, and tensor exchange.
    """

    def __init__(
        self,
        host: str,
        port: int = EXOT_DEFAULT_PORT,
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._sync_info: SyncMessage | None = None

    def connect(self) -> bool:
        """Establish connection to iOS TensorServer."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.settimeout(self.timeout)

            logger.info(f"[EXOT] Connecting to {self.host}:{self.port}")
            self._socket.connect((self.host, self.port))
            self._connected = True
            logger.info(f"[EXOT] Connected to iOS TensorServer")
            return True

        except Exception as e:
            logger.error(f"[EXOT] Connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Close the connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False
        logger.info("[EXOT] Disconnected")

    def sync(self, sync_msg: SyncMessage) -> bool:
        """
        Perform sync handshake with iOS.

        Sends sync message with session/model info and waits for ack.
        """
        if not self._connected or not self._socket:
            return False

        try:
            # Encode sync data as JSON
            sync_data = json.dumps({
                "sessionId": sync_msg.session_id,
                "modelId": sync_msg.model_id,
                "startLayer": sync_msg.start_layer,
                "endLayer": sync_msg.end_layer,
                "worldSize": sync_msg.world_size,
                "rank": sync_msg.rank,
                "totalLayers": sync_msg.total_layers,
                "isFirstStage": sync_msg.is_first_stage,
                "isLastStage": sync_msg.is_last_stage,
                "sequenceId": sync_msg.sequence_id,
            }).encode("utf-8")

            # Create sync message
            msg = TensorMessage(
                msg_type=MessageType.SYNC,
                dtype=TensorDType.INT32,
                shape=[],
                data=sync_data,
            )

            # Send sync
            self._send_raw(msg.serialize())
            logger.info(f"[EXOT] Sent sync: model={sync_msg.model_id}, layers=[{sync_msg.start_layer}, {sync_msg.end_layer})")

            # Wait for sync ack
            response = self._receive_message()
            if response and response.msg_type == MessageType.SYNC_ACK:
                self._sync_info = sync_msg
                logger.info("[EXOT] Received sync ack - ready for tensor exchange")
                return True
            else:
                logger.error("[EXOT] Did not receive sync ack")
                return False

        except Exception as e:
            logger.error(f"[EXOT] Sync failed: {e}")
            return False

    def send_activation(self, activation: mx.array) -> mx.array | None:
        """
        Send activation tensor to iOS and receive processed result.

        This is the main tensor exchange method used during inference.
        The Mac sends intermediate activations, iOS processes them through
        its assigned layers, and returns the result.
        """
        if not self._connected or not self._socket:
            raise RuntimeError("Not connected to iOS TensorServer")

        with self._lock:
            try:
                # Create activation message
                msg = TensorMessage.from_array(MessageType.ACTIVATION, activation)

                # Send activation
                wire_data = msg.serialize()
                self._send_raw(wire_data)
                logger.debug(f"[EXOT] Sent activation: shape={activation.shape}, {len(wire_data)} bytes")

                # Receive result
                response = self._receive_message()
                if response is None:
                    raise RuntimeError("No response from iOS")

                if response.msg_type == MessageType.ERROR:
                    error_msg = response.data.decode("utf-8", errors="replace")
                    raise RuntimeError(f"iOS error: {error_msg}")

                if response.msg_type != MessageType.RESULT:
                    raise RuntimeError(f"Unexpected message type: {response.msg_type}")

                result = response.to_array()
                logger.debug(f"[EXOT] Received result: shape={result.shape}")
                return result

            except Exception as e:
                logger.error(f"[EXOT] Activation exchange failed: {e}")
                raise

    def _send_raw(self, data: bytes):
        """Send raw bytes over socket."""
        if not self._socket:
            raise RuntimeError("Not connected")

        total_sent = 0
        while total_sent < len(data):
            sent = self._socket.send(data[total_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            total_sent += sent

    def _receive_message(self) -> TensorMessage | None:
        """Receive a complete TensorMessage from socket."""
        if not self._socket:
            return None

        # Read minimum header
        header = self._recv_exact(20)
        if not header:
            return None

        # Parse ndim to know header size
        ndim = struct.unpack_from(">I", header, 16)[0]

        # Read rest of header (shape + data_len)
        remaining_header = self._recv_exact(ndim * 4 + 8)
        if not remaining_header:
            return None

        # Parse data length
        data_len = struct.unpack_from(">Q", remaining_header, ndim * 4)[0]

        # Read tensor data
        tensor_data = self._recv_exact(data_len)
        if not tensor_data:
            return None

        # Combine and deserialize
        full_data = header + remaining_header + tensor_data
        return TensorMessage.deserialize(full_data)

    def _recv_exact(self, n: int) -> bytes | None:
        """Receive exactly n bytes from socket."""
        if not self._socket:
            return None

        data = bytearray()
        while len(data) < n:
            try:
                chunk = self._socket.recv(n - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except socket.timeout:
                return None

        return bytes(data)

    @property
    def is_connected(self) -> bool:
        return self._connected


# Global client instance for the current pipeline
_exot_client: ExotClient | None = None
_exot_config: dict | None = None


def initialize_exot_pipeline(
    ios_host: str,
    ios_port: int,
    instance_id: str,
    model_id: str,
    local_start_layer: int,
    local_end_layer: int,
    ios_start_layer: int,
    ios_end_layer: int,
    local_rank: int,
    world_size: int,
) -> bool:
    """
    Initialize EXOT pipeline connection to iOS TensorServer.

    This should be called during distributed init when iOS nodes are detected.

    Args:
        ios_host: IP address of iOS device
        ios_port: Port of iOS TensorServer
        instance_id: Unique instance ID for this inference session
        model_id: Model being used (e.g., "mlx-community/Llama-3-8B")
        local_start_layer: First layer handled by Mac
        local_end_layer: Last layer handled by Mac (exclusive)
        ios_start_layer: First layer handled by iOS
        ios_end_layer: Last layer handled by iOS (exclusive)
        local_rank: This node's rank in the pipeline
        world_size: Total number of pipeline stages

    Returns:
        True if connection established successfully
    """
    global _exot_client, _exot_config

    logger.info(f"[EXOT] Initializing pipeline: Mac layers [{local_start_layer}, {local_end_layer}), iOS layers [{ios_start_layer}, {ios_end_layer})")

    # Create client and connect
    _exot_client = ExotClient(ios_host, ios_port)
    if not _exot_client.connect():
        _exot_client = None
        return False

    # Calculate total layers
    total_layers = max(local_end_layer, ios_end_layer)

    # Perform sync handshake
    sync_msg = SyncMessage(
        session_id=instance_id,
        model_id=model_id,
        start_layer=ios_start_layer,  # Tell iOS which layers it should process
        end_layer=ios_end_layer,
        world_size=world_size,
        rank=local_rank,
        total_layers=total_layers,
        is_first_stage=(ios_start_layer == 0),
        is_last_stage=(ios_end_layer == total_layers),
    )

    if not _exot_client.sync(sync_msg):
        _exot_client.disconnect()
        _exot_client = None
        return False

    # Store config for later use
    _exot_config = {
        "ios_host": ios_host,
        "ios_port": ios_port,
        "local_start_layer": local_start_layer,
        "local_end_layer": local_end_layer,
        "ios_start_layer": ios_start_layer,
        "ios_end_layer": ios_end_layer,
        "total_layers": total_layers,
    }

    logger.info("[EXOT] Pipeline initialized successfully")
    return True


def shutdown_exot_pipeline():
    """Shutdown EXOT pipeline and close connection."""
    global _exot_client, _exot_config

    if _exot_client:
        _exot_client.disconnect()
        _exot_client = None
    _exot_config = None

    logger.info("[EXOT] Pipeline shutdown complete")


def get_exot_client() -> ExotClient | None:
    """Get the global EXOT client instance."""
    return _exot_client


def send_to_ios(activation: mx.array) -> mx.array:
    """
    Send activation to iOS and get result.

    This is the main function called during inference to exchange
    tensors with the iOS device.
    """
    global _exot_client

    if _exot_client is None or not _exot_client.is_connected:
        raise RuntimeError("EXOT pipeline not initialized")

    result = _exot_client.send_activation(activation)
    if result is None:
        raise RuntimeError("Failed to get result from iOS")

    return result


class ExotPipelineWrapper(nn.Module):
    """
    Wrapper module that intercepts forward passes and routes to iOS when needed.

    This wraps the original model and, based on the pipeline configuration,
    sends activations to iOS for processing of its assigned layers.
    """

    def __init__(
        self,
        model: nn.Module,
        local_start_layer: int,
        local_end_layer: int,
        ios_start_layer: int,
        ios_end_layer: int,
        total_layers: int,
    ):
        super().__init__()
        self._model = model
        self._local_start = local_start_layer
        self._local_end = local_end_layer
        self._ios_start = ios_start_layer
        self._ios_end = ios_end_layer
        self._total_layers = total_layers

        # Determine pipeline order
        # Case 1: iOS processes first layers, Mac processes later layers
        # Case 2: Mac processes first layers, iOS processes later layers
        self._ios_first = ios_start_layer < local_start_layer

        logger.info(f"[EXOT] Pipeline wrapper: iOS {'first' if self._ios_first else 'second'}")
        logger.info(f"[EXOT] Local layers: [{local_start_layer}, {local_end_layer})")
        logger.info(f"[EXOT] iOS layers: [{ios_start_layer}, {ios_end_layer})")

    def __call__(self, x: mx.array, cache: list | None = None) -> mx.array:
        """
        Forward pass with iOS pipeline integration.

        Routes tensor to iOS for its portion of the layers.
        """
        if self._ios_first:
            # iOS processes first, then Mac
            # Send tokens/embeddings to iOS, get back intermediate activations
            logger.debug(f"[EXOT] Sending to iOS for layers [{self._ios_start}, {self._ios_end})")
            h = send_to_ios(x)

            # Then process locally
            logger.debug(f"[EXOT] Processing locally for layers [{self._local_start}, {self._local_end})")
            h = self._forward_local_layers(h, cache)

        else:
            # Mac processes first, then iOS
            logger.debug(f"[EXOT] Processing locally for layers [{self._local_start}, {self._local_end})")
            h = self._forward_local_layers(x, cache)

            # Send to iOS for remaining layers
            logger.debug(f"[EXOT] Sending to iOS for layers [{self._ios_start}, {self._ios_end})")
            h = send_to_ios(h)

        return h

    def _forward_local_layers(self, x: mx.array, cache: list | None) -> mx.array:
        """Forward through the locally-assigned layers."""
        # Access the underlying model's forward logic
        # This depends on the model architecture
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            # Standard LLM structure (e.g., Llama, Qwen)
            inner_model = self._model.model

            # If we're the first stage, apply embedding
            if self._local_start == 0 and hasattr(inner_model, "embed_tokens"):
                x = inner_model.embed_tokens(x)

            # Create attention mask
            mask = None
            if hasattr(inner_model, "create_attention_mask"):
                mask = inner_model.create_attention_mask(x, cache[0] if cache else None)

            # Forward through assigned layers
            for i in range(self._local_start, self._local_end):
                layer = inner_model.layers[i]
                cache_i = cache[i - self._local_start] if cache else None
                x = layer(x, mask=mask, cache=cache_i)

            # If we're the last stage, apply final norm and lm_head
            if self._local_end == self._total_layers:
                if hasattr(inner_model, "norm"):
                    x = inner_model.norm(x)
                if hasattr(self._model, "lm_head"):
                    x = self._model.lm_head(x)
                elif hasattr(inner_model, "embed_tokens"):
                    # Tied embeddings
                    x = inner_model.embed_tokens.as_linear(x)

            return x
        else:
            # Fallback - just call the model
            return self._model(x, cache=cache)

    @property
    def layers(self):
        """Access underlying model layers for cache creation."""
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return self._model.model.layers
        return getattr(self._model, "layers", [])

    def parameters(self):
        """Forward parameters to underlying model."""
        return self._model.parameters()

    def __getattr__(self, name: str):
        """Forward attribute access to underlying model."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self._model, name)


def exot_pipeline_auto_parallel(
    model: nn.Module,
    shard_metadata,  # PipelineShardMetadata
) -> nn.Module:
    """
    Apply EXOT pipeline parallelism to a model.

    This wraps the model to intercept forward passes and route tensors
    to iOS for processing of its assigned layers.

    Args:
        model: The MLX model to wrap
        shard_metadata: Pipeline shard metadata with layer assignments

    Returns:
        Wrapped model that communicates with iOS
    """
    global _exot_config

    if _exot_config is None:
        raise RuntimeError("EXOT pipeline not initialized - call initialize_exot_pipeline first")

    local_start = _exot_config["local_start_layer"]
    local_end = _exot_config["local_end_layer"]
    ios_start = _exot_config["ios_start_layer"]
    ios_end = _exot_config["ios_end_layer"]
    total_layers = _exot_config["total_layers"]

    logger.info(f"[EXOT] Wrapping model for EXOT pipeline parallelism")

    return ExotPipelineWrapper(
        model=model,
        local_start_layer=local_start,
        local_end_layer=local_end,
        ios_start_layer=ios_start,
        ios_end_layer=ios_end,
        total_layers=total_layers,
    )
