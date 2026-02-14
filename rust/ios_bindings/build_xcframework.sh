#!/bin/bash
# Build ExoNetworking.xcframework for iOS device and simulator
#
# Usage: ./build_xcframework.sh [--release]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/build"
XCFRAMEWORK_NAME="ExoNetworking"

# Parse arguments
BUILD_TYPE="release"
CARGO_FLAGS="--release"

echo "=== Building ExoNetworking.xcframework ==="
echo "Project root: $PROJECT_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Build type: $BUILD_TYPE"
echo ""

# Ensure iOS targets are installed
echo "=== Checking Rust iOS targets ==="
rustup target add aarch64-apple-ios 2>/dev/null || true
rustup target add aarch64-apple-ios-sim 2>/dev/null || true

# Clean previous build
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Build for iOS device (arm64)
echo ""
echo "=== Building for iOS device (aarch64-apple-ios) ==="
cd "$PROJECT_ROOT"
cargo build -p ios_bindings $CARGO_FLAGS --target aarch64-apple-ios

# Build for iOS simulator (arm64)
echo ""
echo "=== Building for iOS simulator (aarch64-apple-ios-sim) ==="
cargo build -p ios_bindings $CARGO_FLAGS --target aarch64-apple-ios-sim

# Locate built libraries
DEVICE_LIB="$PROJECT_ROOT/target/aarch64-apple-ios/$BUILD_TYPE/libexo_networking.a"
SIM_LIB="$PROJECT_ROOT/target/aarch64-apple-ios-sim/$BUILD_TYPE/libexo_networking.a"

if [ ! -f "$DEVICE_LIB" ]; then
    echo "ERROR: Device library not found at $DEVICE_LIB"
    exit 1
fi

if [ ! -f "$SIM_LIB" ]; then
    echo "ERROR: Simulator library not found at $SIM_LIB"
    exit 1
fi

echo ""
echo "=== Creating xcframework structure ==="

# Create directories for each platform
DEVICE_DIR="$OUTPUT_DIR/ios-arm64"
SIM_DIR="$OUTPUT_DIR/ios-arm64-simulator"
mkdir -p "$DEVICE_DIR/Headers"
mkdir -p "$SIM_DIR/Headers"

# Copy libraries
cp "$DEVICE_LIB" "$DEVICE_DIR/libexo_networking.a"
cp "$SIM_LIB" "$SIM_DIR/libexo_networking.a"

# Generate/copy header
HEADER_CONTENT='#ifndef EXO_NETWORKING_H
#define EXO_NETWORKING_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Callback function type for receiving events from Rust
 *
 * Parameters:
 * - event_type: Type of event (1=PeerConnected, 2=PeerDisconnected, 3=Message, 4=Listening)
 * - data: Topic name (for messages) or address (for listening), null-terminated C string
 * - data_len: Length of data string
 * - message_bytes: Raw message bytes (for Message events)
 * - message_len: Length of message bytes
 * - peer_id: Peer ID as null-terminated C string (for peer events)
 */
typedef void (*EventCallback)(int32_t event_type,
                              const char *data,
                              uintptr_t data_len,
                              const uint8_t *message_bytes,
                              uintptr_t message_len,
                              const char *peer_id);

/**
 * Initialize the networking layer
 *
 * Returns the local peer ID as a C string, or null on failure.
 * The caller must free the returned string using exo_free_string.
 */
const char *exo_init(EventCallback callback);

/**
 * Subscribe to a gossipsub topic
 *
 * Returns true on success, false on failure.
 */
bool exo_subscribe(const char *topic);

/**
 * Publish data to a gossipsub topic
 *
 * Returns true on success, false on failure.
 */
bool exo_publish(const char *topic, const uint8_t *data, uintptr_t data_len);

/**
 * Dial a peer at the given multiaddr
 *
 * Returns true on success, false on failure.
 */
bool exo_dial(const char *addr);

/**
 * Get the local peer ID
 *
 * Returns the peer ID as a C string, or null if not initialized.
 * The caller must free the returned string using exo_free_string.
 */
const char *exo_get_peer_id(void);

/**
 * Shutdown the networking layer
 */
void exo_shutdown(void);

/**
 * Free a string allocated by Rust
 */
void exo_free_string(char *s);

#endif /* EXO_NETWORKING_H */
'

echo "$HEADER_CONTENT" > "$DEVICE_DIR/Headers/exo_networking.h"
echo "$HEADER_CONTENT" > "$SIM_DIR/Headers/exo_networking.h"

# Create module.modulemap
MODULE_MAP='module ExoNetworking {
    header "exo_networking.h"
    export *
}
'

echo "$MODULE_MAP" > "$DEVICE_DIR/Headers/module.modulemap"
echo "$MODULE_MAP" > "$SIM_DIR/Headers/module.modulemap"

# Create xcframework using xcodebuild
echo ""
echo "=== Creating xcframework ==="
XCFRAMEWORK_PATH="$OUTPUT_DIR/${XCFRAMEWORK_NAME}.xcframework"

xcodebuild -create-xcframework \
    -library "$DEVICE_DIR/libexo_networking.a" \
    -headers "$DEVICE_DIR/Headers" \
    -library "$SIM_DIR/libexo_networking.a" \
    -headers "$SIM_DIR/Headers" \
    -output "$XCFRAMEWORK_PATH"

echo ""
echo "=== Build complete ==="
echo "XCFramework created at: $XCFRAMEWORK_PATH"
echo ""

# Show xcframework structure
echo "XCFramework structure:"
find "$XCFRAMEWORK_PATH" -type f | head -20

# Optionally copy to exo-ios project
EXO_IOS_FRAMEWORKS="/Users/alexmcdaniel/Projects/exo-ios/Frameworks"
if [ -d "$EXO_IOS_FRAMEWORKS" ]; then
    echo ""
    echo "=== Copying to exo-ios project ==="
    rm -rf "$EXO_IOS_FRAMEWORKS/${XCFRAMEWORK_NAME}.xcframework"
    cp -R "$XCFRAMEWORK_PATH" "$EXO_IOS_FRAMEWORKS/"
    echo "Copied to: $EXO_IOS_FRAMEWORKS/${XCFRAMEWORK_NAME}.xcframework"
fi

echo ""
echo "=== Done! ==="
