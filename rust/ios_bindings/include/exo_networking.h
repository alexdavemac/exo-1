#ifndef EXO_NETWORKING_H
#define EXO_NETWORKING_H

#include "stdarg.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdlib.h"

/**
 * Callback function type for receiving events from Rust
 *
 * Parameters:
 * - event_type: Type of event (see EventType enum)
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
 * # Safety
 * - `callback` must be a valid function pointer that remains valid for the lifetime of the networking layer
 * - Must only be called once; subsequent calls return the existing peer ID
 *
 * Returns the local peer ID as a C string, or null on failure.
 * The caller must free the returned string using `exo_free_string`.
 */
const char *exo_init(EventCallback callback);

/**
 * Subscribe to a gossipsub topic
 *
 * # Safety
 * - `topic` must be a valid null-terminated C string
 *
 * Returns true on success, false on failure.
 */
bool exo_subscribe(const char *topic);

/**
 * Publish data to a gossipsub topic
 *
 * # Safety
 * - `topic` must be a valid null-terminated C string
 * - `data` must be a valid pointer to `data_len` bytes
 *
 * Returns true on success, false on failure.
 */
bool exo_publish(const char *topic, const uint8_t *data, uintptr_t data_len);

/**
 * Dial a peer at the given multiaddr
 *
 * # Safety
 * - `addr` must be a valid null-terminated C string containing a valid multiaddr
 *
 * Returns true on success, false on failure.
 */
bool exo_dial(const char *addr);

/**
 * Get the local peer ID
 *
 * # Safety
 * - Must be called after `exo_init`
 *
 * Returns the peer ID as a C string, or null if not initialized.
 * The caller must free the returned string using `exo_free_string`.
 */
const char *exo_get_peer_id(void);

/**
 * Shutdown the networking layer
 *
 * # Safety
 * - Can be called multiple times safely
 */
void exo_shutdown(void);

/**
 * Free a string allocated by Rust
 *
 * # Safety
 * - `s` must be a string previously returned by this library, or null
 * - Must not be called more than once for the same string
 */
void exo_free_string(char *s);

#endif  /* EXO_NETWORKING_H */
