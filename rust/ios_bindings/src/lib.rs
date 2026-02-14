//! iOS bindings for exo networking layer.
//!
//! Provides a C-compatible API for Swift integration via FFI.
//! Uses libp2p gossipsub for peer-to-peer messaging.

#![allow(
    clippy::not_unsafe_ptr_arg_deref,
    clippy::missing_safety_doc,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

use libp2p::futures::StreamExt;
use libp2p::gossipsub::{IdentTopic, Message};
use libp2p::swarm::SwarmEvent;
use libp2p::{Multiaddr, PeerId, gossipsub, identity};
use networking::discovery;
use networking::swarm::{BehaviourEvent, create_swarm};
use once_cell::sync::OnceCell;
use std::ffi::{CStr, CString};
use std::net::IpAddr;
use std::os::raw::c_char;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

/// Event types matching Swift's NetworkEventType enum
#[repr(i32)]
pub enum EventType {
    PeerConnected = 1,
    PeerDisconnected = 2,
    Message = 3,
    Listening = 4,
}

/// Callback function type for receiving events from Rust
///
/// Parameters:
/// - event_type: Type of event (see EventType enum)
/// - data: Topic name (for messages) or address (for listening), null-terminated C string
/// - data_len: Length of data string
/// - message_bytes: Raw message bytes (for Message events)
/// - message_len: Length of message bytes
/// - peer_id: Peer ID as null-terminated C string (for peer events)
pub type EventCallback = extern "C" fn(
    event_type: i32,
    data: *const c_char,
    data_len: usize,
    message_bytes: *const u8,
    message_len: usize,
    peer_id: *const c_char,
);

/// Commands sent from the C API to the async networking task
enum Command {
    Subscribe { topic: String },
    Unsubscribe { topic: String },
    Publish { topic: String, data: Vec<u8> },
    Dial { addr: String },
    Shutdown,
}

/// Global state for the networking layer
struct NetworkingState {
    runtime: Runtime,
    command_tx: mpsc::Sender<Command>,
    peer_id: PeerId,
}

static NETWORKING: OnceCell<NetworkingState> = OnceCell::new();

/// Initialize the networking layer
///
/// # Safety
/// - `callback` must be a valid function pointer that remains valid for the lifetime of the networking layer
/// - Must only be called once; subsequent calls return the existing peer ID
///
/// Returns the local peer ID as a C string, or null on failure.
/// The caller must free the returned string using `exo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn exo_init(callback: EventCallback) -> *const c_char {
    // Initialize logging for iOS
    init_logging();

    log::info!("exo_init: starting networking layer");

    // Check if already initialized
    if let Some(state) = NETWORKING.get() {
        log::info!("exo_init: already initialized, returning existing peer ID");
        return peer_id_to_cstring(&state.peer_id);
    }

    // Create tokio runtime
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            log::error!("exo_init: failed to create tokio runtime: {}", e);
            return std::ptr::null();
        }
    };

    // Generate identity keypair
    let keypair = identity::Keypair::generate_ed25519();
    let peer_id = keypair.public().to_peer_id();
    log::info!("exo_init: generated peer ID: {}", peer_id);

    // Create command channel
    let (command_tx, command_rx) = mpsc::channel::<Command>(256);

    // Create swarm within tokio context
    let swarm = match runtime.block_on(async { create_swarm(keypair) }) {
        Ok(s) => s,
        Err(e) => {
            log::error!("exo_init: failed to create swarm: {}", e);
            return std::ptr::null();
        }
    };

    // Spawn the networking task
    let peer_id_clone = peer_id;
    runtime.spawn(async move {
        networking_task(swarm, command_rx, callback, peer_id_clone).await;
    });

    // Store global state
    let state = NetworkingState {
        runtime,
        command_tx,
        peer_id,
    };

    match NETWORKING.set(state) {
        Ok(()) => {
            log::info!("exo_init: networking layer initialized successfully");
            peer_id_to_cstring(&peer_id)
        }
        Err(_) => {
            log::error!("exo_init: failed to store networking state (race condition?)");
            std::ptr::null()
        }
    }
}

/// Subscribe to a gossipsub topic
///
/// # Safety
/// - `topic` must be a valid null-terminated C string
///
/// Returns true on success, false on failure.
#[unsafe(no_mangle)]
pub extern "C" fn exo_subscribe(topic: *const c_char) -> bool {
    let topic = match unsafe { cstr_to_string(topic) } {
        Some(s) => s,
        None => return false,
    };

    log::info!("exo_subscribe: subscribing to topic: {}", topic);

    send_command(Command::Subscribe { topic })
}

/// Publish data to a gossipsub topic
///
/// # Safety
/// - `topic` must be a valid null-terminated C string
/// - `data` must be a valid pointer to `data_len` bytes
///
/// Returns true on success, false on failure.
#[unsafe(no_mangle)]
pub extern "C" fn exo_publish(topic: *const c_char, data: *const u8, data_len: usize) -> bool {
    let topic = match unsafe { cstr_to_string(topic) } {
        Some(s) => s,
        None => return false,
    };

    let data = if data.is_null() || data_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data, data_len) }.to_vec()
    };

    log::debug!("exo_publish: publishing {} bytes to topic: {}", data.len(), topic);

    send_command(Command::Publish { topic, data })
}

/// Dial a peer at the given multiaddr
///
/// # Safety
/// - `addr` must be a valid null-terminated C string containing a valid multiaddr
///
/// Returns true on success, false on failure.
#[unsafe(no_mangle)]
pub extern "C" fn exo_dial(addr: *const c_char) -> bool {
    let addr = match unsafe { cstr_to_string(addr) } {
        Some(s) => s,
        None => return false,
    };

    log::info!("exo_dial: dialing address: {}", addr);

    send_command(Command::Dial { addr })
}

/// Get the local peer ID
///
/// # Safety
/// - Must be called after `exo_init`
///
/// Returns the peer ID as a C string, or null if not initialized.
/// The caller must free the returned string using `exo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn exo_get_peer_id() -> *const c_char {
    match NETWORKING.get() {
        Some(state) => peer_id_to_cstring(&state.peer_id),
        None => {
            log::warn!("exo_get_peer_id: networking not initialized");
            std::ptr::null()
        }
    }
}

/// Shutdown the networking layer
///
/// # Safety
/// - Can be called multiple times safely
#[unsafe(no_mangle)]
pub extern "C" fn exo_shutdown() {
    log::info!("exo_shutdown: shutting down networking layer");

    if let Some(state) = NETWORKING.get() {
        // Send shutdown command (ignore errors if channel is closed)
        let _ = state.runtime.block_on(async {
            state.command_tx.send(Command::Shutdown).await
        });
    }
}

/// Free a string allocated by Rust
///
/// # Safety
/// - `s` must be a string previously returned by this library, or null
/// - Must not be called more than once for the same string
#[unsafe(no_mangle)]
pub extern "C" fn exo_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

// ============================================================================
// Internal implementation
// ============================================================================

fn init_logging() {
    // Use oslog for iOS logging
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Try to use oslog, fall back to env_logger
        if oslog::OsLogger::new("com.exo.networking")
            .level_filter(log::LevelFilter::Debug)
            .init()
            .is_err()
        {
            // Fallback - just ignore logging
        }
    });
}

fn send_command(cmd: Command) -> bool {
    match NETWORKING.get() {
        Some(state) => {
            match state.runtime.block_on(async {
                state.command_tx.send(cmd).await
            }) {
                Ok(()) => true,
                Err(e) => {
                    log::error!("Failed to send command: {}", e);
                    false
                }
            }
        }
        None => {
            log::warn!("Networking not initialized");
            false
        }
    }
}

unsafe fn cstr_to_string(s: *const c_char) -> Option<String> {
    if s.is_null() {
        return None;
    }
    match unsafe { CStr::from_ptr(s) }.to_str() {
        Ok(s) => Some(s.to_owned()),
        Err(e) => {
            log::error!("Invalid UTF-8 in C string: {}", e);
            None
        }
    }
}

fn peer_id_to_cstring(peer_id: &PeerId) -> *const c_char {
    match CString::new(peer_id.to_string()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null(),
    }
}

fn string_to_cstring(s: &str) -> *const c_char {
    match CString::new(s) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null(),
    }
}

/// Main networking task that runs on the tokio runtime
async fn networking_task(
    mut swarm: networking::swarm::Swarm,
    mut command_rx: mpsc::Receiver<Command>,
    callback: EventCallback,
    local_peer_id: PeerId,
) {
    log::info!("networking_task: started");

    // Wrap callback in Arc for potential future cloning needs
    let callback = Arc::new(callback);

    loop {
        tokio::select! {
            // Handle commands from the C API
            cmd = command_rx.recv() => {
                match cmd {
                    Some(Command::Subscribe { topic }) => {
                        match swarm.behaviour_mut().gossipsub.subscribe(&IdentTopic::new(&topic)) {
                            Ok(true) => log::info!("Subscribed to topic: {}", topic),
                            Ok(false) => log::debug!("Already subscribed to topic: {}", topic),
                            Err(e) => log::error!("Failed to subscribe to {}: {:?}", topic, e),
                        }
                    }
                    Some(Command::Unsubscribe { topic }) => {
                        if swarm.behaviour_mut().gossipsub.unsubscribe(&IdentTopic::new(&topic)) {
                            log::info!("Unsubscribed from topic: {}", topic);
                        }
                    }
                    Some(Command::Publish { topic, data }) => {
                        let result = swarm.behaviour_mut().gossipsub.publish(
                            IdentTopic::new(&topic),
                            data,
                        );
                        match result {
                            Ok(_) => {
                                // Send success diagnostic
                                let diag_topic = format!("__publish_ok:{}", topic);
                                send_event(&callback, EventType::Message, Some(&diag_topic), None, None);
                            }
                            Err(e) => {
                                log::error!("Failed to publish to {}: {:?}", topic, e);
                                // Send failure diagnostic
                                let diag_topic = format!("__publish_fail:{}:{:?}", topic, e);
                                send_event(&callback, EventType::Message, Some(&diag_topic), None, None);
                            }
                        }
                    }
                    Some(Command::Dial { addr }) => {
                        match addr.parse::<Multiaddr>() {
                            Ok(multiaddr) => {
                                match swarm.dial(multiaddr.clone()) {
                                    Ok(()) => log::info!("Dialing: {}", multiaddr),
                                    Err(e) => log::error!("Failed to dial {}: {:?}", multiaddr, e),
                                }
                            }
                            Err(e) => log::error!("Invalid multiaddr {}: {:?}", addr, e),
                        }
                    }
                    Some(Command::Shutdown) | None => {
                        log::info!("networking_task: shutting down");
                        break;
                    }
                }
            }

            // Handle swarm events
            event = swarm.select_next_some() => {
                handle_swarm_event(&callback, event, &local_peer_id);
            }
        }
    }

    log::info!("networking_task: stopped");
}

fn handle_swarm_event(
    callback: &EventCallback,
    event: SwarmEvent<BehaviourEvent>,
    _local_peer_id: &PeerId,
) {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            log::info!("Listening on: {}", address);
            send_event(callback, EventType::Listening, Some(&address.to_string()), None, None);
        }

        SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(gossipsub::Event::Message {
            message: Message { topic, data, .. },
            propagation_source,
            ..
        })) => {
            let topic_str = topic.to_string();
            log::debug!("Received message on '{}' ({} bytes) from {}", topic_str, data.len(), propagation_source);
            send_event(
                callback,
                EventType::Message,
                Some(&topic_str),
                Some(&data),
                Some(&propagation_source.to_string()),
            );
        }

        SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(gossipsub::Event::Subscribed { peer_id, topic })) => {
            log::info!("Peer {} subscribed to {}", peer_id, topic);
            let diag_topic = format!("__peer_subscribed:{}:{}", topic, peer_id);
            send_event(callback, EventType::Message, Some(&diag_topic), None, None);
        }

        SwarmEvent::Behaviour(BehaviourEvent::Discovery(discovery::Event::ConnectionEstablished {
            peer_id,
            remote_ip,
            remote_tcp_port,
            ..
        })) => {
            let remote_addr = format_remote_addr(remote_ip, remote_tcp_port);
            log::info!("Peer connected: {} at {}", peer_id, remote_addr);
            send_event(
                callback,
                EventType::PeerConnected,
                Some(&remote_addr),
                None,
                Some(&peer_id.to_string()),
            );
        }

        SwarmEvent::Behaviour(BehaviourEvent::Discovery(discovery::Event::ConnectionClosed {
            peer_id,
            remote_ip,
            remote_tcp_port,
            ..
        })) => {
            let remote_addr = format_remote_addr(remote_ip, remote_tcp_port);
            log::info!("Peer disconnected: {} from {}", peer_id, remote_addr);
            send_event(
                callback,
                EventType::PeerDisconnected,
                None,
                None,
                Some(&peer_id.to_string()),
            );
        }

        // Log other events at debug level
        event => {
            log::debug!("Swarm event: {:?}", event);
        }
    }
}

fn format_remote_addr(ip: IpAddr, port: u16) -> String {
    match ip {
        IpAddr::V4(ip) => format!("/ip4/{}/tcp/{}", ip, port),
        IpAddr::V6(ip) => format!("/ip6/{}/tcp/{}", ip, port),
    }
}

fn send_event(
    callback: &EventCallback,
    event_type: EventType,
    data: Option<&str>,
    message_bytes: Option<&[u8]>,
    peer_id: Option<&str>,
) {
    let data_cstr = data.map(|s| CString::new(s).ok()).flatten();
    let data_ptr = data_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let data_len = data.map_or(0, |s| s.len());

    let msg_ptr = message_bytes.map_or(std::ptr::null(), |b| b.as_ptr());
    let msg_len = message_bytes.map_or(0, |b| b.len());

    let peer_cstr = peer_id.map(|s| CString::new(s).ok()).flatten();
    let peer_ptr = peer_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

    callback(
        event_type as i32,
        data_ptr,
        data_len,
        msg_ptr,
        msg_len,
        peer_ptr,
    );
}
