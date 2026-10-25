//! Bounded, process-local proof observations; enable only in isolated proof children.
//! Engine IDs join actual owners across moves to Install/Transfer owner tokens and
//! StartupOwner snapshots. No pointer identity or ownership substitution is used.
use std::{
    sync::{Arc, Mutex, OnceLock},
    thread::{self, ThreadId},
};

use tokio::runtime::{Handle, Id, RuntimeFlavor};

/// Overflow invalidates a proof interval; events are never silently evicted.
pub const CAPACITY: usize = 65_536;

/// These observations are emitted only at the corresponding SDK execution paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Install,
    Transfer,
    ExactId404,
    ActualDrop,
    AsyncDropDispatch,
    AsyncDropEntry,
}

/// Runtime/thread refer to the caller at the event, before any async-drop migration.
#[derive(Clone, Debug)]
pub struct Event {
    pub seq: u64,
    pub kind: Kind,
    /// Empty only for generic async_drop entry, which has no container argument.
    pub id: String,
    /// Present on Install/Transfer; join subsequent events by exact Engine ID.
    pub token: Option<String>,
    pub runtime: Option<Id>,
    /// Shared verbatim: Tokio flavors are non-Clone and may gain unknown variants.
    pub flavor: Option<Arc<RuntimeFlavor>>,
    pub thread: ThreadId,
    pub disarmed: Option<bool>,
}

#[derive(Default)]
struct Log {
    events: Vec<Event>,
    overflow: bool,
    seq: u64,
    total_async_drop: u64,
}

impl Log {
    fn push(&mut self, mut event: Event) {
        self.seq += 1;
        event.seq = self.seq;
        self.total_async_drop += u64::from(event.kind == Kind::AsyncDropEntry);
        if self.events.len() < CAPACITY {
            self.events.push(event);
        } else {
            self.overflow = true;
        }
    }
}

fn log() -> std::sync::MutexGuard<'static, Log> {
    static LOG: OnceLock<Mutex<Log>> = OnceLock::new();
    LOG.get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|poison| {
            let mut log = poison.into_inner();
            log.overflow = true;
            log
        })
}

pub(crate) fn record(kind: Kind, id: &str, token: Option<&str>, disarmed: Option<bool>) {
    let handle = Handle::try_current().ok();
    let event = Event {
        seq: 0,
        kind,
        id: id.into(),
        token: token.map(str::to_owned),
        runtime: handle.as_ref().map(Handle::id),
        flavor: handle
            .as_ref()
            .map(|handle| Arc::new(handle.runtime_flavor())),
        thread: thread::current().id(),
        disarmed,
    };
    log().push(event);
}

/// Returns ordered events, sticky overflow, and all generic async_drop entries.
/// Uses an independent mutex, never the startup async mutex. Reset by child exit;
/// callers must reject overflow and compare counter deltas over isolated intervals.
pub fn snapshot() -> (Vec<Event>, bool, u64) {
    let log = log();
    (log.events.clone(), log.overflow, log.total_async_drop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_async_drop_entry_keeps_caller_identity_on_both_flavors() {
        for threaded in [false, true] {
            let mut builder = if threaded {
                tokio::runtime::Builder::new_multi_thread()
            } else {
                tokio::runtime::Builder::new_current_thread()
            };
            let runtime = builder.worker_threads(2).build().unwrap();
            runtime.block_on(async {
                let handle = Handle::current();
                let before = snapshot();
                assert!(!before.1);
                let before_seq = before.0.last().map_or(0, |event| event.seq);
                crate::core::async_drop::async_drop(async {});
                let after = snapshot();
                assert!(!after.1);
                assert!(after.2 > before.2);
                let invocation: Vec<_> = after
                    .0
                    .iter()
                    .filter(|event| {
                        event.seq > before_seq
                            && event.runtime == Some(handle.id())
                            && event.thread == thread::current().id()
                    })
                    .collect();
                assert_eq!(invocation.len(), 1);
                let event = invocation[0];
                assert_eq!(event.kind, Kind::AsyncDropEntry);
                assert_eq!(event.thread, thread::current().id());
                assert_eq!(event.flavor.as_deref(), Some(&handle.runtime_flavor()));
                let cloned = event.clone();
                assert!(Arc::ptr_eq(
                    event.flavor.as_ref().unwrap(),
                    cloned.flavor.as_ref().unwrap(),
                ));
                assert_eq!(cloned.seq, event.seq);
                assert_eq!(cloned.kind, event.kind);
                assert_eq!(cloned.id, event.id);
                assert_eq!(cloned.token, event.token);
                assert_eq!(cloned.runtime, event.runtime);
                assert_eq!(cloned.thread, event.thread);
                assert_eq!(cloned.disarmed, event.disarmed);
                assert!(event.id.is_empty());
                assert!(event.token.is_none());
                assert!(event.disarmed.is_none());
                assert!(event.seq > before.0.last().map_or(0, |event| event.seq));
                let mut bounded = Log::default();
                for _ in 0..=CAPACITY {
                    bounded.push(event.clone());
                }
                assert_eq!(bounded.events.len(), CAPACITY);
                assert!(bounded.overflow);
                assert_eq!(bounded.total_async_drop, CAPACITY as u64 + 1);
                assert_eq!(bounded.events.first().unwrap().seq, 1);
                assert_eq!(bounded.events.last().unwrap().seq, CAPACITY as u64);
                assert!(after.0.windows(2).all(|pair| pair[0].seq < pair[1].seq));
            });
        }
    }
}
