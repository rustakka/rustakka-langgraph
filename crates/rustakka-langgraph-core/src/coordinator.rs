//! `GraphCoordinator` rustakka actor enforcing Pregel Plan / Execute / Update.
//!
//! Lifecycle of one invocation:
//!   1. `StartRun { input, cfg, reply }` seeds channels and kicks off step 1.
//!   2. For each superstep:
//!        - Plan: pick nodes whose subscribed channels were updated.
//!        - Execute: spawn each node as an async task into the rustakka
//!          dispatcher; receive `NodeDone` / `Interrupt` messages back.
//!        - Update: apply pending writes, decide next targets, publish
//!          stream events, persist a checkpoint.
//!   3. When no more targets remain (or recursion limit hit) → reply.
//!
//! We keep the actor purely message-driven — no `RwLock` held across awaits.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use rustakka_core::actor::{Actor, Context};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::command::{Command, Interrupt, Send as GraphSend};
use crate::config::{RunnableConfig, StreamMode};
use crate::errors::{GraphError, GraphResult};
use crate::graph::{GraphTopology, END, START};
use crate::node::{NodeKind, NodeOutput};
use crate::state::GraphValues;
use crate::stream::{StreamBus, StreamEvent};

// --------------------------- public message API ---------------------------

/// Reply for a single graph run.
#[derive(Debug)]
pub struct RunResult {
    pub values: BTreeMap<String, Value>,
    pub interrupted: Option<Interrupt>,
    pub steps: u64,
    /// If the final step emitted a [`Command`] targeting the parent graph
    /// (`Command.graph == Some("PARENT")`), it's surfaced here so the subgraph
    /// adapter can forward it to the parent coordinator unchanged.
    pub parent_command: Option<Command>,
}

/// Messages handled by the coordinator actor.
pub enum CoordMsg {
    StartRun {
        input: BTreeMap<String, Value>,
        cfg: RunnableConfig,
        resume: Option<Value>,
        reply: oneshot::Sender<GraphResult<RunResult>>,
    },
    /// Internal: a dispatched node finished. `task_id` uniquely identifies
    /// this dispatch (node-name + a sequence so fan-out via `Send` doesn't
    /// collide on a single pending key).
    NodeDone {
        task_id: String,
        node: String,
        result: GraphResult<NodeOutput>,
    },
    /// Internal: the planner produced the next set of targets.
    Stop,
}

// ----------------------------- actor state -----------------------------

pub struct GraphCoordinator {
    pub topology: Arc<GraphTopology>,
    pub stream_bus: Arc<StreamBus>,
    /// Optional checkpointer (boxed trait object set externally).
    pub checkpointer: Option<Arc<dyn CheckpointerHook>>,
    /// Optional long-term store. When set, we install it into a task-local
    /// around each node invocation so nodes can call
    /// [`crate::context::get_store`] to access it.
    pub store: Option<Arc<dyn crate::context::StoreAccessor>>,
    /// Shared per-graph cache used by nodes with a `CachePolicy`.
    pub cache: Option<Arc<crate::graph::NodeCache>>,
    state: RunState,
}

#[derive(Default)]
struct RunState {
    values: Option<GraphValues>,
    cfg: RunnableConfig,
    step: u64,
    pending: HashSet<String>,
    /// `(node_name, writes_or_command)` collected during the Execute phase.
    in_flight_writes: Vec<(String, NodeOutput)>,
    next_targets: VecDeque<DispatchTarget>,
    /// One-shot reply for the active StartRun, if any.
    reply: Option<oneshot::Sender<GraphResult<RunResult>>>,
    aborted: bool,
    interrupt: Option<Interrupt>,
    /// If a node emitted a `Command { graph: Some("PARENT"), ... }`, we stash
    /// it here and surface it in `RunResult.parent_command` at finish.
    parent_command: Option<Command>,
    /// Set of nodes to interrupt *after* their writes are applied (drained as
    /// they fire so we only interrupt on the first hit).
    interrupt_after_remaining: HashSet<String>,
}

#[derive(Debug, Clone)]
struct DispatchTarget {
    node: String,
    /// Optional explicit per-target input override (for `Send` fan-out).
    input_override: Option<Value>,
}

/// Hook the engine uses for checkpoint persistence without depending on a
/// concrete saver implementation.
#[async_trait]
pub trait CheckpointerHook: Send + Sync + 'static {
    async fn put_step(
        &self,
        cfg: &RunnableConfig,
        step: u64,
        values: &BTreeMap<String, Value>,
        snapshot: &BTreeMap<String, crate::channel::ChannelSnapshot>,
        pending_writes: &[(String, BTreeMap<String, Value>)],
        interrupt: Option<&Interrupt>,
    ) -> GraphResult<()>;

    async fn get_latest(
        &self,
        cfg: &RunnableConfig,
    ) -> GraphResult<Option<CheckpointReplay>>;

    /// Persist per-task "pending writes" immediately after a node completes.
    /// Upstream savers record these so an in-flight fan-out step can be
    /// replayed on crash. Default: no-op (MemorySaver overrides it).
    async fn put_writes(
        &self,
        _cfg: &RunnableConfig,
        _task_id: &str,
        _writes: &[(String, Value)],
    ) -> GraphResult<()> {
        Ok(())
    }

    /// Return a specific checkpoint by `checkpoint_id` (for time-travel).
    /// Default: fall back to `get_latest` (suitable for in-memory savers that
    /// store a chain but don't support per-id lookup directly).
    async fn get_at(
        &self,
        cfg: &RunnableConfig,
    ) -> GraphResult<Option<CheckpointReplay>> {
        self.get_latest(cfg).await
    }

    /// List prior checkpoints for time-travel / history. Default: empty.
    async fn list_checkpoints(
        &self,
        _cfg: &RunnableConfig,
        _limit: Option<u32>,
    ) -> GraphResult<Vec<CheckpointReplay>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointReplay {
    pub step: u64,
    pub snapshot: BTreeMap<String, crate::channel::ChannelSnapshot>,
    pub interrupt: Option<Interrupt>,
}

impl GraphCoordinator {
    pub fn new(
        topology: Arc<GraphTopology>,
        stream_bus: Arc<StreamBus>,
        checkpointer: Option<Arc<dyn CheckpointerHook>>,
    ) -> Self {
        Self {
            topology,
            stream_bus,
            checkpointer,
            store: None,
            cache: None,
            state: RunState::default(),
        }
    }

    pub fn with_store(mut self, store: Option<Arc<dyn crate::context::StoreAccessor>>) -> Self {
        self.store = store;
        self
    }

    pub fn with_cache(mut self, cache: Option<Arc<crate::graph::NodeCache>>) -> Self {
        self.cache = cache;
        self
    }
}

#[async_trait]
impl Actor for GraphCoordinator {
    type Msg = CoordMsg;

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: CoordMsg) {
        match msg {
            CoordMsg::StartRun { input, cfg, resume, reply } => {
                self.start_run(ctx, input, cfg, resume, reply).await;
            }
            CoordMsg::NodeDone { task_id, node, result } => {
                self.on_node_done(ctx, task_id, node, result).await;
            }
            CoordMsg::Stop => {
                ctx.stop_self();
            }
        }
    }
}

impl GraphCoordinator {
    async fn start_run(
        &mut self,
        ctx: &mut Context<Self>,
        input: BTreeMap<String, Value>,
        cfg: RunnableConfig,
        resume: Option<Value>,
        reply: oneshot::Sender<GraphResult<RunResult>>,
    ) {
        // Fresh per-run state.
        let values = GraphValues::new(&self.topology.channel_specs);

        // Replay from checkpoint if available. Use `get_at` when
        // `cfg.checkpoint_id` is set so callers can time-travel.
        let mut start_step: u64 = 0;
        let mut prev_interrupt: Option<Interrupt> = None;
        if let Some(saver) = &self.checkpointer {
            let fetched = if cfg.checkpoint_id.is_some() {
                saver.get_at(&cfg).await
            } else {
                saver.get_latest(&cfg).await
            };
            match fetched {
                Ok(Some(rep)) => {
                    if let Err(e) = values.restore(rep.snapshot) {
                        let _ = reply.send(Err(e));
                        return;
                    }
                    start_step = rep.step;
                    prev_interrupt = rep.interrupt;
                }
                Ok(None) => {}
                Err(e) => {
                    let _ = reply.send(Err(e));
                    return;
                }
            }
        }

        // Apply seed input if not resuming.
        if start_step == 0 {
            if let Err(e) = values.seed(input.clone()) {
                let _ = reply.send(Err(e));
                return;
            }
        }

        let interrupt_after_remaining: HashSet<String> =
            self.topology.cfg.interrupt_after.iter().cloned().collect();
        self.state = RunState {
            values: Some(values),
            cfg,
            step: start_step,
            pending: HashSet::new(),
            in_flight_writes: Vec::new(),
            next_targets: VecDeque::new(),
            reply: Some(reply),
            aborted: false,
            interrupt: None,
            parent_command: None,
            interrupt_after_remaining,
        };

        // If we are resuming after an interrupt, route resume value to that node.
        if let (Some(interrupt), Some(rv)) = (prev_interrupt.clone(), resume) {
            self.state.next_targets.push_back(DispatchTarget {
                node: interrupt.node.clone(),
                input_override: Some(rv),
            });
        } else {
            // Fresh run (step 0) or a re-invoke against a completed thread:
            // start from the graph's entry targets with the current (possibly
            // restored) channel values.
            for t in &self.topology.entry_targets {
                if t == END {
                    continue;
                }
                self.state.next_targets.push_back(DispatchTarget {
                    node: t.clone(),
                    input_override: None,
                });
            }
        }

        self.advance_or_finish(ctx).await;
    }

    async fn on_node_done(
        &mut self,
        ctx: &mut Context<Self>,
        task_id: String,
        node: String,
        result: GraphResult<NodeOutput>,
    ) {
        if !self.state.pending.remove(&task_id) {
            return; // stale or aborted
        }
        match result {
            Err(e) => {
                self.state.aborted = true;
                if let Some(reply) = self.state.reply.take() {
                    let _ = reply.send(Err(e));
                }
                return;
            }
            Ok(NodeOutput::Interrupted(intr)) => {
                self.state.interrupt = Some(intr);
            }
            Ok(out) => {
                // Persist per-task "pending writes" immediately (upstream
                // durability guarantee so crashed fan-out can be replayed).
                self.persist_pending_writes(&task_id, &out).await;
                self.state.in_flight_writes.push((node.clone(), out));
            }
        }
        // When all dispatched nodes have responded, run Update phase.
        if self.state.pending.is_empty() {
            self.run_update_phase(ctx).await;
        }
    }

    /// Forward a node's writes (`NodeOutput::Update` or `Command.update`) to
    /// the checkpointer's per-task ledger, if one is attached.
    async fn persist_pending_writes(&self, task_id: &str, out: &NodeOutput) {
        let Some(saver) = &self.checkpointer else { return };
        let writes: Vec<(String, Value)> = match out {
            NodeOutput::Update(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            NodeOutput::Command(c) => {
                c.update.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            }
            _ => Vec::new(),
        };
        if writes.is_empty() {
            return;
        }
        if let Err(e) = saver.put_writes(&self.state.cfg, task_id, &writes).await {
            tracing::warn!(task_id, error = %e, "put_writes failed");
        }
    }

    /// Write a checkpoint that only captures the current channel state +
    /// pending interrupt (no per-node updates), used for static
    /// `interrupt_before` breakpoints.
    async fn persist_interrupt_checkpoint(&self) {
        let Some(saver) = &self.checkpointer else { return };
        let Some(values) = &self.state.values else { return };
        let snap = values.snapshot();
        let post_values = values.snapshot_values();
        if let Err(e) = saver
            .put_step(
                &self.state.cfg,
                self.state.step,
                &post_values,
                &snap,
                &[],
                self.state.interrupt.as_ref(),
            )
            .await
        {
            tracing::warn!(error = %e, "interrupt-before checkpoint failed");
        }
    }

    /// Either dispatch the next batch of nodes or finish the run.
    async fn advance_or_finish(&mut self, ctx: &mut Context<Self>) {
        if self.state.aborted {
            return;
        }
        if self.state.next_targets.is_empty() {
            self.finish_run().await;
            return;
        }
        let limit = self.effective_recursion_limit();
        if self.state.step >= limit as u64 {
            if let Some(reply) = self.state.reply.take() {
                let _ = reply.send(Err(GraphError::Recursion { limit }));
            }
            return;
        }

        // Static breakpoint: `interrupt_before`. If any queued target is in
        // the break-list, surface an Interrupt and pause the run *before* it
        // executes. We pick the first match for payload stability.
        if !self.topology.cfg.interrupt_before.is_empty() {
            if let Some(hit) = self
                .state
                .next_targets
                .iter()
                .find(|t| self.topology.cfg.interrupt_before.iter().any(|n| n == &t.node))
            {
                let node = hit.node.clone();
                self.state.interrupt = Some(Interrupt::new(
                    node.clone(),
                    serde_json::json!({"static": "before", "node": node}),
                ));
                // Persist the pause-state (restart will resume from here).
                self.persist_interrupt_checkpoint().await;
                self.finish_run().await;
                return;
            }
        }

        self.state.step += 1;
        if let Some(values) = &self.state.values {
            values.begin_step();
        }

        let targets: Vec<DispatchTarget> = self.state.next_targets.drain(..).collect();
        let values_map = self
            .state
            .values
            .as_ref()
            .map(|v| v.snapshot_values())
            .unwrap_or_default();

        self.publish_debug(format!("plan step={} targets={:?}", self.state.step, targets));

        for (i, tgt) in targets.into_iter().enumerate() {
            if tgt.node == END {
                continue;
            }
            let Some(node) = self.topology.nodes.get(&tgt.node) else {
                self.state.aborted = true;
                if let Some(reply) = self.state.reply.take() {
                    let _ = reply.send(Err(GraphError::UnknownNode(tgt.node.clone())));
                }
                return;
            };
            let node_kind = clone_node(node);
            let task_id = format!("{}-{}-{}", tgt.node, self.state.step, i);
            self.state.pending.insert(task_id.clone());

            let input = if let Some(ov) = tgt.input_override.clone() {
                let mut m = values_map.clone();
                m.insert("__send_arg__".into(), ov);
                m
            } else {
                values_map.clone()
            };
            let self_ref = ctx.self_ref().clone();
            let node_name = tgt.node.clone();
            let writer = crate::stream::StreamWriter::new(
                (*self.stream_bus).clone(),
                self.state.step,
                node_name.clone(),
            );
            let retry = self.topology.retries.get(&node_name).cloned();
            let cache_policy = self.topology.caches.get(&node_name).cloned();
            let cache = self.cache.clone();
            // astream_events v2: emit on_chain_start before dispatch.
            self.stream_bus.publish(StreamEvent::OnChainStart {
                step: self.state.step,
                node: node_name.clone(),
                run_id: self.state.cfg.run_id.clone(),
                tags: self.state.cfg.tags.clone(),
                namespace: Vec::new(),
            });
            let store = self.store.clone();
            tokio::spawn(async move {
                // Cache read (if policy attached + entry fresh).
                if let (Some(pol), Some(cache)) = (cache_policy.as_ref(), cache.as_ref()) {
                    let key = (pol.key_func)(&input);
                    let now = std::time::Instant::now();
                    let hit = {
                        let g = cache.read();
                        g.get(&(node_name.clone(), key.clone())).and_then(|e| {
                            match e.expires_at {
                                Some(exp) if exp <= now => None,
                                _ => Some(e.value.clone()),
                            }
                        })
                    };
                    if let Some(values) = hit {
                        let result = Ok(NodeOutput::Update(values));
                        self_ref.tell(CoordMsg::NodeDone { task_id, node: node_name, result });
                        return;
                    }
                    let result =
                        invoke_with_retry(node_kind, input.clone(), writer, retry, store).await;
                    // Cache successful `Update` outputs only (not Command /
                    // Send / Interrupt, which carry control-flow state).
                    if let Ok(NodeOutput::Update(ref values)) = result {
                        let expires_at = pol
                            .ttl_seconds
                            .map(|s| now + std::time::Duration::from_secs(s));
                        cache.write().insert(
                            (node_name.clone(), key),
                            crate::graph::CacheEntry { value: values.clone(), expires_at },
                        );
                    }
                    self_ref.tell(CoordMsg::NodeDone { task_id, node: node_name, result });
                    return;
                }
                let result =
                    invoke_with_retry(node_kind, input, writer, retry, store).await;
                self_ref.tell(CoordMsg::NodeDone { task_id, node: node_name, result });
            });
        }

        if let Some(values) = &self.state.values {
            values.ack_all_planned();
        }

        if self.state.pending.is_empty() {
            // Dispatched only END (or no-ops); finish.
            self.finish_run().await;
        }
    }

    /// Fold `cfg.recursion_limit`, `topology.cfg.recursion_limit`, and the
    /// upstream default (25) in that precedence order.
    fn effective_recursion_limit(&self) -> u32 {
        if let Some(n) = self.state.cfg.recursion_limit {
            return n;
        }
        if let Some(n) = self.topology.cfg.recursion_limit {
            return n;
        }
        25
    }

    /// Aggregate node outputs, emit stream events, persist a checkpoint,
    /// then plan the next batch.
    async fn run_update_phase(&mut self, ctx: &mut Context<Self>) {
        let in_flight = std::mem::take(&mut self.state.in_flight_writes);
        let values = self
            .state
            .values
            .as_ref()
            .expect("values must exist during a run")
            .clone();
        let mut per_node_updates: Vec<(String, BTreeMap<String, Value>)> = Vec::new();
        let mut next_targets: VecDeque<DispatchTarget> = VecDeque::new();

        // Process node outputs. We need pre-computed routes BEFORE mutating
        // values to mirror the upstream behaviour where conditional routers
        // see the *post-update* state — so we apply writes then route.
        let mut all_writes: Vec<(String, Value)> = Vec::new();
        let mut commands_by_node: Vec<(String, Option<Command>, BTreeMap<String, Value>)> =
            Vec::new();
        for (node, out) in in_flight.into_iter() {
            let (cmd, update) = match out {
                NodeOutput::Update(map) => (None, map),
                NodeOutput::Command(cmd) => (Some(cmd), BTreeMap::new()),
                NodeOutput::Halt => (None, BTreeMap::new()),
                NodeOutput::Interrupted(intr) => {
                    self.state.interrupt = Some(intr);
                    (None, BTreeMap::new())
                }
            };
            commands_by_node.push((node.clone(), cmd, update));
        }

        // Apply writes first (Command.update + plain update map).
        for (node, cmd, update) in &commands_by_node {
            let mut combined = update.clone();
            if let Some(c) = cmd {
                for (k, v) in &c.update {
                    combined.insert(k.clone(), v.clone());
                }
            }
            for (k, v) in &combined {
                all_writes.push((k.clone(), v.clone()));
            }
            per_node_updates.push((node.clone(), combined));
        }
        if let Err(e) = values.apply_writes(all_writes) {
            self.state.aborted = true;
            if let Some(reply) = self.state.reply.take() {
                let _ = reply.send(Err(e));
            }
            return;
        }

        // Publish per-node update + values events.
        let post_values = values.snapshot_values();
        for (node, update) in &per_node_updates {
            self.stream_bus.publish(StreamEvent::Updates {
                step: self.state.step,
                node: node.clone(),
                update: update.clone(),
                namespace: Vec::new(),
            });
            // astream_events v2: emit synthetic on_chain_end pairs for
            // observers that only want fine-grained events.
            self.stream_bus.publish(StreamEvent::OnChainEnd {
                step: self.state.step,
                node: node.clone(),
                run_id: self.state.cfg.run_id.clone(),
                output: serde_json::to_value(update).unwrap_or(Value::Null),
                namespace: Vec::new(),
            });
        }
        self.stream_bus.publish(StreamEvent::Values {
            step: self.state.step,
            values: post_values.clone(),
            namespace: Vec::new(),
        });

        // Now route — conditional routers see post-update state.
        let mut hit_interrupt_after: Option<String> = None;
        for (node, cmd, _update) in &commands_by_node {
            // Command targeting parent graph (`Command(graph="PARENT")`) does
            // NOT route within this graph; we stash it for the subgraph
            // adapter to propagate upward at finish.
            if let Some(c) = cmd {
                if c.graph.as_deref() == Some("PARENT") {
                    let mut escape = c.clone();
                    escape.graph = None; // consumed
                    self.state.parent_command = Some(escape);
                    continue;
                }
            }
            // Command-driven explicit goto first.
            let nexts = self.topology.next_targets(node, cmd.as_ref(), &post_values);
            for n in nexts {
                if n == END {
                    continue;
                }
                next_targets.push_back(DispatchTarget { node: n, input_override: None });
            }
            // `Command.send` adds extra fan-out targets in next step.
            if let Some(c) = cmd {
                for s in &c.send {
                    next_targets.push_back(DispatchTarget {
                        node: s.node.clone(),
                        input_override: Some(s.arg.clone()),
                    });
                }
            }

            // Static breakpoint: `interrupt_after`. Trigger after the writes
            // have been applied so the caller can inspect the post-update
            // values via `get_state`.
            if hit_interrupt_after.is_none()
                && self.state.interrupt_after_remaining.contains(node)
            {
                hit_interrupt_after = Some(node.clone());
            }
        }
        if let Some(node) = hit_interrupt_after {
            self.state.interrupt_after_remaining.remove(&node);
            self.state.interrupt = Some(Interrupt::new(
                node.clone(),
                serde_json::json!({"static": "after", "node": node}),
            ));
        }

        // Persist checkpoint. `Durability::Exit` defers writes to finish_run.
        if !matches!(self.topology.cfg.durability, crate::graph::Durability::Exit) {
            if let Some(saver) = &self.checkpointer {
                let snap = values.snapshot();
                let cfg = self.state.cfg.clone();
                let step = self.state.step;
                let updates = per_node_updates.clone();
                let intr = self.state.interrupt.clone();
                let post = post_values.clone();
                let saver = saver.clone();
                let snap_for_sync = snap.clone();
                match self.topology.cfg.durability {
                    crate::graph::Durability::Sync => {
                        if let Err(e) = saver
                            .put_step(&cfg, step, &post, &snap_for_sync, &updates, intr.as_ref())
                            .await
                        {
                            self.state.aborted = true;
                            if let Some(reply) = self.state.reply.take() {
                                let _ = reply.send(Err(e));
                            }
                            return;
                        }
                    }
                    crate::graph::Durability::Async => {
                        tokio::spawn(async move {
                            if let Err(e) = saver
                                .put_step(&cfg, step, &post, &snap, &updates, intr.as_ref())
                                .await
                            {
                                tracing::warn!(error = %e, "async checkpoint failed");
                            }
                        });
                    }
                    crate::graph::Durability::Exit => {}
                }
            }
        }

        // If a node interrupted (user or static), surface immediately.
        if self.state.interrupt.is_some() {
            self.finish_run().await;
            return;
        }

        self.state.next_targets = next_targets;
        self.advance_or_finish(ctx).await;
    }

    async fn finish_run(&mut self) {
        // `Durability::Exit`: persist the final snapshot once on the way out.
        if matches!(self.topology.cfg.durability, crate::graph::Durability::Exit) {
            if let (Some(saver), Some(values)) =
                (&self.checkpointer, self.state.values.as_ref())
            {
                let snap = values.snapshot();
                let post_values = values.snapshot_values();
                if let Err(e) = saver
                    .put_step(
                        &self.state.cfg,
                        self.state.step,
                        &post_values,
                        &snap,
                        &[],
                        self.state.interrupt.as_ref(),
                    )
                    .await
                {
                    tracing::warn!(error = %e, "exit-durability checkpoint failed");
                }
            }
        }
        let values = self
            .state
            .values
            .as_ref()
            .map(|v| v.snapshot_values())
            .unwrap_or_default();
        let interrupted = self.state.interrupt.take();
        let parent_command = self.state.parent_command.take();
        let result = RunResult {
            values,
            interrupted,
            steps: self.state.step,
            parent_command,
        };
        if let Some(reply) = self.state.reply.take() {
            let _ = reply.send(Ok(result));
        }
    }

    fn publish_debug(&self, msg: String) {
        if self.topology.cfg.debug {
            self.stream_bus.publish(StreamEvent::Debug {
                step: self.state.step,
                payload: serde_json::Value::String(msg),
                namespace: Vec::new(),
            });
        }
    }
}

fn clone_node(n: &NodeKind) -> NodeKind {
    match n {
        NodeKind::Rust(f) => NodeKind::Rust(f.clone()),
        NodeKind::Python(p) => NodeKind::Python(p.clone()),
        NodeKind::Subgraph(s) => NodeKind::Subgraph(s.clone()),
    }
}

/// Invoke a node honoring its optional [`RetryPolicy`]. The task-local
/// `StreamWriter` and (optionally) `CURRENT_STORE` are installed once and
/// re-used across retries so any streamed chunks / store operations from
/// partial attempts remain correctly attributed.
async fn invoke_with_retry(
    node_kind: NodeKind,
    input: BTreeMap<String, Value>,
    writer: crate::stream::StreamWriter,
    retry: Option<crate::graph::RetryPolicy>,
    store: Option<Arc<dyn crate::context::StoreAccessor>>,
) -> GraphResult<NodeOutput> {
    let attempts = retry
        .as_ref()
        .map(|r| r.max_attempts.max(1))
        .unwrap_or(1);
    let backoff_ms = retry.as_ref().map(|r| r.backoff_ms).unwrap_or(0);
    let mut last_err: Option<GraphError> = None;
    for attempt in 0..attempts {
        let input_i = input.clone();
        let node = clone_node_kind(&node_kind);
        let writer_i = writer.clone();
        let store_i = store.clone();
        let res = crate::stream::CURRENT_WRITER
            .scope(writer_i, async move {
                match store_i {
                    Some(s) => {
                        crate::context::CURRENT_STORE
                            .scope(s, async move { node.invoke(input_i).await })
                            .await
                    }
                    None => node.invoke(input_i).await,
                }
            })
            .await;
        match res {
            Ok(out) => return Ok(out),
            Err(e) => {
                last_err = Some(e);
                if attempt + 1 < attempts && backoff_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| GraphError::other("retry policy exhausted with no error")))
}

fn clone_node_kind(n: &NodeKind) -> NodeKind {
    clone_node(n)
}

// suppress unused warnings on optional helpers
const _: fn() = || {
    let _: Option<HashMap<String, ()>> = None;
    let _: Option<StreamMode> = None;
    let _: Option<GraphSend> = None;
    let _ = START;
};
