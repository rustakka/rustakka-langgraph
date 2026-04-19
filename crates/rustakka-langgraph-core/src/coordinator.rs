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
        Self { topology, stream_bus, checkpointer, state: RunState::default() }
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

        // Replay from checkpoint if available.
        let mut start_step: u64 = 0;
        let mut prev_interrupt: Option<Interrupt> = None;
        if let Some(saver) = &self.checkpointer {
            match saver.get_latest(&cfg).await {
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
                self.state.in_flight_writes.push((node.clone(), out));
            }
        }
        // When all dispatched nodes have responded, run Update phase.
        if self.state.pending.is_empty() {
            self.run_update_phase(ctx).await;
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
        let limit = self.state.cfg.effective_recursion_limit();
        if self.state.step >= limit as u64 {
            if let Some(reply) = self.state.reply.take() {
                let _ = reply.send(Err(GraphError::Recursion { limit }));
            }
            return;
        }
        self.state.step += 1;
        if let Some(values) = &self.state.values {
            values.begin_step();
        }

        // Plan phase: take all currently-queued targets as the dispatch set.
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
            tokio::spawn(async move {
                let result = crate::stream::CURRENT_WRITER
                    .scope(writer, node_kind.invoke(input))
                    .await;
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
            });
        }
        self.stream_bus.publish(StreamEvent::Values {
            step: self.state.step,
            values: post_values.clone(),
        });

        // Now route — conditional routers see post-update state.
        for (node, cmd, _update) in &commands_by_node {
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
        }

        // Persist checkpoint (best-effort; failure aborts run).
        if let Some(saver) = &self.checkpointer {
            let snap = values.snapshot();
            if let Err(e) = saver
                .put_step(
                    &self.state.cfg,
                    self.state.step,
                    &post_values,
                    &snap,
                    &per_node_updates,
                    self.state.interrupt.as_ref(),
                )
                .await
            {
                self.state.aborted = true;
                if let Some(reply) = self.state.reply.take() {
                    let _ = reply.send(Err(e));
                }
                return;
            }
        }

        // If a node interrupted, surface immediately and pause.
        if self.state.interrupt.is_some() {
            self.finish_run().await;
            return;
        }

        self.state.next_targets = next_targets;
        self.advance_or_finish(ctx).await;
    }

    async fn finish_run(&mut self) {
        let values = self
            .state
            .values
            .as_ref()
            .map(|v| v.snapshot_values())
            .unwrap_or_default();
        let interrupted = self.state.interrupt.take();
        let result = RunResult { values, interrupted, steps: self.state.step };
        if let Some(reply) = self.state.reply.take() {
            let _ = reply.send(Ok(result));
        }
    }

    fn publish_debug(&self, msg: String) {
        if self.topology.cfg.debug {
            self.stream_bus.publish(StreamEvent::Debug {
                step: self.state.step,
                payload: serde_json::Value::String(msg),
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

// suppress unused warnings on optional helpers
const _: fn() = || {
    let _: Option<HashMap<String, ()>> = None;
    let _: Option<StreamMode> = None;
    let _: Option<GraphSend> = None;
    let _ = START;
};
