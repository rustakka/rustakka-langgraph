//! Graph visualization helpers. Mirrors upstream
//! `CompiledStateGraph.get_graph().draw_mermaid()` / `draw_ascii()`.
//!
//! Conditional edges are rendered as a single dotted edge from the source to
//! every static branch target (when a `branches` map is declared) plus a
//! `<node>_cond` diamond when no branches are known. We don't evaluate the
//! router at render time — that would require a concrete state.

use crate::graph::{CompiledStateGraph, END, START};

/// Render a Mermaid `flowchart TD` representation of the compiled graph.
///
/// Reserved node names `__start__` / `__end__` are rendered as rounded
/// stadium nodes labelled `START` / `END` to match upstream's output.
pub fn draw_mermaid(graph: &CompiledStateGraph) -> String {
    let topo = graph.topology();
    let mut out = String::from("flowchart TD\n");
    out.push_str(&format!("    {}([START])\n", mermaid_id(START)));
    out.push_str(&format!("    {}([END])\n", mermaid_id(END)));
    for name in topo.nodes.keys() {
        out.push_str(&format!("    {}[\"{}\"]\n", mermaid_id(name), escape_label(name)));
    }
    for (src, tgt) in &topo.static_edges {
        out.push_str(&format!("    {} --> {}\n", mermaid_id(src), mermaid_id(tgt)));
    }
    for ce in &topo.conditional_edges {
        match &ce.branches {
            Some(map) => {
                for (label, tgt) in map {
                    out.push_str(&format!(
                        "    {} -. {} .-> {}\n",
                        mermaid_id(&ce.source),
                        escape_label(label),
                        mermaid_id(tgt),
                    ));
                }
            }
            None => {
                // Unknown branch destinations: emit a diamond placeholder so
                // readers can see *something* is dynamic.
                let diamond = format!("{}_cond", mermaid_id(&ce.source));
                out.push_str(&format!("    {}{{?}}\n", diamond));
                out.push_str(&format!("    {} -.-> {}\n", mermaid_id(&ce.source), diamond));
            }
        }
    }
    out
}

/// Render a minimal ASCII representation — one line per edge — useful for
/// logs / unit-test snapshots.
pub fn draw_ascii(graph: &CompiledStateGraph) -> String {
    let topo = graph.topology();
    let mut lines = Vec::new();
    lines.push("Nodes:".to_string());
    let mut names: Vec<&String> = topo.nodes.keys().collect();
    names.sort();
    for name in names {
        lines.push(format!("  - {}", name));
    }
    lines.push("Edges:".to_string());
    let mut edges = topo.static_edges.clone();
    edges.sort();
    for (src, tgt) in edges {
        lines.push(format!("  {} -> {}", src, tgt));
    }
    for ce in &topo.conditional_edges {
        match &ce.branches {
            Some(map) => {
                let mut pairs: Vec<_> = map.iter().collect();
                pairs.sort();
                for (label, tgt) in pairs {
                    lines.push(format!("  {} -[{}]-> {}", ce.source, label, tgt));
                }
            }
            None => lines.push(format!("  {} -[?]-> (dynamic)", ce.source)),
        }
    }
    lines.join("\n")
}

fn mermaid_id(s: &str) -> String {
    // Mermaid node ids must match `[A-Za-z0-9_]+`. Escape the reserved
    // `__start__` / `__end__` without collisions.
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('n');
    }
    out
}

fn escape_label(s: &str) -> String {
    s.replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CompileConfig, StateGraph};
    use crate::node::{NodeKind, NodeOutput};
    use crate::state::DynamicState;
    use std::collections::BTreeMap;

    async fn two_node_graph() -> CompiledStateGraph {
        let mut g = StateGraph::<DynamicState>::new();
        g.add_node(
            "a",
            NodeKind::from_fn(|_| async { Ok(NodeOutput::Update(BTreeMap::new())) }),
        )
        .unwrap();
        g.add_node(
            "b",
            NodeKind::from_fn(|_| async { Ok(NodeOutput::Update(BTreeMap::new())) }),
        )
        .unwrap();
        g.add_edge(START, "a");
        g.add_edge("a", "b");
        g.add_edge("b", END);
        g.compile(CompileConfig::default()).await.unwrap()
    }

    #[tokio::test]
    async fn mermaid_contains_all_nodes_and_edges() {
        let g = two_node_graph().await;
        let m = draw_mermaid(&g);
        assert!(m.starts_with("flowchart TD"));
        assert!(m.contains("([START])"));
        assert!(m.contains("([END])"));
        assert!(m.contains("\"a\""));
        assert!(m.contains("\"b\""));
        assert!(m.contains("__start__ --> a"));
        assert!(m.contains("a --> b"));
        assert!(m.contains("b --> __end__"));
    }

    #[tokio::test]
    async fn ascii_renders_nodes_and_edges() {
        let g = two_node_graph().await;
        let a = draw_ascii(&g);
        assert!(a.contains("- a"));
        assert!(a.contains("- b"));
        assert!(a.contains("a -> b"));
    }
}
