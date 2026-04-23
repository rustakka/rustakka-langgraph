//! Minimal embedding hook used by semantic-search-enabled stores.
//!
//! Upstream LangGraph lets callers pass a full `Embeddings` object (e.g.
//! `OpenAIEmbeddings`). Keeping that surface pluggable here without pulling a
//! provider dep into this crate, we just expose an [`Embedder`] trait. A
//! deterministic [`HashingEmbedder`] is included for tests and CI.
//!
//! Real provider-backed embedders live in `rustakka-langgraph-providers`
//! (or user crates) and need only implement [`Embedder`].

use async_trait::async_trait;

use rustakka_langgraph_core::errors::GraphResult;

/// Produce a dense vector for a piece of text.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    fn dims(&self) -> usize;
    async fn embed(&self, text: &str) -> GraphResult<Vec<f32>>;
}

/// Deterministic, dependency-free hashing embedder. Not semantically
/// meaningful but good enough for tests and as a default so search works
/// out of the box.
pub struct HashingEmbedder {
    dims: usize,
}

impl HashingEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims: dims.max(1) }
    }
}

#[async_trait]
impl Embedder for HashingEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }
    async fn embed(&self, text: &str) -> GraphResult<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut v = vec![0f32; self.dims];
        // Token-level DJB2-style hashing into random bucket + sign.
        for tok in text.split_whitespace() {
            let mut h = DefaultHasher::new();
            tok.to_ascii_lowercase().hash(&mut h);
            let hv = h.finish();
            let idx = (hv as usize) % self.dims;
            let sign = if (hv >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[idx] += sign;
        }
        // L2 normalize.
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }
}

/// Cosine similarity between two equal-length vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}
