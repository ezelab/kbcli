//! Deterministic synthetic corpora.
//!
//! `seeded_corpus(n, seed)` generates `n` short pseudo-natural-language
//! documents from a fixed vocabulary, with reproducible KV metadata. These
//! are the inputs used by recall and storage benchmarks where we need a
//! known ground truth without downloading external data.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const VOCAB: &[&str] = &[
    "rust",
    "python",
    "go",
    "javascript",
    "language",
    "compiler",
    "performance",
    "memory",
    "safe",
    "fast",
    "thread",
    "async",
    "embedding",
    "vector",
    "search",
    "database",
    "query",
    "index",
    "tokenizer",
    "neural",
    "model",
    "matrix",
    "cosine",
    "similarity",
    "ranking",
    "lexical",
    "semantic",
    "hybrid",
    "filter",
    "metadata",
    "document",
    "chunk",
    "stream",
    "ingest",
    "binary",
    "release",
    "build",
    "test",
    "bench",
    "scalable",
];

/// One synthetic document.
#[derive(Clone, Debug)]
pub struct SynthDoc {
    pub id: String,
    pub text: String,
    pub lang: &'static str,
    pub bucket: i64,
}

/// Generate `n` synthetic documents with the given seed.
pub fn seeded_corpus(n: usize, seed: u64) -> Vec<SynthDoc> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let langs = ["rust", "python", "go", "js"];
    for i in 0..n {
        let len = rng.gen_range(8..32);
        let words: Vec<&str> = (0..len)
            .map(|_| VOCAB[rng.gen_range(0..VOCAB.len())])
            .collect();
        out.push(SynthDoc {
            id: format!("d{i}"),
            text: words.join(" "),
            lang: langs[rng.gen_range(0..langs.len())],
            bucket: rng.gen_range(0..10),
        });
    }
    out
}
