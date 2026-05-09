//! kbcli-embed: `EmbeddingRuntime` trait + tokenizer/chunker.
//!
//! L1 of the layered workspace. Defines the async trait that every
//! embedding-runtime impl must implement, plus the runtime-agnostic
//! token-aware chunker. Depends only on `kbcli-core`.

pub use kbcli_core as core;

mod chunker;
mod hash_runtime;
mod runtime;

pub use chunker::{ChunkConfig, Chunker};
pub use hash_runtime::HashRuntime;
pub use runtime::{cosine, l2_normalize, matryoshka_truncate, EmbeddingRuntime, RuntimeConfig};
