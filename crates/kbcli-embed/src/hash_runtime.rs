//! Deterministic, dependency-free `EmbeddingRuntime` impl.
//!
//! [`HashRuntime`] hashes whitespace tokens into a fixed-size float vector.
//! It is **not** a real semantic embedder — it has no notion of meaning —
//! but it is fast, offline, and reproducible, which makes it perfect for:
//!
//! * functional and CLI tests in `kbcli-tests`,
//! * the out-of-the-box CLI experience before users build with
//!   `--features model-llama`,
//! * sanity-checking the storage / hybrid-search plumbing.
//!
//! It is **not** quality-acceptable for retrieval-quality bench tests —
//! those gate on a real embedding model.

use async_trait::async_trait;

use kbcli_core::Result;

use crate::runtime::{l2_normalize, EmbeddingRuntime};

/// A deterministic, hash-based embedding runtime.
///
/// Each whitespace token is hashed into the vector via a small set of
/// independent hash functions; the result is mean-pooled and L2-normalized
/// so cosine similarity behaves the same way as a real model.
pub struct HashRuntime {
    dim: usize,
}

impl HashRuntime {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }
}

impl Default for HashRuntime {
    fn default() -> Self {
        Self::new(128)
    }
}

#[async_trait]
impl EmbeddingRuntime for HashRuntime {
    fn name(&self) -> &'static str {
        "hash"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_input_tokens(&self) -> usize {
        usize::MAX
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(embed_one(t, self.dim));
        }
        Ok(out)
    }
}

fn embed_one(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let mut tokens = 0u32;
    for tok in text.to_lowercase().split_whitespace() {
        let h = fnv_hash(tok.as_bytes());
        // Splat across a few positions for a denser representation.
        let positions = [
            (h as usize) % dim,
            ((h >> 16) as usize) % dim,
            ((h >> 32) as usize) % dim,
            ((h >> 48) as usize) % dim,
        ];
        let signs = [
            if h & 1 == 0 { 1.0 } else { -1.0 },
            if h & 2 == 0 { 1.0 } else { -1.0 },
            if h & 4 == 0 { 1.0 } else { -1.0 },
            if h & 8 == 0 { 1.0 } else { -1.0 },
        ];
        for (p, s) in positions.iter().zip(signs.iter()) {
            v[*p] += *s;
        }
        tokens += 1;
    }
    if tokens == 0 {
        // Avoid the all-zero vector; pick a fixed unit direction.
        v[0] = 1.0;
        return v;
    }
    let inv = 1.0 / tokens as f32;
    for x in &mut v {
        *x *= inv;
    }
    l2_normalize(&mut v);
    v
}

/// FNV-1a 64-bit. Tiny, fast, and deterministic across platforms.
fn fnv_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_and_normalized() {
        let rt = HashRuntime::new(64);
        let a = rt.embed("hello world").await.unwrap();
        let b = rt.embed("hello world").await.unwrap();
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[tokio::test]
    async fn similar_strings_have_higher_cosine_than_unrelated() {
        let rt = HashRuntime::new(256);
        let v = rt
            .embed_batch(&[
                "the quick brown fox jumps over the lazy dog",
                "the fast brown fox jumps over the lazy dog",
                "completely unrelated text about quantum mechanics",
            ])
            .await
            .unwrap();
        let s_close = crate::runtime::cosine(&v[0], &v[1]);
        let s_far = crate::runtime::cosine(&v[0], &v[2]);
        assert!(s_close > s_far, "close={s_close} far={s_far}");
    }

    #[tokio::test]
    async fn empty_input_is_unit_vector() {
        let rt = HashRuntime::new(32);
        let v = rt.embed("").await.unwrap();
        assert_eq!(v.len(), 32);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
