//! Runtime-agnostic chunker.
//!
//! kbcli does not require a model-specific tokenizer at the chunker layer;
//! we use a Unicode-word-based approximation that is good enough for
//! splitting and yields stable, deterministic chunk boundaries. Each
//! `EmbeddingRuntime` impl is responsible for re-tokenizing the chunk text
//! with the model's actual tokenizer when computing embeddings.

use unicode_segmentation::UnicodeSegmentation;

use kbcli_core::{Chunk, DocId, Error, Result};

/// Chunking configuration.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Maximum tokens per chunk.
    pub size: u32,
    /// Number of overlapping tokens between consecutive chunks.
    pub overlap: u32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            size: 512,
            overlap: 64,
        }
    }
}

impl ChunkConfig {
    pub fn validate(&self) -> Result<()> {
        if self.size == 0 {
            return Err(Error::invalid("chunk size must be > 0"));
        }
        if self.overlap >= self.size {
            return Err(Error::invalid("chunk overlap must be < chunk size"));
        }
        Ok(())
    }
}

/// Unicode-word-based chunker.
#[derive(Debug, Clone, Copy)]
pub struct Chunker {
    cfg: ChunkConfig,
}

impl Chunker {
    pub fn new(cfg: ChunkConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    pub fn config(&self) -> ChunkConfig {
        self.cfg
    }

    /// Split `text` into chunks for `doc_id`.
    pub fn chunk<I: Into<DocId>>(&self, doc_id: I, text: &str) -> Vec<Chunk> {
        let doc_id = doc_id.into();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return vec![];
        }

        // Token model: Unicode word-bounds. We track byte offsets for each
        // token so we can reconstruct the chunk's text without copying.
        let tokens: Vec<(usize, &str)> = trimmed
            .split_word_bound_indices()
            .filter(|(_, w)| !w.chars().all(char::is_whitespace))
            .collect();

        if tokens.is_empty() {
            return vec![];
        }

        let size = self.cfg.size as usize;
        let overlap = self.cfg.overlap as usize;
        let stride = size.saturating_sub(overlap).max(1);

        let mut out = Vec::new();
        let mut start = 0usize;
        let mut ord: u32 = 0;
        while start < tokens.len() {
            let end = (start + size).min(tokens.len());

            // Prefer breaking on a sentence boundary near `end` for cleaner
            // chunks: scan back up to `overlap` tokens looking for one that
            // ends in a sentence-final punctuation mark.
            let mut break_at = end;
            if end < tokens.len() && overlap > 0 {
                let lo = end.saturating_sub(overlap);
                for i in (lo..end).rev() {
                    let tok = tokens[i].1.trim_end();
                    if tok.ends_with('.') || tok.ends_with('!') || tok.ends_with('?') {
                        break_at = i + 1;
                        break;
                    }
                }
            }

            let s_byte = tokens[start].0;
            let e_byte = if break_at < tokens.len() {
                tokens[break_at].0
            } else {
                trimmed.len()
            };

            let slice = trimmed[s_byte..e_byte].trim();
            if !slice.is_empty() {
                let mut c = Chunk::new(doc_id.clone(), ord, slice);
                // token_count is a better count than whitespace-split:
                c.token_count = (break_at - start) as u32;
                out.push(c);
                ord += 1;
            }

            if break_at >= tokens.len() {
                break;
            }
            start = break_at
                .saturating_sub(overlap)
                .max(start + stride.min(break_at - start));
            if start >= tokens.len() {
                break;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_no_chunks() {
        let chunker = Chunker::new(ChunkConfig::default()).unwrap();
        assert!(chunker.chunk("d1", "").is_empty());
        assert!(chunker.chunk("d1", "   \n  ").is_empty());
    }

    #[test]
    fn short_text_yields_one_chunk() {
        let chunker = Chunker::new(ChunkConfig::default()).unwrap();
        let chunks = chunker.chunk("d1", "Rust is a systems programming language.");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Rust"));
        assert!(chunks[0].token_count > 0);
    }

    #[test]
    fn long_text_yields_multiple_chunks_with_overlap() {
        let cfg = ChunkConfig {
            size: 10,
            overlap: 3,
        };
        let chunker = Chunker::new(cfg).unwrap();
        let words: Vec<String> = (0..50).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunks = chunker.chunk("d1", &text);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.token_count <= 10);
        }
        // The last chunk should contain the last word.
        assert!(chunks.last().unwrap().text.contains("word49"));
    }

    #[test]
    fn validates_overlap() {
        assert!(ChunkConfig {
            size: 10,
            overlap: 10
        }
        .validate()
        .is_err());
        assert!(ChunkConfig {
            size: 0,
            overlap: 0
        }
        .validate()
        .is_err());
    }

    #[test]
    fn unicode_chunking() {
        let chunker = Chunker::new(ChunkConfig {
            size: 5,
            overlap: 1,
        })
        .unwrap();
        let chunks = chunker.chunk("d1", "café résumé naïve über fünf sechs sieben acht");
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.text.contains("café")));
    }
}
