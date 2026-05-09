//! Unicode + edge-case chunking tests.

use kbcli_embed::{ChunkConfig, Chunker};

#[test]
fn empty_yields_no_chunks() {
    let c = Chunker::new(ChunkConfig::default()).unwrap();
    assert!(c.chunk("d", "").is_empty());
    assert!(c.chunk("d", "   \n  ").is_empty());
}

#[test]
fn short_doc_one_chunk() {
    let c = Chunker::new(ChunkConfig::default()).unwrap();
    let v = c.chunk("d", "Rust is great.");
    assert_eq!(v.len(), 1);
    assert!(v[0].text.contains("Rust"));
}

#[test]
fn long_doc_overlap_invariant() {
    let c = Chunker::new(ChunkConfig {
        size: 10,
        overlap: 3,
    })
    .unwrap();
    let words: Vec<String> = (0..200).map(|i| format!("w{i}")).collect();
    let text = words.join(" ");
    let chunks = c.chunk("d", &text);
    assert!(chunks.len() > 1);
    for ch in &chunks {
        assert!(ch.token_count <= 10);
        assert!(ch.token_count > 0);
    }
    assert!(chunks.last().unwrap().text.contains("w199"));
    for w in &chunks
        .windows(2)
        .map(|w| (w[0].ord, w[1].ord))
        .collect::<Vec<_>>()
    {
        assert!(w.1 == w.0 + 1, "monotonic ord");
    }
}

#[test]
fn unicode_preserved() {
    let c = Chunker::new(ChunkConfig {
        size: 4,
        overlap: 1,
    })
    .unwrap();
    let chunks = c.chunk("d", "café résumé naïve über fünf 漢字 日本語 한국어");
    assert!(!chunks.is_empty());
    let joined = chunks
        .iter()
        .map(|c| c.text.clone())
        .collect::<Vec<_>>()
        .join(" ");
    for token in ["café", "résumé", "naïve", "über", "漢字", "日本語"] {
        assert!(joined.contains(token), "missing {token}");
    }
}

#[test]
fn invalid_overlap_rejected() {
    assert!(ChunkConfig {
        size: 5,
        overlap: 5
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
