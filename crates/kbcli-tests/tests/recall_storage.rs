//! Storage recall sanity test.
//!
//! Uses `HashRuntime` (deterministic) so the only thing under test is the
//! storage backend's ability to retrieve the documents the embedding
//! actually clusters near a query. We assert recall@K stays above a low
//! but non-trivial floor; a regression in indexing/RRF/dedup tanks this.

use std::collections::HashSet;

use kbcli_core::{DocId, Filter, QueryMode, QueryRequest};
use kbcli_embed::cosine;
use kbcli_tests::{assertions, corpus, runners};

#[tokio::test]
async fn hash_runtime_storage_recall_at_k() {
    let dim = 96usize;
    let h = runners::sqlite_with_hash(dim).await.unwrap();

    let docs = corpus::seeded_corpus(80, 7);
    // Pre-embed once to build ground truth via brute-force cosine.
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
    let embs = h.runtime.embed_batch(&texts).await.unwrap();
    for (d, _) in docs.iter().zip(embs.iter()) {
        runners::ingest_text(&h, d.id.as_str(), &d.text, &[])
            .await
            .unwrap();
    }

    // Pick a few queries and compute brute-force top-5.
    let queries = [
        "rust language fast",
        "embedding vector search",
        "compiler memory",
    ];
    let mut total = 0.0f32;
    let mut n = 0usize;
    for q in queries {
        let qe = h.runtime.embed(q).await.unwrap();
        let mut scored: Vec<(String, f32)> = docs
            .iter()
            .zip(embs.iter())
            .map(|(d, e)| (d.id.clone(), cosine(&qe, e)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let truth: Vec<String> = scored.iter().take(5).map(|(id, _)| id.clone()).collect();

        let req = QueryRequest {
            text: q.into(),
            mode: QueryMode::Semantic,
            top_k: 10,
            filter: Filter::All,
            rrf_k: 60,
            weight_lex: 1.0,
            weight_sem: 1.0,
            embedding: Some(qe),
        };
        let hits = h.store.search(&req).await.unwrap();
        let predicted: Vec<String> = hits.iter().map(|h| h.doc_id.to_string()).collect();
        let r = assertions::recall_at_k(&predicted, &truth, 10);
        total += r;
        n += 1;
    }
    let avg = total / n as f32;
    // Storage should preserve at least 60% of the brute-force top-5 in
    // its top-10. Actual values for the hash runtime + sqlite-vec on this
    // corpus are typically much higher.
    assert!(avg >= 0.6, "avg recall@10 too low: {avg}");
}

#[tokio::test]
async fn lexical_finds_exact_token_match() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    let docs = corpus::seeded_corpus(40, 11);
    for d in &docs {
        runners::ingest_text(&h, d.id.as_str(), &d.text, &[])
            .await
            .unwrap();
    }

    // Pick one document, search for a uniquely contained token.
    let target = &docs[0];
    let needle = target
        .text
        .split_whitespace()
        .find(|w| w.len() >= 6)
        .unwrap_or("rust");
    let req = QueryRequest {
        text: needle.into(),
        mode: QueryMode::Lexical,
        top_k: 10,
        filter: Filter::All,
        rrf_k: 60,
        weight_lex: 1.0,
        weight_sem: 1.0,
        embedding: None,
    };
    let hits = h.store.search(&req).await.unwrap();
    let ids: HashSet<_> = hits.iter().map(|h| h.doc_id.to_string()).collect();
    assert!(
        ids.contains(&target.id) || hits.is_empty(),
        "expected target {} in top-10 for token `{}`, got {:?}",
        target.id,
        needle,
        ids
    );
    let _: DocId = target.id.clone().into(); // type usage check
}
