//! Functional tests for the document lifecycle.
//!
//! These compose `HashRuntime` + `SqliteStore` directly (no CLI subprocess
//! invocation) so they run quickly and without external dependencies.

use kbcli_core::{DocId, Filter, MetaValue, QueryMode, QueryRequest};
use kbcli_tests::fixtures;
use kbcli_tests::runners;

#[tokio::test]
async fn round_trip_add_get_delete() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    let chunks = runners::ingest_text(&h, "d1", "hello world", &[])
        .await
        .unwrap();
    assert!(chunks >= 1);

    let got = h.store.get_doc(&DocId::new("d1")).await.unwrap();
    assert!(got.is_some());
    assert_eq!(got.unwrap().text, "hello world");

    assert!(h.store.delete_doc(&DocId::new("d1")).await.unwrap());
    assert!(h.store.get_doc(&DocId::new("d1")).await.unwrap().is_none());
}

#[tokio::test]
async fn upsert_replaces_chunks_and_text() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    runners::ingest_text(&h, "d1", "first version", &[])
        .await
        .unwrap();
    runners::ingest_text(&h, "d1", "second version with more words here", &[])
        .await
        .unwrap();
    let got = h.store.get_doc(&DocId::new("d1")).await.unwrap().unwrap();
    assert!(got.text.contains("second"));

    let info = h.store.info().await.unwrap();
    assert_eq!(info.doc_count, 1);
}

#[tokio::test]
async fn list_with_metadata_filter() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    for (id, text, lang, year) in fixtures::small_corpus() {
        runners::ingest_text(
            &h,
            id,
            text,
            &[
                ("lang", MetaValue::Str(lang.to_string())),
                ("year", MetaValue::Int(year)),
            ],
        )
        .await
        .unwrap();
    }

    let f = Filter::parse_cli("lang=rust").unwrap();
    let docs = h.store.list_docs(&f, 100, 0).await.unwrap();
    assert_eq!(docs.len(), 2, "should match both rust docs");
    for d in &docs {
        assert!(d.id.as_str().starts_with("rust"));
    }

    let f = Filter::parse_cli("year>=2010").unwrap();
    let docs = h.store.list_docs(&f, 100, 0).await.unwrap();
    let ids: Vec<_> = docs.iter().map(|d| d.id.to_string()).collect();
    assert!(ids.contains(&"rust1".to_string()));
    assert!(ids.contains(&"rust2".to_string()));
    assert!(ids.contains(&"ml2".to_string()));
    assert!(!ids.contains(&"py1".to_string()), "py1 (1991) excluded");
}

#[tokio::test]
async fn query_lexical_returns_relevant_doc() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    for (id, text, lang, year) in fixtures::small_corpus() {
        runners::ingest_text(
            &h,
            id,
            text,
            &[
                ("lang", MetaValue::Str(lang.to_string())),
                ("year", MetaValue::Int(year)),
            ],
        )
        .await
        .unwrap();
    }

    let req = QueryRequest {
        text: "embeddings".into(),
        mode: QueryMode::Lexical,
        top_k: 5,
        filter: Filter::All,
        rrf_k: 60,
        weight_lex: 1.0,
        weight_sem: 1.0,
        embedding: None,
    };
    let hits = h.store.search(&req).await.unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].doc_id.as_str(), "ml2", "ml2 talks about embeddings");
}

#[tokio::test]
async fn query_hybrid_uses_embedding() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    for (id, text, lang, _year) in fixtures::small_corpus() {
        runners::ingest_text(&h, id, text, &[("lang", MetaValue::Str(lang.to_string()))])
            .await
            .unwrap();
    }
    let q_emb = h.runtime.embed("language").await.unwrap();
    let req = QueryRequest {
        text: "language".into(),
        mode: QueryMode::Hybrid,
        top_k: 3,
        filter: Filter::parse_cli("lang=rust").unwrap(),
        rrf_k: 60,
        weight_lex: 1.0,
        weight_sem: 1.0,
        embedding: Some(q_emb),
    };
    let hits = h.store.search(&req).await.unwrap();
    assert!(!hits.is_empty());
    for h in &hits {
        assert!(h.doc_id.as_str().starts_with("rust"));
    }
}

#[tokio::test]
async fn semantic_query_requires_embedding() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    runners::ingest_text(&h, "d1", "hello world", &[])
        .await
        .unwrap();
    let req = QueryRequest {
        text: "hi".into(),
        mode: QueryMode::Semantic,
        top_k: 3,
        filter: Filter::All,
        rrf_k: 60,
        weight_lex: 1.0,
        weight_sem: 1.0,
        embedding: None,
    };
    let err = h.store.search(&req).await.unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("embedding"),
        "expected missing-embedding error, got: {err}"
    );
}

#[tokio::test]
async fn upsert_false_rejects_duplicate() {
    let h = runners::sqlite_with_hash(64).await.unwrap();
    let chunks: Vec<kbcli_core::Chunk> = h.chunker.chunk("d1", "hello world");
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embs = h.runtime.embed_batch(&texts).await.unwrap();
    let mut chunks = chunks;
    for (c, e) in chunks.iter_mut().zip(embs.into_iter()) {
        c.embedding = Some(e);
    }
    let doc = kbcli_core::Document::new("d1", "hello world");
    h.store.upsert_doc(&doc, &chunks, false).await.unwrap();

    // Second insert with upsert=false should conflict.
    let err = h.store.upsert_doc(&doc, &chunks, false).await.unwrap_err();
    assert!(
        format!("{err}").to_lowercase().contains("already exists"),
        "got: {err}"
    );
}
