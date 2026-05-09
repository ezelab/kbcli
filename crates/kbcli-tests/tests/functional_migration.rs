//! Migration / re-open behavior.

use kbcli_core::{DocId, MetaValue};
use kbcli_store::{StoreConfig, VectorStore};

use kbcli_tests::runners;

#[tokio::test]
async fn config_persists_across_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("p.db");
    let cfg = kbcli_store::StoreConfig {
        embed_dim: 48,
        chunk_size: 64,
        chunk_overlap: 8,
        runtime_name: "hash".into(),
        model_id: "hash".into(),
    };
    {
        let store = kbcli_store_sqlite::SqliteStore::open(path.clone(), &cfg)
            .await
            .unwrap();
        store.migrate().await.unwrap();
        store.put_config(&cfg).await.unwrap();

        // Round-trip a doc via the harness-style helper to exercise insert.
        let h = runners::Harness {
            _dir: tempfile::TempDir::new().unwrap(), // unused, kept for type
            path: path.clone(),
            store: std::sync::Arc::new(
                kbcli_store_sqlite::SqliteStore::open(path.clone(), &cfg)
                    .await
                    .unwrap(),
            ),
            runtime: std::sync::Arc::new(kbcli_embed::HashRuntime::new(48)),
            chunker: kbcli_embed::Chunker::new(kbcli_embed::ChunkConfig {
                size: 64,
                overlap: 8,
            })
            .unwrap(),
            config: cfg.clone(),
        };
        runners::ingest_text(
            &h,
            "d1",
            "the only document here",
            &[("lang", MetaValue::Str("rust".into()))],
        )
        .await
        .unwrap();
    }

    // Re-open with default StoreConfig and check the persisted one is read back.
    let cfg_default = StoreConfig::default();
    let store = kbcli_store_sqlite::SqliteStore::open(path.clone(), &cfg_default)
        .await
        .unwrap();
    let stored = store.get_config().await.unwrap().expect("config persisted");
    assert_eq!(stored.embed_dim, 48);
    assert_eq!(stored.runtime_name, "hash");

    let got = store.get_doc(&DocId::new("d1")).await.unwrap();
    assert!(got.is_some());
}

#[tokio::test]
async fn migrate_is_idempotent() {
    let h = runners::sqlite_with_hash(32).await.unwrap();
    h.store.migrate().await.unwrap();
    h.store.migrate().await.unwrap();
}
