//! Runtime benchmark: measures embedding throughput per runtime.
//!
//! Usage: `cargo run --release -p kbcli-tests --bin run_runtime_bench`
//!
//! Each enabled runtime embeds a fixed batch of synthetic prompts. Results
//! are printed as JSON to stdout so a wrapper script can aggregate them.

use std::time::Instant;

use kbcli_embed::{EmbeddingRuntime, HashRuntime, RuntimeConfig};

#[tokio::main]
async fn main() {
    let n = std::env::var("KBCLI_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256usize);
    let dim = std::env::var("KBCLI_BENCH_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256usize);
    let only = std::env::var("KBCLI_BENCH_RUNTIME").ok();
    let want = |name: &str| only.as_deref().map_or(true, |o| o == name);

    let docs = kbcli_tests::corpus::seeded_corpus(n, 42);
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
    let warmup: Vec<&str> = texts.iter().take(4).copied().collect();

    let mut results = Vec::new();

    if want("hash") {
        results.push(
            load_and_bench("hash", dim, &warmup, &texts, |cfg| async move {
                Ok::<_, kbcli_core::Error>(HashRuntime::new(cfg.matryoshka_dim.unwrap_or(dim)))
            })
            .await,
        );
    }

    #[cfg(feature = "model-llama")]
    if want("llama") {
        results.push(
            load_and_bench("llama", dim, &warmup, &texts, |cfg| async move {
                kbcli_embed_llama::LlamaRuntime::new(cfg).await
            })
            .await,
        );
    }

    let out = serde_json::json!({
        "kind": "runtime_bench",
        "n": n,
        "dim": dim,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

async fn load_and_bench<R, F, Fut>(
    name: &str,
    dim: usize,
    warmup: &[&str],
    texts: &[&str],
    ctor: F,
) -> serde_json::Value
where
    R: EmbeddingRuntime,
    F: FnOnce(RuntimeConfig) -> Fut,
    Fut: std::future::Future<Output = Result<R, kbcli_core::Error>>,
{
    eprintln!("[bench] loading runtime: {name}");
    let load_start = Instant::now();
    let cfg = RuntimeConfig {
        matryoshka_dim: Some(dim),
        ..Default::default()
    };
    let rt = match ctor(cfg).await {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("[bench] {name}: load failed: {e}");
            return serde_json::json!({
                "runtime": name,
                "ok": false,
                "stage": "load",
                "error": e.to_string(),
            });
        }
    };
    let load_ms = load_start.elapsed().as_millis() as u64;
    eprintln!("[bench] {name}: loaded in {load_ms} ms; running warmup");

    if let Err(e) = rt.embed_batch(warmup).await {
        eprintln!("[bench] {name}: warmup failed: {e}");
        return serde_json::json!({
            "runtime": rt.name(),
            "dim": rt.dim(),
            "ok": false,
            "stage": "warmup",
            "load_ms": load_ms,
            "error": e.to_string(),
        });
    }
    eprintln!(
        "[bench] {name}: warmup ok; embedding {} texts at dim={}",
        texts.len(),
        rt.dim()
    );

    let started = Instant::now();
    let micro = std::env::var("KBCLI_BENCH_MICRO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32usize);
    let mut total: usize = 0;
    let mut last_err: Option<kbcli_core::Error> = None;
    for chunk in texts.chunks(micro) {
        match rt.embed_batch(chunk).await {
            Ok(v) => total += v.len(),
            Err(e) => {
                last_err = Some(e);
                break;
            }
        }
    }
    let elapsed = started.elapsed();
    if let Some(e) = last_err {
        eprintln!("[bench] {name}: bench failed after {total} texts: {e}");
        return serde_json::json!({
            "runtime": rt.name(),
            "ok": false,
            "stage": "bench",
            "load_ms": load_ms,
            "n_done": total,
            "error": e.to_string(),
        });
    }
    let qps = (total as f64) / elapsed.as_secs_f64();
    eprintln!(
        "[bench] {name}: {total} texts in {} ms ({:.1} qps, micro={micro})",
        elapsed.as_millis(),
        qps
    );
    serde_json::json!({
        "runtime": rt.name(),
        "dim": rt.dim(),
        "ok": true,
        "n": total,
        "load_ms": load_ms,
        "elapsed_ms": elapsed.as_millis() as u64,
        "qps": qps,
        "micro_batch": micro,
    })
}
