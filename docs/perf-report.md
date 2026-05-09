# Performance report

## Shipped stack

After the bake-off documented below, the workspace ships a single
runtime/backend pair:

| Axis    | Impl                                              | Notes                                  |
| ------- | ------------------------------------------------- | -------------------------------------- |
| Runtime | **llama** (llama-cpp-2 + Metal, EmbeddingGemma Q8_0 GGUF) | ~28% faster than ONNX Runtime CPU on Apple Silicon |
| Storage | **sqlite-vec** (vanilla SQLite + sqlite-vec extension)    | Fastest query at every dim tested, ~22-37× smaller on disk than libSQL |

The default-feature build is offline-friendly: it pulls in the
deterministic `hash` runtime so `cargo build` works without `cmake` or
network. Real semantic embeddings are enabled with `--features model-llama`,
which links `llama-cpp-2` and downloads EmbeddingGemma weights on first use.

The four candidates discarded during the bake-off were removed from the
repository to keep the build surface small. A short post-mortem is at the
end of this document.

## Reproducing the bench

Headless bench binaries live in `kbcli-tests`:

```sh
# embedding throughput (hash + llama)
cargo run --release -p kbcli-tests --features model-llama \
    --bin run_runtime_bench

# index + query latency for sqlite-vec
cargo run --release -p kbcli-tests --bin run_storage_bench

# composed end-to-end timings
cargo run --release -p kbcli-tests --bin run_e2e_bench

# semantic-search quality (NDCG@10) on BEIR/SciFact
cargo run --release -p kbcli-tests --features model-llama \
    --bin run_search_bench
```

Reference numbers below come from a development workstation
(`aarch64-apple-darwin`, M-series, 1 process, no other workload).

### Embedding runtime — EmbeddingGemma 300m, dim=768, micro-batch=64, n=100,000

`KBCLI_BENCH_N=100000 KBCLI_BENCH_DIM=768 KBCLI_BENCH_MICRO=64`

| Runtime  | Model                                              | Load   | Embed 100k         | qps          |
| -------- | -------------------------------------------------- | ------ | ------------------ | ------------ |
| **llama** | `ggml-org/embeddinggemma-300M-GGUF` (Q8_0)        | 0.2 s  | **6 m 40 s**       | **249.8**    |
| hash     | (deterministic baseline)                           | 0      | 76 ms              | 1,301,624    |

`llama` is ~1.28× faster than `ort` CPU was on the same hardware before
`ort` was removed from the workspace.

### Storage backend — `sqlite-vec`, 100k docs, 100 hybrid queries

| Dim  | Index time     | Query (100q) | qps  | DB size  |
| ---- | -------------- | ------------ | ---- | -------- |
| 128  | 12 m 0 s       | 11.0 s       | 9.1  | 99.9 MiB |
| 256  | 11 m 53 s      | 11.9 s       | 8.4  | 151 MiB  |
| 768  | ~12 m          | ~12 s        | ~8   | ~440 MiB |

Brute-force cosine over a contiguous `vec0` table absorbs into M-series
SIMD comfortably. Production deployments at >100k docs would generally
reach for an ANN structure, but for the cross-platform single-binary
target we ship, `vec0` is a great fit (small, fast queries, simple
on-disk format).

### Semantic-search quality — BEIR/SciFact, NDCG@10

`run_search_bench` builds a real EmbeddingGemma index over BEIR/SciFact
(5,183 docs / 300 test queries) and scores against the published qrels:

| Runtime / dim   | NDCG@10 | Recall@10 | MRR@10 | Notes                                            |
| --------------- | ------- | --------- | ------ | ------------------------------------------------ |
| llama / 768     | 0.682   | 0.870     | 0.626  | Real EmbeddingGemma, Q8_0 GGUF (~0.74 model-card target at FP16) |
| hash / 128      | n/a     | n/a       | n/a    | Deterministic baseline; meaningless for semantic recall |

Measured on `aarch64-apple-darwin` (Apple M5 Pro), llama-cpp + Metal,
chunk size 512, mean pooling. Index time: ~3 min. Query time: 5.6 s
for all 300 queries (~54 qps end-to-end including embedding).

The `hash` baseline is a hash-fingerprint runtime intended only as a
deterministic functional/CI fixture; it has no semantic signal so we
don't compute BEIR scores against it (an always-on synthetic
trigger-token test in `tests/semantic_search.rs` covers the
ranking-pipeline regression for `hash`).

Run yourself: `cargo run --release -p kbcli-tests --features model-llama
--bin run_search_bench`.

## Re-evaluating

When a new runtime or backend is proposed:

1. Add a new L2 impl crate that implements the `EmbeddingRuntime` /
   `VectorStore` trait.
2. Run all four benches with that feature enabled.
3. Apply the decision rules: cosine parity ≥ 0.97 vs FP32 reference,
   speed gap ≥ 1.5× to displace the incumbent, recall ≥ 0.95 vs brute
   force.
4. Update default Cargo features and the CI release matrix.

## Post-mortem on the discarded candidates

Removed during simplification (commit history retains the working code):

* **`candle`** — `candle-transformers` ≤ 0.10 only ships a *causal*
  Gemma3 (`Gemma3Model` emits last-token logits). EmbeddingGemma uses
  bidirectional attention with mean-pooled hidden states, so a parity
  test against any reference fails. Forking `gemma3.rs` to expose hidden
  states with a non-causal mask was ~600 lines mechanical work; not
  worth carrying when `llama` is faster anyway.
* **`ort`** — implemented and worked end-to-end via the
  `onnx-community/embeddinggemma-300m-ONNX` export, but lost throughput
  to `llama` on Apple Silicon (195 vs 250 qps). On x86 with AVX-512 the
  picture typically flips; if a future kbcli release targets a Linux
  server build, `ort` is the natural runtime to bring back.
* **`mistralrs`** — pinned 0.8 has no embedding entrypoint for Gemma.
  Returned `Unimplemented` at runtime; deleted as dead weight.
* **`libsql`** — implemented on top of libSQL's native vector type
  (`F32_BLOB(N)`, `vector_distance_cos`, `libsql_vector_idx`). At
  dim=128 it indexed ~2.3× faster than `sqlite-vec` but lost queries
  at every dim and was 22-37× larger on disk. At dim=768 / N=100k it
  did not converge in a 60-min wall-clock window (graph past 15 GiB
  on disk).

## Operational notes

* `model-llama` requires `cmake` and a C/C++ toolchain because
  `llama-cpp-2` builds llama.cpp from source. On macOS:
  `brew install cmake`.
* On Apple Silicon, llama-cpp picks up Metal automatically.
* The `hash` runtime is for parity / regression testing and offline
  CLI smoke; its output has no semantic meaning and **must not** be
  used to populate a real database.
