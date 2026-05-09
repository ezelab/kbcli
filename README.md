# kbcli

Fully-local, cross-platform CLI for creating and querying **semantic databases**
of documents. Documents have free-form text plus arbitrary key/value metadata.
Queries combine **lexical** and **semantic** matching with optional metadata
filters — all computed locally with no external services.

## Quickstart

```sh
# create a database
kbcli db create notes

# add documents (multiple input modes)
kbcli doc add notes --text "Rust is a systems programming language" \
    --meta lang=rust --meta tag=lang
echo "Hello world" | kbcli doc add notes --stdin --meta source=manual
kbcli doc add notes --from-dir ./docs --meta project=kbcli
cat seed.jsonl | kbcli doc add notes --jsonl

# search (hybrid lexical + semantic by default)
kbcli query notes "memory safety" --top-k 10
kbcli query notes "memory safety" --filter lang=rust --mode hybrid
```

## Architecture

Strict 4-layer Rust workspace:

```
L3   kbcli-cli, kbcli-tests          ─ consumers
L2   kbcli-embed-llama, kbcli-store-sqlite
L1   kbcli-embed (EmbeddingRuntime trait), kbcli-store (VectorStore trait)
L0   kbcli-core (pure domain types)
```

Two pluggable axes — embedding runtime and storage backend — sit behind
trait objects so additional impls can be dropped in without touching the
CLI. The shipped binary uses the components that won the bake-off
documented in [`docs/perf-report.md`](docs/perf-report.md):

| Axis    | Shipped impl   | Why                                                |
| ------- | -------------- | -------------------------------------------------- |
| Runtime | **`llama`** (llama-cpp + Metal, EmbeddingGemma Q8_0 GGUF) | Real EmbeddingGemma, fastest measured runtime      |
| Storage | **`sqlite-vec`** (vanilla SQLite + sqlite-vec extension)  | Fastest query at every dim, ~22-37× smaller on disk |

The default-feature build is **offline and fast**: it ships the
deterministic `hash` runtime so `cargo install kbcli-cli` works without
downloading model weights or requiring `cmake`. To enable real semantic
embeddings, build with `--features model-llama`:

```sh
# install build deps for llama-cpp-2 (one-time)
brew install cmake          # or apt-get install cmake build-essential

cargo build --release -p kbcli-cli --features model-llama
```

EmbeddingGemma weights (≈300 MB GGUF) download from Hugging Face on first
use into `~/.kbcli/models/embeddinggemma-300m/llama/`.

## Features (Cargo)

| Feature        | Effect                                                                     |
| -------------- | -------------------------------------------------------------------------- |
| (default)      | `hash` runtime + `sqlite-vec` backend. Offline, no native build deps.      |
| `model-llama`  | Compile in `llama-cpp-2` + `hf-hub` and enable real EmbeddingGemma loading. |

## Command reference

### `kbcli db`

```
db create <name> [--path PATH] [--backend NAME] [--runtime NAME]
                 [--dim N] [--chunk-size N] [--chunk-overlap N] [--force]
db list   [--path DIR]
db info   <name>  [--path PATH]
db delete <name>  [--path PATH] [-y]
```

`--runtime` accepts `hash` (default) or `llama` (requires `model-llama`).
`--backend` accepts `sqlite-vec` (default).

### `kbcli doc`

```
doc add  <db> [--text "..." | --file PATH | --stdin | --from-dir DIR | --jsonl]
              [--id ID] [--upsert] [--meta K=V]... [--runtime NAME]
              [--chunk-size N] [--chunk-overlap N] [--path PATH]
doc get  <db> <id>  [--path PATH]
doc list <db> [--filter EXPR]... [--limit N] [--offset N] [--path PATH]
doc update <db> <id> [--text ... | --file PATH] [--meta K=V]... [--unset K]...
                     [--runtime NAME] [--path PATH]
doc delete <db> <id> [--path PATH]
```

### `kbcli query`

```
query <db> <text> [--mode lexical|semantic|hybrid]
                  [--top-k N] [--filter EXPR]...
                  [--rrf-k 60] [--weight-lex 1.0] [--weight-sem 1.0]
                  [--runtime NAME] [--path PATH]
```

### Filter expression syntax

```
key                       # exists
!key                      # missing
key=value                 # eq
key!=value                # ne
key>10  key>=10  key<10 key<=10
key in [a,b,c]
key contains "substr"
```

Repeat `--filter` to AND together multiple expressions. Values are auto-typed
(`true`, `false`, `null`, integers, floats, strings).

### Common flags

* `--json` – emit machine-readable JSON.
* `-v / -vv / -vvv` – tracing verbosity.
* `--path PATH` – override the default `~/.kbcli/<name>.db` location.

## Performance

Headless benchmarks live in `kbcli-tests`:

```sh
# embedding throughput (per runtime)
cargo run --release -p kbcli-tests --features model-llama \
    --bin run_runtime_bench

# index + query latency on the storage backend
cargo run --release -p kbcli-tests --bin run_storage_bench

# composed end-to-end timings
cargo run --release -p kbcli-tests --bin run_e2e_bench

# semantic-search retrieval quality on BEIR/SciFact
cargo run --release -p kbcli-tests --features model-llama \
    --bin run_search_bench
```

Reference numbers and decision rationale are in
[`docs/perf-report.md`](docs/perf-report.md).

## License

MIT OR Apache-2.0
