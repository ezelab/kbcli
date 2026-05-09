//! llama.cpp (GGUF) EmbeddingRuntime impl.
//!
//! When the `model` feature is on, this crate downloads a GGUF build of
//! EmbeddingGemma from HF, opens it in embedding mode with `llama-cpp-2`,
//! tokenizes inputs, and reads sequence-level embeddings produced by the
//! configured pooling strategy.

use async_trait::async_trait;
use kbcli_core::{Error, Result};
use kbcli_embed::{EmbeddingRuntime, RuntimeConfig};

pub struct LlamaRuntime {
    cfg: RuntimeConfig,
    #[cfg(feature = "model")]
    inner: model::Model,
}

impl LlamaRuntime {
    pub async fn new(cfg: RuntimeConfig) -> Result<Self> {
        #[cfg(feature = "model")]
        {
            let inner = model::Model::load(&cfg).await?;
            Ok(Self { cfg, inner })
        }
        #[cfg(not(feature = "model"))]
        {
            let _ = cfg;
            Err(Error::FeatureDisabled("kbcli-embed-llama/model"))
        }
    }
}

#[async_trait]
impl EmbeddingRuntime for LlamaRuntime {
    fn name(&self) -> &'static str {
        "llama"
    }
    fn dim(&self) -> usize {
        #[cfg(feature = "model")]
        {
            self.cfg
                .matryoshka_dim
                .unwrap_or_else(|| self.inner.native_dim())
        }
        #[cfg(not(feature = "model"))]
        {
            self.cfg.matryoshka_dim.unwrap_or(768)
        }
    }
    fn max_input_tokens(&self) -> usize {
        2048
    }
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "model")]
        {
            self.inner.embed_batch(texts, self.cfg.matryoshka_dim).await
        }
        #[cfg(not(feature = "model"))]
        {
            let _ = texts;
            Err(Error::FeatureDisabled("kbcli-embed-llama/model"))
        }
    }
}

#[cfg(feature = "model")]
mod model {
    use super::*;
    use hf_hub::api::tokio::Api;
    use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel, Special};
    use std::num::NonZeroU32;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    pub struct Model {
        backend: Arc<LlamaBackend>,
        model: Arc<LlamaModel>,
        native_dim: usize,
        max_ctx: u32,
        // Serialize llama context creation; llama.cpp is single-threaded
        // per-context.
        guard: Arc<Mutex<()>>,
    }

    impl Model {
        pub fn native_dim(&self) -> usize {
            self.native_dim
        }

        pub async fn load(cfg: &RuntimeConfig) -> Result<Self> {
            let path: PathBuf = if let Some(p) = &cfg.model_path {
                if p.is_file() {
                    p.clone()
                } else {
                    download_gguf().await?
                }
            } else {
                download_gguf().await?
            };

            let backend =
                LlamaBackend::init().map_err(|e| Error::embed(format!("llama backend: {e}")))?;
            let backend = Arc::new(backend);

            let mparams = LlamaModelParams::default();
            let model = LlamaModel::load_from_file(&backend, &path, &mparams)
                .map_err(|e| Error::embed(format!("llama load: {e}")))?;
            let native_dim = model.n_embd() as usize;

            Ok(Self {
                backend,
                model: Arc::new(model),
                native_dim,
                max_ctx: 2048,
                guard: Arc::new(Mutex::new(())),
            })
        }

        pub async fn embed_batch(
            &self,
            texts: &[&str],
            matryoshka_dim: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let model = self.model.clone();
            let backend = self.backend.clone();
            let max_ctx = self.max_ctx;
            let native_dim = self.native_dim;
            let texts: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
            let _g = self.guard.lock().await;

            tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
                let cparams = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(max_ctx))
                    .with_n_batch(max_ctx)
                    .with_n_ubatch(max_ctx)
                    .with_embeddings(true)
                    .with_pooling_type(LlamaPoolingType::Mean);
                let mut ctx = model
                    .new_context(&backend, cparams)
                    .map_err(|e| Error::embed(format!("llama ctx: {e}")))?;

                let max_toks = max_ctx as usize;
                let mut out = Vec::with_capacity(texts.len());
                for text in &texts {
                    let mut toks = model
                        .str_to_token(text, AddBos::Always)
                        .map_err(|e| Error::embed(format!("tokenize: {e}")))?;
                    if toks.len() > max_toks {
                        toks.truncate(max_toks);
                    }
                    let n = toks.len() as i32;
                    if n == 0 {
                        out.push(vec![0f32; native_dim]);
                        continue;
                    }
                    let mut batch = LlamaBatch::new(n.max(1) as usize, 1);
                    for (i, t) in toks.iter().enumerate() {
                        let last = i as i32 == n - 1;
                        batch
                            .add(*t, i as i32, &[0], last)
                            .map_err(|e| Error::embed(format!("batch add: {e}")))?;
                    }
                    ctx.clear_kv_cache();
                    ctx.decode(&mut batch)
                        .map_err(|e| Error::embed(format!("decode: {e}")))?;

                    let emb = ctx
                        .embeddings_seq_ith(0)
                        .map_err(|e| Error::embed(format!("embeddings: {e}")))?
                        .to_vec();
                    out.push(emb);
                    let _ = Special::Tokenize; // touch to silence unused-imports under some llama-cpp-2 versions
                }

                let mut result = Vec::with_capacity(out.len());
                for mut v in out {
                    kbcli_embed::l2_normalize(&mut v);
                    if let Some(d) = matryoshka_dim {
                        v = kbcli_embed::matryoshka_truncate(v, d);
                    }
                    result.push(v);
                }
                Ok(result)
            })
            .await
            .map_err(|e| Error::embed(format!("llama blocking: {e}")))?
        }
    }

    async fn download_gguf() -> Result<PathBuf> {
        let api = Api::new().map_err(|e| Error::embed(format!("hf-hub: {e}")))?;
        // Try a known-good GGUF repo + filename. Users can override via
        // `RuntimeConfig::model_path` to point at a local file.
        let candidates = [
            (
                "ggml-org/embeddinggemma-300M-GGUF",
                "embeddinggemma-300M-Q8_0.gguf",
            ),
            (
                "lmstudio-community/embeddinggemma-300M-GGUF",
                "embeddinggemma-300M-Q8_0.gguf",
            ),
            (
                "ggml-org/embeddinggemma-300m-GGUF",
                "embeddinggemma-300m-Q8_0.gguf",
            ),
        ];
        let mut last = None;
        for (repo, file) in candidates {
            let r = api.model(repo.to_string());
            match r.get(file).await {
                Ok(p) => return Ok(p),
                Err(e) => last = Some(format!("{repo}/{file}: {e}")),
            }
        }
        Err(Error::embed(format!(
            "could not download EmbeddingGemma GGUF (last error: {})",
            last.unwrap_or_default()
        )))
    }
}
