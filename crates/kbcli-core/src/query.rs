use serde::{Deserialize, Serialize};

use crate::{DocId, Document, Filter};

/// Search mode for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    Lexical,
    Semantic,
    Hybrid,
}

impl Default for QueryMode {
    fn default() -> Self {
        QueryMode::Hybrid
    }
}

impl std::str::FromStr for QueryMode {
    type Err = crate::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lexical" | "lex" => Ok(QueryMode::Lexical),
            "semantic" | "sem" => Ok(QueryMode::Semantic),
            "hybrid" => Ok(QueryMode::Hybrid),
            other => Err(crate::Error::invalid(format!(
                "unknown query mode: {other}"
            ))),
        }
    }
}

/// A user query against a kbcli database.
///
/// `embedding` is filled in by the consumer (CLI / tests) before handing
/// the request to the storage backend; the backend never invokes the
/// embedding runtime itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub text: String,
    #[serde(default)]
    pub mode: QueryMode,
    pub top_k: u32,
    #[serde(default)]
    pub filter: Filter,
    /// Reciprocal Rank Fusion `k` (typical ~60).
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,
    #[serde(default = "default_weight")]
    pub weight_lex: f32,
    #[serde(default = "default_weight")]
    pub weight_sem: f32,
    /// Pre-computed query embedding (semantic / hybrid modes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

fn default_rrf_k() -> u32 {
    60
}
fn default_weight() -> f32 {
    1.0
}

impl QueryRequest {
    pub fn new<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            mode: QueryMode::Hybrid,
            top_k: 10,
            filter: Filter::All,
            rrf_k: 60,
            weight_lex: 1.0,
            weight_sem: 1.0,
            embedding: None,
        }
    }

    pub fn needs_embedding(&self) -> bool {
        matches!(self.mode, QueryMode::Semantic | QueryMode::Hybrid)
    }
}

/// Per-component scores attached to a query hit (for diagnostics).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HitComponents {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lex_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sem_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lex_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sem_rank: Option<u32>,
}

/// A single result row from a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub doc_id: DocId,
    pub score: f32,
    #[serde(default)]
    pub components: HitComponents,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<Document>,
}
