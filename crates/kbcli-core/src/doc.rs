use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::Error;

/// Stable, opaque document identifier.
///
/// Wraps a UTF-8 string. Generated as UUIDv7 by default but accepts any
/// caller-supplied ID (e.g., a relative file path).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocId(pub String);

impl DocId {
    pub fn new<S: Into<String>>(s: S) -> Self {
        DocId(s.into())
    }

    /// Generate a fresh UUIDv7-based id.
    pub fn fresh() -> Self {
        DocId(uuid::Uuid::now_v7().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DocId {
    fn from(s: String) -> Self {
        DocId(s)
    }
}

impl From<&str> for DocId {
    fn from(s: &str) -> Self {
        DocId(s.to_string())
    }
}

impl FromStr for DocId {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            Err(Error::invalid("doc id cannot be empty"))
        } else {
            Ok(DocId(s.to_string()))
        }
    }
}

/// Internal chunk identifier (assigned by the store).
pub type ChunkId = i64;

/// A typed metadata value. Kept narrow: strings, numbers, booleans, null,
/// arrays of the same. Heterogeneous nesting is allowed via `Object`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<MetaValue>),
    Object(BTreeMap<String, MetaValue>),
}

impl MetaValue {
    /// Parse a single CLI `k=v` value with light type inference:
    /// "true"/"false" → bool, integer → i64, float → f64, else → string.
    pub fn parse_cli(value: &str) -> Self {
        match value {
            "true" => MetaValue::Bool(true),
            "false" => MetaValue::Bool(false),
            "null" => MetaValue::Null,
            _ => {
                if let Ok(i) = value.parse::<i64>() {
                    return MetaValue::Int(i);
                }
                if let Ok(f) = value.parse::<f64>() {
                    return MetaValue::Float(f);
                }
                MetaValue::Str(value.to_string())
            }
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            MetaValue::Int(i) => Some(*i),
            MetaValue::Float(f) if f.fract() == 0.0 => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::Int(i) => Some(*i as f64),
            MetaValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl From<serde_json::Value> for MetaValue {
    fn from(v: serde_json::Value) -> Self {
        use serde_json::Value as J;
        match v {
            J::Null => MetaValue::Null,
            J::Bool(b) => MetaValue::Bool(b),
            J::Number(n) => {
                if let Some(i) = n.as_i64() {
                    MetaValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    MetaValue::Float(f)
                } else {
                    MetaValue::Str(n.to_string())
                }
            }
            J::String(s) => MetaValue::Str(s),
            J::Array(a) => MetaValue::Array(a.into_iter().map(Into::into).collect()),
            J::Object(o) => MetaValue::Object(
                o.into_iter()
                    .map(|(k, v)| (k, MetaValue::from(v)))
                    .collect(),
            ),
        }
    }
}

impl From<MetaValue> for serde_json::Value {
    fn from(v: MetaValue) -> Self {
        use serde_json::Value as J;
        match v {
            MetaValue::Null => J::Null,
            MetaValue::Bool(b) => J::Bool(b),
            MetaValue::Int(i) => J::Number(i.into()),
            MetaValue::Float(f) => serde_json::Number::from_f64(f).map_or(J::Null, J::Number),
            MetaValue::Str(s) => J::String(s),
            MetaValue::Array(a) => J::Array(a.into_iter().map(Into::into).collect()),
            MetaValue::Object(o) => J::Object(
                o.into_iter()
                    .map(|(k, v)| (k, serde_json::Value::from(v)))
                    .collect(),
            ),
        }
    }
}

/// Key/value metadata associated with a document.
pub type Metadata = BTreeMap<String, MetaValue>;

/// A document persisted in a kbcli database.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: DocId,
    pub text: String,
    #[serde(default)]
    pub metadata: Metadata,
    /// Unix epoch milliseconds; 0 if unknown.
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub updated_at: i64,
}

impl Document {
    pub fn new<I: Into<DocId>, T: Into<String>>(id: I, text: T) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: BTreeMap::new(),
            created_at: 0,
            updated_at: 0,
        }
    }
}

/// A short, listing-friendly view of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSummary {
    pub id: DocId,
    pub text_preview: String,
    pub metadata: Metadata,
    pub created_at: i64,
    pub updated_at: i64,
    pub chunk_count: u32,
}

/// A chunk produced by the chunker, stored alongside its embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Option<ChunkId>,
    pub doc_id: DocId,
    /// Zero-based ordinal of this chunk inside its parent document.
    pub ord: u32,
    pub text: String,
    pub token_count: u32,
    /// Embedding vector. `None` until the embedding pipeline fills it in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl Chunk {
    pub fn new<I: Into<DocId>, T: Into<String>>(doc_id: I, ord: u32, text: T) -> Self {
        let text = text.into();
        let token_count = text.split_whitespace().count() as u32;
        Self {
            id: None,
            doc_id: doc_id.into(),
            ord,
            text,
            token_count,
            embedding: None,
        }
    }
}
