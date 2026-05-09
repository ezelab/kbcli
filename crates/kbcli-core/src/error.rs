use std::fmt;

use serde::{Deserialize, Serialize};

/// Unified error type for the kbcli workspace.
///
/// Lower layers (`kbcli-embed`, `kbcli-store`, …) convert their internal
/// errors into this enum so that callers (CLI, tests) deal with a single
/// concrete type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("schema/migration error: {0}")]
    Schema(String),

    #[error("storage error: {0}")]
    Store(String),

    #[error("embedding error: {0}")]
    Embed(String),

    #[error("feature `{0}` is not enabled in this build")]
    FeatureDisabled(&'static str),

    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn invalid<S: Into<String>>(msg: S) -> Self {
        Error::InvalidInput(msg.into())
    }
    pub fn not_found<S: fmt::Display>(what: S) -> Self {
        Error::NotFound(what.to_string())
    }
    pub fn conflict<S: Into<String>>(msg: S) -> Self {
        Error::Conflict(msg.into())
    }
    pub fn store<S: Into<String>>(msg: S) -> Self {
        Error::Store(msg.into())
    }
    pub fn embed<S: Into<String>>(msg: S) -> Self {
        Error::Embed(msg.into())
    }
    pub fn other<S: Into<String>>(msg: S) -> Self {
        Error::Other(msg.into())
    }
}

/// Convenience alias for results returned by kbcli APIs.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Stable, machine-readable error code (for `--json` output).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Io,
    Serde,
    InvalidInput,
    NotFound,
    Conflict,
    Schema,
    Store,
    Embed,
    FeatureDisabled,
    Unimplemented,
    Other,
}

impl Error {
    pub fn code(&self) -> ErrorCode {
        match self {
            Error::Io(_) => ErrorCode::Io,
            Error::Serde(_) => ErrorCode::Serde,
            Error::InvalidInput(_) => ErrorCode::InvalidInput,
            Error::NotFound(_) => ErrorCode::NotFound,
            Error::Conflict(_) => ErrorCode::Conflict,
            Error::Schema(_) => ErrorCode::Schema,
            Error::Store(_) => ErrorCode::Store,
            Error::Embed(_) => ErrorCode::Embed,
            Error::FeatureDisabled(_) => ErrorCode::FeatureDisabled,
            Error::Unimplemented(_) => ErrorCode::Unimplemented,
            Error::Other(_) => ErrorCode::Other,
        }
    }
}
