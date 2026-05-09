//! kbcli-core: pure domain types shared across the workspace.
//!
//! This crate defines the data model and error type. It performs no I/O,
//! no SQL, and no ML — it is the L0 of the layered workspace.

mod doc;
mod error;
mod filter;
mod query;

pub use doc::{Chunk, ChunkId, DocId, DocSummary, Document, MetaValue, Metadata};
pub use error::{Error, Result};
pub use filter::{Filter, Predicate};
pub use query::{Hit, HitComponents, QueryMode, QueryRequest};
