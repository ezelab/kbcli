//! SQLite + sqlite-vec storage backend.

mod schema;
mod store_impl;

pub use store_impl::SqliteStore;
