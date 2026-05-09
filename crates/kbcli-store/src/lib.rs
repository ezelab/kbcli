//! kbcli-store: `VectorStore` trait + shared SQL/migration helpers.

mod sql_filter;
mod store;

pub use sql_filter::{filter_to_sql, FilterSql};
pub use store::{StoreConfig, StoreInfo, UpsertResult, VectorStore};
