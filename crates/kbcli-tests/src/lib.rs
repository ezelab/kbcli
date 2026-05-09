//! Shared helpers for kbcli-tests.
//!
//! This crate hosts every kind of kbcli test (functional, parity, recall,
//! benchmarks). The library module exposes shared building blocks; the
//! `tests/` directory is where the actual `cargo test` entry-points live.

pub mod assertions;
#[cfg(feature = "model-llama")]
pub mod beir;
pub mod corpus;
pub mod eval;
pub mod fixtures;
pub mod runners;
