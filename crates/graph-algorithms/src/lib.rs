//! Graph algorithms composed with Massively's traversal algebra.
//!
//! `massively::graph` provides traversal and aggregation primitives. This crate
//! composes those primitives into complete graph algorithms.

mod common;

pub use common::{CsrGraph, WeightedCsr};

pub mod bc;
pub mod bfs;
pub mod color;
pub mod forman_ricci;
pub mod geo;
pub mod hits;
pub mod kcore;
pub mod mst;
pub mod ppr;
pub mod pr;
pub mod spgemm;
pub mod spmv;
pub mod sssp;
pub mod tc;
