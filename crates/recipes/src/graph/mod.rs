//! Graph algorithms composed with Massively's traversal algebra.

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
