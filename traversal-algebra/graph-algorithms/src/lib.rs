//! Graph algorithms composed with Massively's traversal algebra.
//!
//! `massively::graph` provides traversal and aggregation primitives. This crate
//! composes those primitives into complete graph algorithms.

mod common;

pub use common::{CsrGraph, DeviceCsr, DeviceWeightedCsr, WeightedCsr};

pub mod bc;
pub mod bfs;
pub mod cc;
pub mod color;
pub mod forman_ricci;
pub mod geo;
pub mod hits;
pub mod kcore;
pub mod louvain;
pub mod mst;
pub mod ppr;
pub mod pr;
pub mod pr_nibble;
pub mod rw;
pub mod sm;
pub mod spgemm;
pub mod spmv;
pub mod sssp;
pub mod tc;
