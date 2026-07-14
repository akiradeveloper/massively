//! Lean-generated regression cases for Traversal Algebra.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeContext {
    pub source: u32,
    pub destination: u32,
    pub edge: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct OracleCase {
    pub name: &'static str,
    pub offsets: &'static [u32],
    pub destinations: &'static [u32],
    pub frontier: &'static [u32],
    pub expected_edges: &'static [EdgeContext],
    pub expected_source_counts: &'static [u32],
    pub expected_destination_counts: &'static [u32],
}

mod generated;

pub use generated::CASES;
