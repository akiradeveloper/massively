//! Independent sequential CPU oracle for Massively Traversal Algebra.
//!
//! The public [`graph`] module mirrors the semantic shape of
//! `massively::graph` without depending on its implementation. [`CpuOracle`]
//! evaluates its queries directly from host-owned CSR arrays.

mod cpu;
pub mod graph;

use std::fmt;

pub use cpu::CpuOracle;
pub use graph::EdgeContext;

/// Failure returned while constructing or evaluating an oracle query.
#[derive(Debug)]
pub enum Error {
    InvalidInput(String),
    ArithmeticOverflow,
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid oracle input: {message}"),
            Self::ArithmeticOverflow => {
                write!(formatter, "oracle natural-number result does not fit u32")
            }
            Self::Internal(message) => write!(formatter, "CPU oracle invariant failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
