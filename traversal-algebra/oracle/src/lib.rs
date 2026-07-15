//! Independent Rust façade for the Lean Traversal Algebra oracle.
//!
//! The public [`graph`] module intentionally mirrors the semantic shape of
//! `massively::graph` without depending on its implementation.  Queries are
//! evaluated by a persistent Lean process through [`LeanOracle`].

mod client;
mod cost;
pub mod graph;
mod protocol;

use std::{fmt, io};

pub use client::LeanOracle;
pub use cost::{CubeClCertificate, CubeClCost, CubeClMachine, DestinationStrategy};
pub use graph::{EdgeContext, OracleCase};

mod generated;

pub use generated::CASES;

/// Failure returned while constructing or evaluating an oracle query.
#[derive(Debug)]
pub enum Error {
    InvalidInput(String),
    Io(io::Error),
    Protocol(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid oracle input: {message}"),
            Self::Io(error) => write!(formatter, "oracle I/O failed: {error}"),
            Self::Protocol(message) => write!(formatter, "Lean oracle protocol failed: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidInput(_) | Self::Protocol(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
