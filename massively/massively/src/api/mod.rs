pub mod algorithm;
pub mod iter;
pub mod lazy;
pub mod op;
pub mod runtime;
pub mod tuple;
pub mod util;

pub use crate::core::error::Error;

/// Index and count type returned by public algorithms.
pub type MIndex = u32;
