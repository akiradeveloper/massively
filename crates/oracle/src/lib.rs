//! Full CPU reference implementations for `massively` property tests.
//!
//! Reference algorithms are classified as [`vector`] and [`seg`]. Operation
//! traits live in [`op`] and intentionally mirror the public GPU operation traits
//! without CubeCL runtime constraints.

pub mod op;
pub mod seg;
pub mod vector;
