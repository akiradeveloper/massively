//! Segmented scan over flat-row storage leaves and separate head flags.

#[path = "segmented_fixed.rs"]
mod fixed;

pub(crate) use fixed::{segmented_exclusive, segmented_inclusive, segmented_reduce};
