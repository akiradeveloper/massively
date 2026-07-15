//! Segmented scan over canonical storage leaves and separate head flags.

#[path = "segmented_fixed.rs"]
mod fixed;

pub(crate) use fixed::segmented_fixed;
