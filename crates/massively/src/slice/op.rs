//! Compile-time operations used by logical slice generation.

use cubecl::prelude::*;

/// Compile-time operation used by generated lazy slices.
///
/// Implement this trait on a unit-like marker type. The operation receives
/// the logical `u32` index and returns one value for the generated slice.
#[cube]
pub trait TabulateOp<B, T>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    T: crate::Scalar,
{
    /// Generates one output value from its logical index.
    fn apply(index: u32) -> T;
}

/// Compile-time operation used by transform-style lazy slices.
#[cube]
pub trait TransformOp<B, Input, Output>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    Input: crate::MItem<B>,
    Output: crate::Scalar,
{
    /// Maps one logical input element to one generated slice value.
    fn apply(input: Input) -> Output;
}
