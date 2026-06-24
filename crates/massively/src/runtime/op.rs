//! Compile-time operations used by runtime memory APIs.

use std::marker::PhantomData;

use cubecl::prelude::*;

/// Compile-time operation used by [`Executor::tabulate`](crate::Executor::tabulate).
///
/// Implement this trait on a unit-like marker type. The operation receives
/// the logical `u32` index and returns one value for the generated device
/// column.
#[cube]
pub trait TabulateOp<B, T>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    T: crate::Scalar,
{
    /// Generates one output value from its logical index.
    fn apply(index: u32) -> T;
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KernelTabulateOp<B, Op, Output>(PhantomData<fn() -> (B, Op, Output)>);

impl<B, Op, Output> KernelTabulateOp<B, Op, Output> {
    pub(crate) fn new() -> Self {
        Self(PhantomData)
    }
}

#[cubecl::cube]
impl<B, Op, Output> crate::algorithm::op::kernel::UnaryOp<u32> for KernelTabulateOp<B, Op, Output>
where
    B: cubecl::prelude::Runtime,
    Output: crate::Scalar,
    Op: TabulateOp<B, Output>,
{
    type Output = Output;

    fn apply(input: u32) -> Output {
        Op::apply(input)
    }
}
