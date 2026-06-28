//! Massively logical item traits.

use cubecl::prelude::{CubeType, Runtime};

use crate::detail::dispatch;

/// Logical item handled by massively algorithms.
///
/// An `MItem` is one element of an [`crate::iter::MIter`]. The current public
/// model represents items as tuples such as `(T,)`, `(T, U)`, and `(T, U, V)`;
/// internally those tuples are stored as SoA device columns for backend `R`.
pub trait MItem<R: Runtime>: dispatch::MItemDispatch<R> + CubeType + Sized + 'static {
    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    type Vec: MVec<R, Item = Self>;

    #[doc(hidden)]
    fn vec_from_inner(inner: Self::Inner) -> Self::Vec;
}

#[doc(hidden)]
pub trait MVec<R: Runtime>: Sized {
    type Item: MItem<R>;

    fn from_inner(inner: <Self::Item as MItem<R>>::Inner) -> Self;
}
