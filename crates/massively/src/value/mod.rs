//! Massively logical item and owned vector traits.

use cubecl::prelude::{CubeType, Runtime};

use crate::detail::dispatch;

/// Logical item handled by massively algorithms.
///
/// An `MItem` is one element of an [`crate::iter::MIter`] or [`MVec`]. The
/// current public model represents items as tuples such as `(T,)`, `(T, U)`,
/// and `(T, U, V)`; internally those tuples are stored as SoA device columns for
/// backend `R`.
pub trait MItem<R: Runtime>: dispatch::MItemDispatch<R> + CubeType + Sized + 'static {
    #[doc(hidden)]
    type Inner;
}

/// Owned massively vector for a logical item.
pub trait MVec<R: Runtime>: Sized {
    type Item: MItem<R>;

    #[doc(hidden)]
    fn from_inner(inner: <Self::Item as MItem<R>>::Inner) -> Self;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this array has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
