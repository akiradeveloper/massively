//! Public massively facade traits.

use cubecl::prelude::CubeType;

use crate::algorithm::api::sealed;
use cubecl::prelude::Runtime;

/// Logical item handled by massively algorithms.
///
/// An `MItem` is one element of an [`MIter`] or [`MVec`]. The current public
/// model represents items as tuples such as `(T,)`, `(T, U)`, and `(T, U, V)`;
/// internally those tuples are stored as SoA device columns for backend `B`.
pub trait MItem<B: Runtime>: sealed::MItemDispatch<B> + CubeType + Sized + 'static {
    #[doc(hidden)]
    type Inner;
}

/// Owned massively vector for a logical item.
pub trait MVec<B: Runtime>: Sized {
    type Item: MItem<B>;

    #[doc(hidden)]
    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this array has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Massively iterator.
pub trait MIter<B: Runtime>: sealed::MIterDispatch<B> + Sized {
    type Item: MItem<B>;

    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
