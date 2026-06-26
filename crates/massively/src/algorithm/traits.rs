//! Public massively facade traits.

use cubecl::prelude::CubeType;

use crate::Error;
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

/// Mutable massively iterator used as an explicit algorithm output.
pub trait MIterMut<B: Runtime>: sealed::MIterMutDispatch<B> + Sized {
    type Item: MItem<B>;

    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner;

    #[doc(hidden)]
    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<B>,
        inner: <Self::Item as MItem<B>>::Inner,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn write_where_from_inner<Stencil>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        inner: <Self::Item as MItem<B>>::Inner,
        stencil: Stencil,
    ) -> Result<(), Error>
    where
        Stencil: MIter<B, Item = (u32,)>;

    #[doc(hidden)]
    fn replace_where_inner<Stencil>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        replacement: Self::Item,
        stencil: Stencil,
    ) -> Result<(), Error>
    where
        Stencil: MIter<B, Item = (u32,)>;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this output slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
