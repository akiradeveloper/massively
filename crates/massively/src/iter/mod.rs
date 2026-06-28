//! Massively iterator traits and Structure-of-Arrays wrappers.

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch;
use crate::value::MItem;

/// Single-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA1<A>(pub A);

/// Two-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA2<A, B>(pub A, pub B);

/// Three-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA3<A, B, C>(pub A, pub B, pub C);

impl<A> From<(A,)> for SoA1<A> {
    fn from(value: (A,)) -> Self {
        Self(value.0)
    }
}

impl<A, B> From<(A, B)> for SoA2<A, B> {
    fn from(value: (A, B)) -> Self {
        Self(value.0, value.1)
    }
}

impl<A, B, C> From<(A, B, C)> for SoA3<A, B, C> {
    fn from(value: (A, B, C)) -> Self {
        Self(value.0, value.1, value.2)
    }
}

/// Massively iterator.
pub trait MIter<R: Runtime>: dispatch::MIterDispatch<R> + Sized {
    type Item: MItem<R>;

    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner;

    #[doc(hidden)]
    fn into_inner_with_policy(
        self,
        _policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Self::Inner, Error> {
        Ok(self.into_inner())
    }

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Mutable massively iterator used as an explicit algorithm output.
pub trait MIterMut<R: Runtime>: dispatch::MIterMutDispatch<R> + Sized {
    type Item: MItem<R>;

    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner;

    #[doc(hidden)]
    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MItem<R>>::Inner,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MItem<R>>::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn replace_where_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        replacement: Self::Item,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error>;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this output slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
