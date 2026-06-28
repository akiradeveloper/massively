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

/// Four-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA4<A, B, C, D>(pub A, pub B, pub C, pub D);

/// Five-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA5<A, B, C, D, E>(pub A, pub B, pub C, pub D, pub E);

/// Six-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA6<A, B, C, D, E, F>(pub A, pub B, pub C, pub D, pub E, pub F);

/// Seven-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA7<A, B, C, D, E, F, G>(pub A, pub B, pub C, pub D, pub E, pub F, pub G);

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

impl<A, B, C, D> From<(A, B, C, D)> for SoA4<A, B, C, D> {
    fn from(value: (A, B, C, D)) -> Self {
        Self(value.0, value.1, value.2, value.3)
    }
}

impl<A, B, C, D, E> From<(A, B, C, D, E)> for SoA5<A, B, C, D, E> {
    fn from(value: (A, B, C, D, E)) -> Self {
        Self(value.0, value.1, value.2, value.3, value.4)
    }
}

impl<A, B, C, D, E, F> From<(A, B, C, D, E, F)> for SoA6<A, B, C, D, E, F> {
    fn from(value: (A, B, C, D, E, F)) -> Self {
        Self(value.0, value.1, value.2, value.3, value.4, value.5)
    }
}

impl<A, B, C, D, E, F, G> From<(A, B, C, D, E, F, G)> for SoA7<A, B, C, D, E, F, G> {
    fn from(value: (A, B, C, D, E, F, G)) -> Self {
        Self(
            value.0, value.1, value.2, value.3, value.4, value.5, value.6,
        )
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
