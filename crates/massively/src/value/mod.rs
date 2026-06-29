//! Massively logical item traits.

use std::ops::RangeBounds;

use cubecl::prelude::{CubeType, Runtime};

use crate::Error;
use crate::detail::dispatch;
use crate::iter::{MIter, MIterMut};
use crate::runtime::Executor;

/// Logical item handled by massively algorithms.
///
/// An `MItem` is one element of an [`crate::iter::MIter`]. The current public
/// model represents items as tuples such as `(T,)`, `(T, U)`, and `(T, U, V)`;
/// internally those tuples are stored as SoA device columns for backend `R`.
pub trait MItem<R: Runtime>:
    dispatch::MItemDispatch<R> + CubeType + Copy + Sized + 'static
{
    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    type View;

    #[doc(hidden)]
    type Vec: MVec<R, Item = Self>;

    #[doc(hidden)]
    fn vec_from_inner(inner: Self::Inner) -> Self::Vec;

    #[doc(hidden)]
    fn alloc_vec(exec: &Executor<R>, len: usize) -> Result<Self::Vec, Error>;
}

/// Owned device storage for an [`MItem`].
///
/// Algorithms that return owned output use this trait through `MItem::Vec`.
/// `Executor::alloc::<Item>(len)` also returns this storage shape, and
/// `slice` / `slice_mut` turn it into device-backed SoA views.
pub trait MVec<R: Runtime>: Sized {
    type Item: MItem<R>;
    type Slice<'a>: MIter<R, Item = Self::Item>
    where
        Self: 'a;
    type SliceMut<'a>: MIterMut<R, Item = Self::Item>
    where
        Self: 'a;

    fn from_inner(inner: <Self::Item as MItem<R>>::Inner) -> Self;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<usize>;

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<usize>;
}
