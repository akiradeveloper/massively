//! Massively logical item traits.

use cubecl::prelude::{CubeElement, CubePrimitive, CubeType, Runtime};

use crate::Error;
use crate::detail::dispatch;
use crate::index::MIndex;
use crate::runtime::Executor;

/// Logical item handled by massively algorithms.
///
/// An `MItem` is one element of an [`crate::iter::MIter`]. The current public
/// model represents items as tuples such as `(T,)`, `(T, U)`, and `(T, U, V)`;
/// internally those tuples are stored as SoA device columns for backend `R`.
pub trait MItem<R: Runtime>:
    dispatch::MItemDispatch<R> + CubeType + Copy + Sized + 'static
{
}

/// Physical element that can be stored in a device column.
pub trait MStorageElement: CubePrimitive + CubeElement {}
impl<T> MStorageElement for T where T: CubePrimitive + CubeElement {}

/// Logical item that has an owned/writable device storage shape.
pub trait MAlloc<R: Runtime>: MItem<R> {
    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    type View;

    #[doc(hidden)]
    type Storage: StorageFromInner<R, Item = Self>;

    #[doc(hidden)]
    fn storage_from_inner(inner: Self::Inner) -> Self::Storage;

    #[doc(hidden)]
    fn alloc_storage(exec: &Executor<R>, len: MIndex) -> Result<Self::Storage, Error>;
}

#[doc(hidden)]
pub trait StorageFromInner<R: Runtime>: Sized {
    type Item: MAlloc<R>;

    fn from_inner(inner: <Self::Item as MAlloc<R>>::Inner) -> Self;

    #[doc(hidden)]
    fn into_inner(self) -> <Self::Item as MAlloc<R>>::Inner;

    fn len(&self) -> MIndex;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
