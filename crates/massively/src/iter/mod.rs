//! Massively iterator traits and Structure-of-Arrays wrappers.

use std::ops::{Bound, Range, RangeBounds};

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch;
use crate::index::MIndex;
use crate::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec};
use crate::value::{MAlloc, MItem};

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

/// Device-backed value that can produce a read-only slice view.
pub trait ToSlice {
    type Slice<'a>
    where
        Self: 'a;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<MIndex>;
}

/// Device-backed value that can produce a mutable slice view.
pub trait ToSliceMut {
    type SliceMut<'a>
    where
        Self: 'a;

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<MIndex>;
}

impl<R, T> ToSlice for DeviceVec<R, T>
where
    R: Runtime,
{
    type Slice<'a>
        = DeviceSlice<'a, R, T>
    where
        Self: 'a;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<MIndex>,
    {
        DeviceVec::slice(self, range)
    }
}

impl<R, T> ToSliceMut for DeviceVec<R, T>
where
    R: Runtime,
{
    type SliceMut<'a>
        = DeviceSliceMut<'a, R, T>
    where
        Self: 'a;

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<MIndex>,
    {
        DeviceVec::slice_mut(self, range)
    }
}

impl<'a, R, T> ToSlice for DeviceSlice<'a, R, T>
where
    R: Runtime,
{
    type Slice<'b>
        = DeviceSlice<'b, R, T>
    where
        Self: 'b;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<MIndex>,
    {
        DeviceSlice::slice(self, range)
    }
}

impl<'a, R, T> ToSlice for DeviceSliceMut<'a, R, T>
where
    R: Runtime,
{
    type Slice<'b>
        = DeviceSlice<'b, R, T>
    where
        Self: 'b;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<MIndex>,
    {
        DeviceSliceMut::slice(self, range)
    }
}

impl<'a, R, T> ToSliceMut for DeviceSliceMut<'a, R, T>
where
    R: Runtime,
{
    type SliceMut<'b>
        = DeviceSliceMut<'b, R, T>
    where
        Self: 'b;

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<MIndex>,
    {
        DeviceSliceMut::slice_mut(self, range)
    }
}

pub(crate) fn normalize_soa_range<Bounds>(len: MIndex, range: Bounds) -> Range<MIndex>
where
    Bounds: RangeBounds<MIndex>,
{
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "slice start ({start}) is greater than slice end ({end})"
    );
    assert!(
        end <= len,
        "slice end ({end}) is out of bounds for SoA of length {len}"
    );
    start..end
}

macro_rules! impl_soa_slice_api {
    ($name:ident < $( $ty:ident : $idx:tt ),+ >) => {
        impl<R, $( $ty ),+> $name<$( DeviceVec<R, $ty> ),+>
        where
            R: Runtime,
        {
            /// Returns read-only device slices for the given logical row range.
            pub fn slice<Bounds>(&self, range: Bounds) -> $name<$( DeviceSlice<'_, R, $ty> ),+>
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_soa_range(self.0.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            /// Returns mutable device slices for the given logical row range.
            pub fn slice_mut<Bounds>(&self, range: Bounds) -> $name<$( DeviceSliceMut<'_, R, $ty> ),+>
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_soa_range(self.0.len(), range);
                $name($( self.$idx.slice_mut(range.clone()) ),+)
            }
        }

        impl<R, $( $ty ),+> ToSlice for $name<$( DeviceVec<R, $ty> ),+>
        where
            R: Runtime,
        {
            type Slice<'a>
                = $name<$( DeviceSlice<'a, R, $ty> ),+>
            where
                Self: 'a;

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: RangeBounds<MIndex>,
            {
                <$name<$( DeviceVec<R, $ty> ),+>>::slice(self, range)
            }
        }

        impl<R, $( $ty ),+> ToSliceMut for $name<$( DeviceVec<R, $ty> ),+>
        where
            R: Runtime,
        {
            type SliceMut<'a>
                = $name<$( DeviceSliceMut<'a, R, $ty> ),+>
            where
                Self: 'a;

            fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
            where
                Bounds: RangeBounds<MIndex>,
            {
                <$name<$( DeviceVec<R, $ty> ),+>>::slice_mut(self, range)
            }
        }

        impl<'a, R, $( $ty ),+> $name<$( DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
        {
            /// Returns read-only device subslices for the given logical row range.
            pub fn slice<Bounds>(&self, range: Bounds) -> $name<$( DeviceSlice<'_, R, $ty> ),+>
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_soa_range(self.0.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }
        }

        impl<'a, R, $( $ty ),+> ToSlice for $name<$( DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
        {
            type Slice<'b>
                = $name<$( DeviceSlice<'b, R, $ty> ),+>
            where
                Self: 'b;

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: RangeBounds<MIndex>,
            {
                <$name<$( DeviceSlice<'a, R, $ty> ),+>>::slice(self, range)
            }
        }

        impl<'a, R, $( $ty ),+> $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
        {
            /// Returns read-only device subslices for the given logical row range.
            pub fn slice<Bounds>(&self, range: Bounds) -> $name<$( DeviceSlice<'_, R, $ty> ),+>
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_soa_range(self.0.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            /// Returns mutable device subslices for the given logical row range.
            pub fn slice_mut<Bounds>(&self, range: Bounds) -> $name<$( DeviceSliceMut<'_, R, $ty> ),+>
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_soa_range(self.0.len(), range);
                $name($( self.$idx.slice_mut(range.clone()) ),+)
            }
        }

        impl<'a, R, $( $ty ),+> ToSlice for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
        {
            type Slice<'b>
                = $name<$( DeviceSlice<'b, R, $ty> ),+>
            where
                Self: 'b;

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: RangeBounds<MIndex>,
            {
                <$name<$( DeviceSliceMut<'a, R, $ty> ),+>>::slice(self, range)
            }
        }

        impl<'a, R, $( $ty ),+> ToSliceMut for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
        {
            type SliceMut<'b>
                = $name<$( DeviceSliceMut<'b, R, $ty> ),+>
            where
                Self: 'b;

            fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
            where
                Bounds: RangeBounds<MIndex>,
            {
                <$name<$( DeviceSliceMut<'a, R, $ty> ),+>>::slice_mut(self, range)
            }
        }

        impl<R, $( $ty ),+> dispatch::ToHostDispatch<R> for $name<$( DeviceVec<R, $ty> ),+>
        where
            R: Runtime,
            $( DeviceVec<R, $ty>: dispatch::ToHostDispatch<R>, )+
        {
            type Output = ($( <DeviceVec<R, $ty> as dispatch::ToHostDispatch<R>>::Output, )+);

            fn to_host_with(&self, exec: &crate::runtime::Executor<R>) -> Result<Self::Output, Error> {
                Ok(($( dispatch::ToHostDispatch::to_host_with(&self.$idx, exec)?, )+))
            }
        }

        impl<'a, R, $( $ty ),+> dispatch::ToHostDispatch<R> for $name<$( DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( DeviceSlice<'a, R, $ty>: dispatch::ToHostDispatch<R>, )+
        {
            type Output = ($( <DeviceSlice<'a, R, $ty> as dispatch::ToHostDispatch<R>>::Output, )+);

            fn to_host_with(&self, exec: &crate::runtime::Executor<R>) -> Result<Self::Output, Error> {
                Ok(($( dispatch::ToHostDispatch::to_host_with(&self.$idx, exec)?, )+))
            }
        }
    };
}

impl_soa_slice_api!(SoA1<A: 0>);
impl_soa_slice_api!(SoA2<A: 0, B: 1>);
impl_soa_slice_api!(SoA3<A: 0, B: 1, C: 2>);
impl_soa_slice_api!(SoA4<A: 0, B: 1, C: 2, D: 3>);
impl_soa_slice_api!(SoA5<A: 0, B: 1, C: 2, D: 3, E: 4>);
impl_soa_slice_api!(SoA6<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5>);
impl_soa_slice_api!(SoA7<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6>);

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

    #[doc(hidden)]
    fn into_view_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error>
    where
        Self::Item: MAlloc<R>;

    /// Returns the logical length.
    fn len(&self) -> MIndex;

    /// Returns whether this slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Mutable massively iterator used as an explicit algorithm output.
pub trait MIterMut<R: Runtime>: dispatch::MIterMutDispatch<R> + Sized {
    type Item: MAlloc<R>;

    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner;

    #[doc(hidden)]
    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn write_prefix_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn write_split_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        selected: <Self::Item as MAlloc<R>>::Inner,
        rejected: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn replace_where_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        replacement: Self::Item,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error>;

    #[doc(hidden)]
    fn fill_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: Self::Item,
    ) -> Result<(), Error>;

    /// Returns the logical length.
    fn len(&self) -> MIndex;

    /// Returns whether this output slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
