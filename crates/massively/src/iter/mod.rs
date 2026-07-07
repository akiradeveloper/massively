//! Massively iterator traits and Zip wrappers.

use std::ops::{Bound, Range, RangeBounds};

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch;
use crate::index::MIndex;
use crate::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec};
use crate::value::{MAlloc, MItem, MStorageElement, StorageFromInner};

/// Single-column Zip container.
#[derive(Clone, Copy, Debug)]
pub struct Zip1<A>(pub A);

/// Two-column Zip container.
#[derive(Clone, Copy, Debug)]
pub struct Zip2<A, B>(pub A, pub B);

/// Three-column Zip container.
#[derive(Clone, Copy, Debug)]
pub struct Zip3<A, B, C>(pub A, pub B, pub C);

/// Four-column Zip container.
#[derive(Clone, Copy, Debug)]
pub struct Zip4<A, B, C, D>(pub A, pub B, pub C, pub D);

/// Five-column Zip container.
#[derive(Clone, Copy, Debug)]
pub struct Zip5<A, B, C, D, E>(pub A, pub B, pub C, pub D, pub E);

/// Six-column Zip container.
#[derive(Clone, Copy, Debug)]
pub struct Zip6<A, B, C, D, E, F>(pub A, pub B, pub C, pub D, pub E, pub F);

/// Seven-column Zip container.
#[derive(Clone, Copy, Debug)]
pub struct Zip7<A, B, C, D, E, F, G>(pub A, pub B, pub C, pub D, pub E, pub F, pub G);

impl<A> From<(A,)> for Zip1<A> {
    fn from(value: (A,)) -> Self {
        Self(value.0)
    }
}

impl<A, B> From<(A, B)> for Zip2<A, B> {
    fn from(value: (A, B)) -> Self {
        Self(value.0, value.1)
    }
}

impl<A, B, C> From<(A, B, C)> for Zip3<A, B, C> {
    fn from(value: (A, B, C)) -> Self {
        Self(value.0, value.1, value.2)
    }
}

impl<A, B, C, D> From<(A, B, C, D)> for Zip4<A, B, C, D> {
    fn from(value: (A, B, C, D)) -> Self {
        Self(value.0, value.1, value.2, value.3)
    }
}

impl<A, B, C, D, E> From<(A, B, C, D, E)> for Zip5<A, B, C, D, E> {
    fn from(value: (A, B, C, D, E)) -> Self {
        Self(value.0, value.1, value.2, value.3, value.4)
    }
}

impl<A, B, C, D, E, F> From<(A, B, C, D, E, F)> for Zip6<A, B, C, D, E, F> {
    fn from(value: (A, B, C, D, E, F)) -> Self {
        Self(value.0, value.1, value.2, value.3, value.4, value.5)
    }
}

impl<A, B, C, D, E, F, G> From<(A, B, C, D, E, F, G)> for Zip7<A, B, C, D, E, F, G> {
    fn from(value: (A, B, C, D, E, F, G)) -> Self {
        Self(
            value.0, value.1, value.2, value.3, value.4, value.5, value.6,
        )
    }
}

pub(crate) fn normalize_zip_range<Bounds>(len: MIndex, range: Bounds) -> Range<MIndex>
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
        "slice end ({end}) is out of bounds for Zip of length {len}"
    );
    start..end
}

#[derive(Clone, Copy, Debug)]
struct MIterSliceRange {
    start: MIndex,
    end: MIndex,
}

impl MIterSliceRange {
    fn new(range: Range<MIndex>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }

    fn len(self) -> MIndex {
        self.end - self.start
    }

    fn to_range(self) -> Range<MIndex> {
        self.start..self.end
    }
}

/// Read-only slice adapter over an `MIter`.
#[derive(Debug)]
pub struct MIterSlice<'a, Iter> {
    inner: &'a Iter,
    range: MIterSliceRange,
}

impl<'a, Iter> MIterSlice<'a, Iter> {
    pub(crate) fn new(inner: &'a Iter, range: Range<MIndex>) -> Self {
        Self {
            inner,
            range: MIterSliceRange::new(range),
        }
    }
}

impl<'a, 'b, R, T> dispatch::ToHostDispatch<R> for MIterSlice<'a, DeviceSlice<'b, R, T>>
where
    R: Runtime,
    T: MStorageElement,
{
    type Output = Vec<T>;

    fn to_host_with(&self, exec: &crate::runtime::Executor<R>) -> Result<Self::Output, Error> {
        dispatch::ToHostDispatch::to_host_with(
            &DeviceSlice::slice(&self.inner, self.range.to_range()),
            exec,
        )
    }
}

/// Mutable slice adapter over an `MIterMut`.
#[derive(Debug)]
pub struct MIterMutSliceMut<'a, Iter> {
    inner: &'a Iter,
    range: MIterSliceRange,
}

impl<'a, Iter> MIterMutSliceMut<'a, Iter> {
    pub(crate) fn new(inner: &'a Iter, range: Range<MIndex>) -> Self {
        Self {
            inner,
            range: MIterSliceRange::new(range),
        }
    }
}

/// Read-only slice adapter over an `MIterMut`.
#[derive(Debug)]
pub struct MIterMutSlice<'a, Iter> {
    inner: &'a Iter,
    range: MIterSliceRange,
}

impl<'a, Iter> MIterMutSlice<'a, Iter> {
    pub(crate) fn new(inner: &'a Iter, range: Range<MIndex>) -> Self {
        Self {
            inner,
            range: MIterSliceRange::new(range),
        }
    }
}

macro_rules! impl_zip_slice_api {
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
                let range = normalize_zip_range(self.0.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            /// Returns mutable device slices for the given logical row range.
            pub fn slice_mut<Bounds>(&self, range: Bounds) -> $name<$( DeviceSliceMut<'_, R, $ty> ),+>
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_zip_range(self.0.len(), range);
                $name($( self.$idx.slice_mut(range.clone()) ),+)
            }
        }

        impl<R, $( $ty ),+> MStorage<R> for $name<$( DeviceVec<R, $ty> ),+>
        where
            R: Runtime,
            Self: StorageFromInner<R, Item = ($( $ty, )+)>,
            ($( $ty, )+): MAlloc<R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
            >,
            $( $ty: MStorageElement + 'static, )+
        {
            fn len(&self) -> MIndex {
                self.0.len()
            }

            fn slice<Bounds>(&self, range: Bounds) -> impl MIter<R, Item = Self::Item> + '_
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_zip_range(self.0.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            fn slice_mut<Bounds>(&self, range: Bounds) -> impl MIterMut<R, Item = Self::Item> + '_
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_zip_range(self.0.len(), range);
                $name($( self.$idx.slice_mut(range.clone()) ),+)
            }

            fn into_alloc_view_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<<Self::Item as MAlloc<R>>::View, Error> {
                let _ = policy;
                Ok(($( self.$idx.slice(..).column_view(), )+))
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
                let range = normalize_zip_range(self.0.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
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
                let range = normalize_zip_range(self.0.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            /// Returns mutable device subslices for the given logical row range.
            pub fn slice_mut<Bounds>(&self, range: Bounds) -> $name<$( DeviceSliceMut<'_, R, $ty> ),+>
            where
                Bounds: RangeBounds<MIndex>,
            {
                let range = normalize_zip_range(self.0.len(), range);
                $name($( self.$idx.slice_mut(range.clone()) ),+)
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

impl_zip_slice_api!(Zip1<A: 0>);
impl_zip_slice_api!(Zip2<A: 0, B: 1>);
impl_zip_slice_api!(Zip3<A: 0, B: 1, C: 2>);
impl_zip_slice_api!(Zip4<A: 0, B: 1, C: 2, D: 3>);
impl_zip_slice_api!(Zip5<A: 0, B: 1, C: 2, D: 3, E: 4>);
impl_zip_slice_api!(Zip6<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5>);
impl_zip_slice_api!(Zip7<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6>);

/// Allocated device storage that can be sliced back into algorithm views.
pub trait MStorage<R: Runtime>: StorageFromInner<R>
where
    Self::Item: MAlloc<R>,
{
    fn len(&self) -> MIndex;

    fn is_empty(&self) -> bool {
        MStorage::len(self) == 0
    }

    fn slice<Bounds>(&self, range: Bounds) -> impl MIter<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<MIndex>;

    fn slice_mut<Bounds>(&self, range: Bounds) -> impl MIterMut<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<MIndex>;

    #[doc(hidden)]
    fn into_alloc_view_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error>;
}

pub(crate) fn materialized_view_with_policy<R, Item>(
    policy: &crate::detail::CubePolicy<R>,
    inner: <Item as MAlloc<R>>::Inner,
) -> Result<<Item as MAlloc<R>>::View, Error>
where
    R: Runtime,
    Item: MAlloc<R>,
{
    let storage = Item::storage_from_inner(inner);
    storage.into_alloc_view_with_policy(policy)
}

/// Massively read iterator.
pub trait MIter<R: Runtime>: Sized {
    type Item: MItem<R>;

    fn len(&self) -> MIndex;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slice<Bounds>(&self, range: Bounds) -> MIterSlice<'_, Self>
    where
        Bounds: RangeBounds<MIndex>,
    {
        let range = normalize_zip_range(self.len(), range);
        MIterSlice::new(self, range)
    }

    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    type Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner;

    #[doc(hidden)]
    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error>;

    #[doc(hidden)]
    fn lower_read(self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        self.lower_read_ref(policy)
    }

    #[doc(hidden)]
    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error>;

    #[doc(hidden)]
    fn into_inner_with_policy(
        self,
        _policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Self::Inner, Error> {
        Ok(self.into_inner())
    }

    #[doc(hidden)]
    fn into_alloc_view_with_policy(
        self,
        _policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error>
    where
        Self::Item: MAlloc<R>,
    {
        Err(Error::Launch {
            message: "alloc view lowering is not supported for this iterator shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn stencil_selection_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
        invert: bool,
        flags_only: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
    where
        Self: MIter<R, Item = bool>,
        Self::Read: crate::detail::read::KernelStencilSelection<R>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelStencilSelection::stencil_selection(
            read, policy, invert, flags_only,
        )
    }

    #[doc(hidden)]
    fn transform_with_policy<Output, Op>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Output::Item: MAlloc<R> + dispatch::MItemDispatch<R>,
        Op: crate::op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::transform_read(read, policy, op, output)
    }

    #[doc(hidden)]
    fn transform_where_with_policy<Output, Op>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Output::Item: MAlloc<R> + dispatch::MItemDispatch<R>,
        Op: crate::op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::transform_where_read(read, policy, op, stencil, output)
    }

    #[doc(hidden)]
    fn count_if_with_policy<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<MIndex, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Pred: crate::op::PredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::count_if_read(read, policy, pred)
    }

    #[doc(hidden)]
    fn all_of_with_policy<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Pred: crate::op::PredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::all_of_read(read, policy, pred)
    }

    #[doc(hidden)]
    fn any_of_with_policy<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Pred: crate::op::PredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::any_of_read(read, policy, pred)
    }

    #[doc(hidden)]
    fn none_of_with_policy<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Pred: crate::op::PredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::none_of_read(read, policy, pred)
    }

    #[doc(hidden)]
    fn find_if_with_policy<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<MIndex>, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Pred: crate::op::PredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::find_if_read(read, policy, pred)
    }

    #[doc(hidden)]
    fn is_partitioned_with_policy<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Pred: crate::op::PredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::is_partitioned_read(read, policy, pred)
    }

    #[doc(hidden)]
    fn reduce_value_with_policy<Op>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Op: crate::op::ReductionOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        use crate::detail::read::KernelRead as _;
        read.validate()?;
        read.reduce_value_read(policy, init, op)
    }

    #[doc(hidden)]
    fn adjacent_find_with_policy<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<MIndex>, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Pred: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::adjacent_find_read(read, policy, pred)
    }

    #[doc(hidden)]
    fn equal_with_policy<Right, Eq>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        eq: Eq,
    ) -> Result<bool, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Right::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Eq: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let left = self.lower_read(policy)?;
        let right = right.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&left)?;
        crate::detail::read::KernelRead::validate(&right)?;
        crate::detail::read::KernelRead::equal_read(left, policy, right, eq)
    }

    #[doc(hidden)]
    fn find_first_of_with_policy<Needles, Eq>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        needles: Needles,
        eq: Eq,
    ) -> Result<Option<MIndex>, Error>
    where
        Needles: MIter<R, Item = Self::Item>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Needles::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Eq: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        let needles = needles.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::validate(&needles)?;
        crate::detail::read::KernelRead::find_first_of_read(read, policy, needles, eq)
    }

    #[doc(hidden)]
    fn is_sorted_with_policy<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<bool, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::is_sorted_read(read, policy, less)
    }

    #[doc(hidden)]
    fn is_sorted_until_with_policy<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<MIndex, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::is_sorted_until_read(read, policy, less)
    }

    #[doc(hidden)]
    fn lexicographical_compare_with_policy<Right, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        less: Less,
    ) -> Result<bool, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Right::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let left = self.lower_read(policy)?;
        let right = right.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&left)?;
        crate::detail::read::KernelRead::validate(&right)?;
        crate::detail::read::KernelRead::lexicographical_compare_read(left, policy, right, less)
    }

    #[doc(hidden)]
    fn lower_bound_many_with_policy<Values, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
    where
        Values: MIter<R, Item = Self::Item>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Values::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Self::Item: Send + Sync,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        let values = values.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::validate(&values)?;
        crate::detail::read::KernelReadBoundMany::lower_bound_many_logical(
            read, policy, values, less,
        )
    }

    #[doc(hidden)]
    fn upper_bound_many_with_policy<Values, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
    where
        Values: MIter<R, Item = Self::Item>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Values::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Self::Item: Send + Sync,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        let values = values.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::validate(&values)?;
        crate::detail::read::KernelReadBoundMany::upper_bound_many_logical(
            read, policy, values, less,
        )
    }

    #[doc(hidden)]
    fn max_element_with_policy<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::max_element_read(read, policy, less)
    }

    #[doc(hidden)]
    fn min_element_with_policy<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::min_element_read(read, policy, less)
    }

    #[doc(hidden)]
    fn minmax_element_with_policy<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<(MIndex, MIndex)>, Error>
    where
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let read = self.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        crate::detail::read::KernelRead::minmax_element_read(read, policy, less)
    }

    #[doc(hidden)]
    fn mismatch_with_policy<Right, Eq>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        eq: Eq,
    ) -> Result<Option<MIndex>, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Right::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Eq: crate::op::BinaryPredicateOp<R, Self::Item>,
    {
        let left = self.lower_read(policy)?;
        let right = right.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&left)?;
        crate::detail::read::KernelRead::validate(&right)?;
        crate::detail::read::KernelRead::mismatch_read(left, policy, right, eq)
    }

    #[doc(hidden)]
    fn exclusive_scan_by_key_with_policy<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: Values::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Values::Read: crate::detail::read::KernelReadBoundMany<R, Item = Values::Item>,
        Values::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Values::Item>,
        KeyEq: crate::op::BinaryPredicateOp<R, Self::Item>,
        Op: crate::op::ReductionOp<R, Values::Item>,
    {
        let keys = self.lower_read(policy)?;
        let values = values.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&keys)?;
        crate::detail::read::KernelRead::validate(&values)?;
        crate::detail::read::KernelRead::exclusive_scan_by_key_read(
            keys, policy, values, key_eq, init, op, output,
        )
    }

    #[doc(hidden)]
    fn inclusive_scan_by_key_with_policy<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        Self::Read: crate::detail::read::KernelReadBoundMany<R, Item = Self::Item>,
        Values::Read: crate::detail::read::KernelReadBoundMany<R, Item = Values::Item>,
        Values::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Values::Item>,
        KeyEq: crate::op::BinaryPredicateOp<R, Self::Item>,
        Op: crate::op::ReductionOp<R, Values::Item>,
    {
        let keys = self.lower_read(policy)?;
        let values = values.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&keys)?;
        crate::detail::read::KernelRead::validate(&values)?;
        crate::detail::read::KernelRead::inclusive_scan_by_key_read(
            keys, policy, values, key_eq, op, output,
        )
    }

    #[doc(hidden)]
    fn copy_selected_with_policy<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::copy_selected_from_view(policy, source, stencil, output)
    }

    #[doc(hidden)]
    fn gather_with_policy<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Self::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        let read = indices.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        <Self::Item as MAlloc<R>>::gather_from_view(policy, source, read, output)
    }

    #[doc(hidden)]
    fn gather_where_with_policy<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Self::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        let read = indices.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        <Self::Item as MAlloc<R>>::gather_where_from_view(policy, source, read, stencil, output)
    }

    #[doc(hidden)]
    fn scatter_with_policy<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Self::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        let read = indices.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        <Self::Item as MAlloc<R>>::scatter_from_view(policy, source, read, output)
    }

    #[doc(hidden)]
    fn scatter_where_with_policy<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Self::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        let read = indices.lower_read(policy)?;
        crate::detail::read::KernelRead::validate(&read)?;
        <Self::Item as MAlloc<R>>::scatter_where_from_view(policy, source, read, stencil, output)
    }

    #[doc(hidden)]
    fn unique_with_policy<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Pred: crate::op::BinaryPredicateOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::unique_from_view(policy, source, pred, output)
    }

    #[doc(hidden)]
    fn partition_with_policy<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Pred: crate::op::PredicateOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::partition_from_view(policy, source, pred, output)
    }

    #[doc(hidden)]
    fn adjacent_difference_with_policy<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        Op: crate::op::ReductionOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::adjacent_difference_from_view(policy, source, op, output)
    }

    #[doc(hidden)]
    fn inclusive_scan_with_policy<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        Op: crate::op::ReductionOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::inclusive_scan_from_view(policy, source, op, output)
    }

    #[doc(hidden)]
    fn exclusive_scan_with_policy<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        Op: crate::op::ReductionOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::exclusive_scan_from_view(policy, source, init, op, output)
    }

    #[doc(hidden)]
    fn unique_by_key_with_policy<Values, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        eq: Eq,
        out_k: KeyOutput,
        out_v: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Values: MIter<R, Item = ValueOutput::Item>,
        Eq: crate::op::BinaryPredicateOp<R, Self::Item>,
        KeyOutput: MIterMut<R, Item = Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: MAlloc<R>,
    {
        let keys = self.into_alloc_view_with_policy(policy)?;
        let values = values.into_alloc_view_with_policy(policy)?;
        let (keys, control) =
            <Self::Item as MAlloc<R>>::unique_by_key_control_from_view(policy, keys, eq)?;
        let len = crate::index::mindex_from_usize(control.count)?;
        out_k.write_prefix_from_inner(policy, keys)?;
        <ValueOutput::Item as MAlloc<R>>::unique_by_key_values_from_view(
            policy, values, &control, out_v,
        )?;
        Ok(len)
    }

    #[doc(hidden)]
    fn reduce_by_key_with_policy<Values, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: Values::Item,
        op: Op,
        out_k: KeyOutput,
        out_v: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Values: MIter<R, Item = ValueOutput::Item>,
        KeyEq: crate::op::BinaryPredicateOp<R, Self::Item>,
        Op: crate::op::ReductionOp<R, Values::Item>,
        KeyOutput: MIterMut<R, Item = Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: MAlloc<R>,
    {
        let keys = self.into_alloc_view_with_policy(policy)?;
        let values = values.into_alloc_view_with_policy(policy)?;
        let (keys, control) =
            <Self::Item as MAlloc<R>>::reduce_by_key_control_from_view(policy, keys, key_eq)?;
        let len = crate::index::mindex_from_usize(control.output_count)?;
        out_k.write_prefix_from_inner(policy, keys)?;
        <ValueOutput::Item as MAlloc<R>>::reduce_by_key_values_from_view::<KeyEq, Op, _>(
            policy, values, &control, init, op, out_v,
        )?;
        Ok(len)
    }

    #[doc(hidden)]
    fn reverse_with_policy<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::reverse_from_view(policy, source, output)
    }

    #[doc(hidden)]
    fn sort_with_policy<Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let source = self.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::sort_from_view(policy, source, less, output)
    }

    #[doc(hidden)]
    fn sort_by_key_with_policy<Values, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        less: Less,
        out_k: KeyOutput,
        out_v: ValueOutput,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        Values: MIter<R, Item = ValueOutput::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
        KeyOutput: MIterMut<R, Item = Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: MAlloc<R>,
    {
        let keys = self.into_alloc_view_with_policy(policy)?;
        let values = values.into_alloc_view_with_policy(policy)?;
        let (keys, indices) =
            <Self::Item as MAlloc<R>>::sort_by_key_control_from_view(policy, keys, less)?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        out_k.write_from_inner(policy, keys)?;
        <ValueOutput::Item as MAlloc<R>>::sort_by_key_values_from_view(
            policy,
            values,
            control.permutation(),
            out_v,
        )
    }

    #[doc(hidden)]
    fn merge_with_policy<Right, Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        Right: MIter<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let left = self.into_alloc_view_with_policy(policy)?;
        let right = right.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::merge_from_views(policy, left, right, less, output)
    }

    #[doc(hidden)]
    fn merge_by_key_with_policy<LeftValues, RightKeys, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_values: LeftValues,
        right_keys: RightKeys,
        right_values: RightValues,
        less: Less,
        out_k: KeyOutput,
        out_v: ValueOutput,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>,
        LeftValues: MIter<R, Item = ValueOutput::Item>,
        RightKeys: MIter<R, Item = Self::Item>,
        RightValues: MIter<R, Item = LeftValues::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
        KeyOutput: MIterMut<R, Item = Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: MAlloc<R>,
    {
        let left_keys = self.into_alloc_view_with_policy(policy)?;
        let right_keys = right_keys.into_alloc_view_with_policy(policy)?;
        let left_values = left_values.into_alloc_view_with_policy(policy)?;
        let right_values = right_values.into_alloc_view_with_policy(policy)?;
        let (keys, control) = <Self::Item as MAlloc<R>>::merge_by_key_control_from_views(
            policy, left_keys, right_keys, less,
        )?;
        out_k.write_from_inner(policy, keys)?;
        <ValueOutput::Item as MAlloc<R>>::merge_by_key_values_from_views(
            policy,
            left_values,
            right_values,
            &control,
            out_v,
        )
    }

    #[doc(hidden)]
    fn set_difference_with_policy<Right, Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Right: MIter<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let left = self.into_alloc_view_with_policy(policy)?;
        let right = right.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::set_difference_from_views(policy, left, right, less, output)
    }

    #[doc(hidden)]
    fn set_intersection_with_policy<Right, Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Right: MIter<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let left = self.into_alloc_view_with_policy(policy)?;
        let right = right.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::set_intersection_from_views(policy, left, right, less, output)
    }

    #[doc(hidden)]
    fn set_union_with_policy<Right, Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self::Item: MAlloc<R>,
        Right: MIter<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<R, Self::Item>,
        Output: MIterMut<R, Item = Self::Item>,
    {
        let left = self.into_alloc_view_with_policy(policy)?;
        let right = right.into_alloc_view_with_policy(policy)?;
        <Self::Item as MAlloc<R>>::set_union_from_views(policy, left, right, less, output)
    }
}

impl<R, Iter> MIter<R> for MIterSlice<'_, Iter>
where
    R: Runtime,
    Iter: MIter<R>,
    crate::detail::read::SliceRead<Iter::Read>:
        crate::detail::read::KernelReadBoundMany<R, Item = Iter::Item>,
{
    type Item = Iter::Item;
    type Inner = Iter::Inner;
    type Read = crate::detail::read::SliceRead<Iter::Read>;

    fn len(&self) -> MIndex {
        self.range.len()
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("sliced read-only MIter lowering requires a CubePolicy")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        Ok(crate::detail::read::SliceRead::new(
            self.inner.lower_read_ref(policy)?,
            self.range.start as usize,
            self.range.len() as usize,
        ))
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.inner.validate_executor(exec)
    }
}

/// Massively mutable iterator used as an explicit algorithm destination.
pub trait MIterMut<R: Runtime>: Sized {
    type Item: MAlloc<R>;

    fn len(&self) -> MIndex;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slice<Bounds>(&self, range: Bounds) -> MIterMutSlice<'_, Self>
    where
        Bounds: RangeBounds<MIndex>,
    {
        let range = normalize_zip_range(self.len(), range);
        MIterMutSlice::new(self, range)
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> MIterMutSliceMut<'_, Self>
    where
        Bounds: RangeBounds<MIndex>,
    {
        let range = normalize_zip_range(self.len(), range);
        MIterMutSliceMut::new(self, range)
    }

    #[doc(hidden)]
    fn validate_executor(&self, _exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        Ok(())
    }

    #[doc(hidden)]
    fn column_mut_view_inner<T: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, T>>, Error>
    where
        T: crate::value::MStorageElement,
    {
        Ok(None)
    }

    #[doc(hidden)]
    fn column_mut_view_by_index_inner<T: 'static>(
        &self,
        index: usize,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, T>>, Error>
    where
        T: crate::value::MStorageElement,
    {
        if index == 0 {
            self.column_mut_view_inner::<T>()
        } else {
            Ok(None)
        }
    }

    #[doc(hidden)]
    type Inner: SlicedOutputInner<R, Self::Item>;

    #[doc(hidden)]
    fn inner(&self) -> Self::Inner;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner {
        self.inner()
    }

    #[doc(hidden)]
    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>;

    #[doc(hidden)]
    fn write_prefix_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>;

    #[doc(hidden)]
    fn write_split_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        selected: <Self::Item as MAlloc<R>>::Inner,
        rejected: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>;

    #[doc(hidden)]
    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error>
    where
        Self::Item: MAlloc<R>;

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
}

#[doc(hidden)]
pub trait SlicedOutputInner<R: Runtime, Item: MAlloc<R>>: Sized {
    type Read: crate::detail::read::KernelReadBoundMany<R, Item = Item>;

    fn slice_inner(self, range: Range<MIndex>) -> Self;
    fn into_read(self) -> Self::Read;
    fn into_alloc_view(self) -> Item::View;
    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: Item::Inner,
    ) -> Result<(), Error>;
    fn write_prefix_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: Item::Inner,
    ) -> Result<(), Error>;
    fn write_split_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        selected: Item::Inner,
        rejected: Item::Inner,
    ) -> Result<(), Error>;
    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: Item::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error>;
    fn replace_where_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        replacement: Item,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error>;
    fn fill_inner(self, policy: &crate::detail::CubePolicy<R>, value: Item) -> Result<(), Error>;
}

impl<R, Output> MIterMut<R> for MIterMutSliceMut<'_, Output>
where
    R: Runtime,
    Output: MIterMut<R>,
    Output::Inner: SlicedOutputInner<R, Output::Item>,
{
    type Item = Output::Item;
    type Inner = Output::Inner;

    fn len(&self) -> MIndex {
        self.range.len()
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.inner.validate_executor(exec)
    }

    fn column_mut_view_inner<T: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, T>>, Error>
    where
        T: crate::value::MStorageElement,
    {
        let Some(mut view) = self.inner.column_mut_view_inner::<T>()? else {
            return Ok(None);
        };
        view.offset += self.range.start as usize;
        view.len = self.range.len() as usize;
        Ok(Some(view))
    }

    fn column_mut_view_by_index_inner<T: 'static>(
        &self,
        index: usize,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, T>>, Error>
    where
        T: crate::value::MStorageElement,
    {
        let Some(mut view) = self.inner.column_mut_view_by_index_inner::<T>(index)? else {
            return Ok(None);
        };
        view.offset += self.range.start as usize;
        view.len = self.range.len() as usize;
        Ok(Some(view))
    }

    fn inner(&self) -> Self::Inner {
        self.inner.inner().slice_inner(self.range.to_range())
    }

    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        SlicedOutputInner::write_from_inner(MIterMut::into_inner(self), policy, inner)
    }

    fn write_prefix_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        SlicedOutputInner::write_prefix_from_inner(MIterMut::into_inner(self), policy, inner)
    }

    fn write_split_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        selected: <Self::Item as MAlloc<R>>::Inner,
        rejected: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        SlicedOutputInner::write_split_from_inner(
            MIterMut::into_inner(self),
            policy,
            selected,
            rejected,
        )
    }

    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error> {
        SlicedOutputInner::write_where_from_inner(
            MIterMut::into_inner(self),
            policy,
            inner,
            stencil,
        )
    }

    fn replace_where_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        replacement: Self::Item,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error> {
        SlicedOutputInner::replace_where_inner(
            MIterMut::into_inner(self),
            policy,
            replacement,
            stencil,
        )
    }

    fn fill_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: Self::Item,
    ) -> Result<(), Error> {
        SlicedOutputInner::fill_inner(MIterMut::into_inner(self), policy, value)
    }
}

impl<R, Output> MIter<R> for MIterMutSlice<'_, Output>
where
    R: Runtime,
    Output: MIterMut<R>,
    Output::Inner: SlicedOutputInner<R, Output::Item>,
{
    type Item = Output::Item;
    type Inner = Output::Inner;
    type Read = <Output::Inner as SlicedOutputInner<R, Output::Item>>::Read;

    fn len(&self) -> MIndex {
        self.range.len()
    }

    fn into_inner(self) -> Self::Inner {
        self.inner.inner().slice_inner(self.range.to_range())
    }

    fn lower_read_ref(&self, _policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        Ok(SlicedOutputInner::into_read(
            self.inner.inner().slice_inner(self.range.to_range()),
        ))
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.inner.validate_executor(exec)
    }

    fn into_alloc_view_with_policy(
        self,
        _policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error>
    where
        Self::Item: MAlloc<R>,
    {
        Ok(SlicedOutputInner::into_alloc_view(self.into_inner()))
    }
}
