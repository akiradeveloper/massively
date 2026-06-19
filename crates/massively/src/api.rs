//! Public API for `massively`.
//!
//! This crate intentionally keeps CubeCL runtime types out of public algorithm
//! signatures. The implementation delegates to the internal detail layer.

use std::any::Any;
use std::marker::PhantomData;

pub use crate::detail::Error;
pub use crate::detail::op;

mod sealed {
    use super::{Error, MIter, MVec, op};

    pub trait Backend {
        type Runtime: cubecl::prelude::Runtime;
    }

    pub trait Value: cubecl::prelude::CubeType {}
    impl<T> Value for T where T: cubecl::prelude::CubeType {}

    pub trait Scalar:
        Value + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement
    {
    }
    impl<T> Scalar for T where T: cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement {}

    pub trait MIterDispatch<B: super::Backend>: Sized {
        fn index_inner(
            &self,
        ) -> Option<(&crate::detail::DeviceVec<<B as Backend>::Runtime, u32>,)> {
            None
        }

        fn column_inner<T: 'static>(
            &self,
        ) -> Option<&crate::detail::DeviceVec<<B as Backend>::Runtime, T>> {
            None
        }

        fn transform_dispatch<Op, Output, Y>(self, op: Op) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
            Y: super::StorageOutput<B>,
            Output: MVec<B, Item = Y>;

        fn reverse_dispatch<Output>(self) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn sort_dispatch<Less, Output>(self, less: Less) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn gather_dispatch<Indices, Output>(self, indices: Indices) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn reduce_dispatch<Op>(
            self,
            init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<<Self as MIter<B>>::Item, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>;

        fn inclusive_scan_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn exclusive_scan_dispatch<Op, Output>(
            self,
            init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn adjacent_difference_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn copy_if_dispatch<Stencil, StencilScalar, Pred, Output>(
            self,
            stencil: Stencil,
            pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Stencil: MIter<B, Item = (StencilScalar,)>,
            StencilScalar: super::Scalar<B> + 'static,
            Pred: op::PredicateOp<(StencilScalar,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn remove_if_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn count_if_dispatch<Pred>(self, pred: Pred) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn all_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn any_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn none_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn find_if_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn partition_dispatch<Pred, Output>(self, pred: Pred) -> Result<(Output, Output), Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn is_partitioned_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn replace_if_dispatch<Stencil, StencilScalar, Pred, Output>(
            self,
            replacement: <Self as MIter<B>>::Item,
            stencil: Stencil,
            pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Stencil: MIter<B, Item = (StencilScalar,)>,
            StencilScalar: super::Scalar<B> + 'static,
            Pred: op::PredicateOp<(StencilScalar,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn unique_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;
    }

    pub trait StorageOutputDispatch<B: super::Backend>: Sized {
        fn transform_unary<Input, Op>(
            input: &crate::detail::DeviceVec<<B as Backend>::Runtime, Input>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            Input: super::Scalar<B>,
            Op: op::UnaryOp<(Input,), Output = Self>;

        fn transform_binary<Left, Right, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            left: &crate::detail::DeviceVec<<B as Backend>::Runtime, Left>,
            right: &crate::detail::DeviceVec<<B as Backend>::Runtime, Right>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            Left: super::Scalar<B>,
            Right: super::Scalar<B>,
            Op: op::UnaryOp<(Left, Right), Output = Self>;

        fn transform_ternary<First, Second, Third, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            first: &crate::detail::DeviceVec<<B as Backend>::Runtime, First>,
            second: &crate::detail::DeviceVec<<B as Backend>::Runtime, Second>,
            third: &crate::detail::DeviceVec<<B as Backend>::Runtime, Third>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            First: super::Scalar<B>,
            Second: super::Scalar<B>,
            Third: super::Scalar<B>,
            Op: op::UnaryOp<(First, Second, Third), Output = Self>;
    }
}

/// Execution backend marker.
///
/// Backend implementations hide the CubeCL runtime type used by the lower
/// implementation layer.
pub trait Backend: sealed::Backend + Copy + Clone + Default + 'static {}

/// Value that can appear as one logical device item.
pub trait Value<B: Backend>: sealed::Value {}
impl<B, T> Value<B> for T
where
    B: Backend,
    T: sealed::Value,
{
}

/// Scalar value that can be stored in one device column.
pub trait Scalar<B: Backend>: Value<B> + sealed::Scalar {}
impl<B, T> Scalar<B> for T
where
    B: Backend,
    T: Value<B> + sealed::Scalar,
{
}

/// WGPU backend marker.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, Default)]
pub struct Wgpu;

#[cfg(feature = "wgpu")]
impl sealed::Backend for Wgpu {
    type Runtime = cubecl::wgpu::WgpuRuntime;
}

#[cfg(feature = "wgpu")]
impl Backend for Wgpu {}

/// Compatibility alias for the default WGPU execution policy.
#[cfg(feature = "wgpu")]
pub type CubeWgpu = Policy<Wgpu>;

/// Execution policy for a facade backend.
#[derive(Debug)]
pub struct Policy<B: Backend> {
    inner: crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> Clone for Policy<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Policy<B> {
    fn from_inner(inner: crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    /// Copies host data to device-resident storage.
    pub fn to_device<T>(&self, input: &[T]) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar<B>,
    {
        Ok(DeviceVec::from_inner(self.inner.to_device(input)?))
    }

    /// Waits until all previously submitted work for this policy is complete.
    pub fn sync(&self) -> Result<(), Error> {
        futures_lite::future::block_on(self.inner.client().sync()).map_err(|err| Error::Launch {
            message: err.to_string(),
        })
    }
}

#[cfg(feature = "wgpu")]
impl Policy<Wgpu> {
    /// Creates a WGPU policy backed by the default device.
    pub fn new() -> Self {
        Self::from_inner(crate::detail::CubeWgpu::new())
    }

    /// Creates a WGPU policy backed by the CPU adapter.
    pub fn cpu() -> Self {
        Self::from_inner(crate::detail::CubeWgpu::cpu())
    }
}

#[cfg(feature = "wgpu")]
impl Default for Policy<Wgpu> {
    fn default() -> Self {
        Self::new()
    }
}

/// Owned device column.
#[derive(Debug)]
pub struct DeviceVec<B: Backend, T> {
    inner: crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,
    _backend: PhantomData<fn() -> B>,
}

impl<B, T> DeviceVec<B, T>
where
    B: Backend,
{
    fn from_inner(inner: crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether this column is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Copies this device column to host memory.
    pub fn to_vec(&self) -> Result<Vec<T>, Error>
    where
        T: Scalar<B>,
    {
        self.inner.to_vec()
    }
}

/// Storage materialization for owned outputs.
pub trait StorageOutput<B: Backend>: sealed::StorageOutputDispatch<B> + Value<B> + Sized {
    #[doc(hidden)]
    type Inner;
}

/// Owned massively vector for a logical item.
pub trait MVec<B: Backend>: Sized {
    type Item: StorageOutput<B>;

    #[doc(hidden)]
    fn from_inner(inner: <Self::Item as StorageOutput<B>>::Inner) -> Self;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this array has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn array_from_inner<B, Item, Output>(inner: <Item as StorageOutput<B>>::Inner) -> Output
where
    B: Backend,
    Item: StorageOutput<B>,
    Output: MVec<B, Item = Item>,
{
    <Output as MVec<B>>::from_inner(inner)
}

fn gather_index_inner<B, Indices>(
    indices: &Indices,
) -> Result<(&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, u32>,), Error>
where
    B: Backend,
    Indices: MIter<B, Item = (u32,)>,
{
    <Indices as sealed::MIterDispatch<B>>::index_inner(indices).ok_or_else(|| Error::Launch {
        message: "gather indices must be backed by one u32 DeviceVec".to_string(),
    })
}

macro_rules! impl_storage_output_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<B, $( $ty ),+> StorageOutput<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
        {
            type Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as StorageOutput<B>>::Inner) -> Self {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl<B, $( $ty ),+> sealed::StorageOutputDispatch<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
        {
            fn transform_unary<Input, Op>(
                input: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Input>,
                op: Op,
            ) -> Result<<Self as StorageOutput<B>>::Inner, Error>
            where
                Input: Scalar<B>,
                Op: op::UnaryOp<(Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    <B as sealed::Backend>::Runtime,
                    Input,
                    Op,
                >,
                <Self as crate::detail::StorageOutput<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformUnaryOutput<
                        <B as sealed::Backend>::Runtime,
                        Input,
                        Op,
                    >>::run(input)?;
                crate::detail::MaterializeOutput::materialize_output(storage)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Left>,
                right: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Right>,
                op: Op,
            ) -> Result<<Self as StorageOutput<B>>::Inner, Error>
            where
                Left: Scalar<B>,
                Right: Scalar<B>,
                Op: op::UnaryOp<(Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    <B as sealed::Backend>::Runtime,
                    Left,
                    Right,
                    Op,
                >,
                <Self as crate::detail::StorageOutput<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA2Output<
                        <B as sealed::Backend>::Runtime,
                        Left,
                        Right,
                        Op,
                    >>::run(policy, left, right)?;
                crate::detail::MaterializeOutput::materialize_output(storage)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                first: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, First>,
                second: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Second>,
                third: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Third>,
                op: Op,
            ) -> Result<<Self as StorageOutput<B>>::Inner, Error>
            where
                First: Scalar<B>,
                Second: Scalar<B>,
                Third: Scalar<B>,
                Op: op::UnaryOp<(First, Second, Third), Output = Self>,
                Self: crate::detail::TransformSoA3Output<
                    <B as sealed::Backend>::Runtime,
                    First,
                    Second,
                    Third,
                    Op,
                >,
                <Self as crate::detail::StorageOutput<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA3Output<
                        <B as sealed::Backend>::Runtime,
                        First,
                        Second,
                        Third,
                        Op,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                    )?;
                crate::detail::MaterializeOutput::materialize_output(storage)
            }
        }
    };
}

impl_storage_output_tuple!(A: a);
impl_storage_output_tuple!(A: a, B0: b);
impl_storage_output_tuple!(A: a, B0: b, C: c);

/// Massively iterator.
pub trait MIter<B: Backend>: sealed::MIterDispatch<B> + Sized {
    type Item: StorageOutput<B>;

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

impl<'a, B, T> MIter<B> for &'a DeviceVec<B, T>
where
    B: Backend,
    T: Scalar<B> + 'static,
    (&'a DeviceVec<B, T>,): MIter<B, Item = (T,)>,
{
    type Item = (T,);
    type Inner = <(&'a DeviceVec<B, T>,) as MIter<B>>::Inner;

    fn len(&self) -> usize {
        <(&'a DeviceVec<B, T>,) as MIter<B>>::len(&(*self,))
    }

    fn into_inner(self) -> Self::Inner {
        <(&'a DeviceVec<B, T>,) as MIter<B>>::into_inner((self,))
    }
}

impl<'a, B, T> sealed::MIterDispatch<B> for &'a DeviceVec<B, T>
where
    B: Backend,
    T: Scalar<B> + 'static,
    (&'a DeviceVec<B, T>,): sealed::MIterDispatch<B> + MIter<B, Item = (T,)>,
{
    fn index_inner(
        &self,
    ) -> Option<(&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, u32>,)> {
        let column = *self as &dyn Any;
        let column = column.downcast_ref::<DeviceVec<B, u32>>()?;
        Some((&column.inner,))
    }

    fn column_inner<U: 'static>(
        &self,
    ) -> Option<&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, U>> {
        let column = *self as &dyn Any;
        let column = column.downcast_ref::<DeviceVec<B, U>>()?;
        Some(&column.inner)
    }

    fn transform_dispatch<Op, Output, Y>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
        Y: StorageOutput<B>,
        Output: MVec<B, Item = Y>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::transform_dispatch((self,), op)
    }

    fn reverse_dispatch<Output>(self) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::reverse_dispatch((self,))
    }

    fn sort_dispatch<Less, Output>(self, less: Less) -> Result<Output, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::sort_dispatch((self,), less)
    }

    fn gather_dispatch<Indices, Output>(self, indices: Indices) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::gather_dispatch((self,), indices)
    }

    fn reduce_dispatch<Op>(
        self,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::reduce_dispatch((self,), init, op)
    }

    fn inclusive_scan_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::inclusive_scan_dispatch((self,), op)
    }

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::exclusive_scan_dispatch(
            (self,),
            init,
            op,
        )
    }

    fn adjacent_difference_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::adjacent_difference_dispatch(
            (self,),
            op,
        )
    }

    fn copy_if_dispatch<Stencil, StencilScalar, Pred, Output>(
        self,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B, Item = (StencilScalar,)>,
        StencilScalar: Scalar<B> + 'static,
        Pred: op::PredicateOp<(StencilScalar,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::copy_if_dispatch(
            (self,),
            stencil,
            pred,
        )
    }

    fn remove_if_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::remove_if_dispatch((self,), pred)
    }

    fn count_if_dispatch<Pred>(self, pred: Pred) -> Result<usize, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::count_if_dispatch((self,), pred)
    }

    fn all_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::all_of_dispatch((self,), pred)
    }

    fn any_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::any_of_dispatch((self,), pred)
    }

    fn none_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::none_of_dispatch((self,), pred)
    }

    fn find_if_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::find_if_dispatch((self,), pred)
    }

    fn partition_dispatch<Pred, Output>(self, pred: Pred) -> Result<(Output, Output), Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::partition_dispatch((self,), pred)
    }

    fn is_partitioned_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::is_partitioned_dispatch((self,), pred)
    }

    fn replace_if_dispatch<Stencil, StencilScalar, Pred, Output>(
        self,
        replacement: <Self as MIter<B>>::Item,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B, Item = (StencilScalar,)>,
        StencilScalar: Scalar<B> + 'static,
        Pred: op::PredicateOp<(StencilScalar,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::replace_if_dispatch(
            (self,),
            replacement,
            stencil,
            pred,
        )
    }

    fn unique_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
    where
        Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        <(&'a DeviceVec<B, T>,) as sealed::MIterDispatch<B>>::unique_dispatch((self,), pred)
    }
}

impl<'a, B, T> MIter<B> for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    (T,): StorageOutput<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
{
    type Item = (T,);
    type Inner = (&'a crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,);

    fn len(&self) -> usize {
        self.0.inner.len()
    }

    fn into_inner(self) -> Self::Inner {
        (&self.0.inner,)
    }
}

impl<'a, B, T> sealed::MIterDispatch<B> for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    (T,): StorageOutput<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
{
    fn index_inner(
        &self,
    ) -> Option<(&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, u32>,)> {
        let column = self.0 as &dyn Any;
        let column = column.downcast_ref::<DeviceVec<B, u32>>()?;
        Some((&column.inner,))
    }

    fn column_inner<U: 'static>(
        &self,
    ) -> Option<&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, U>> {
        let column = self.0 as &dyn Any;
        let column = column.downcast_ref::<DeviceVec<B, U>>()?;
        Some(&column.inner)
    }

    fn transform_dispatch<Op, Output, Y>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
        Y: StorageOutput<B>,
        Output: MVec<B, Item = Y>,
    {
        let input = self.into_inner();
        let inner = <Y as sealed::StorageOutputDispatch<B>>::transform_unary(input.0, op)?;
        Ok(array_from_inner::<B, Y, Output>(inner))
    }

    fn reverse_dispatch<Output>(self) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::reverse(self.into_inner())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn sort_dispatch<Less, Output>(self, _less: Less) -> Result<Output, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::sort(self.into_inner(), _less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn gather_dispatch<Indices, Output>(self, indices: Indices) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(&indices)?;
        let inner = crate::detail::gather(self.into_inner(), indices)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn reduce_dispatch<Op>(
        self,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::reduce(self.into_inner(), init, op)
    }

    fn inclusive_scan_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::inclusive_scan(self.into_inner(), op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::exclusive_scan(self.into_inner(), init, op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn adjacent_difference_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::adjacent_difference(self.into_inner(), op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn copy_if_dispatch<Stencil, StencilScalar, Pred, Output>(
        self,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B, Item = (StencilScalar,)>,
        StencilScalar: Scalar<B> + 'static,
        Pred: op::PredicateOp<(StencilScalar,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let stencil =
            <Stencil as sealed::MIterDispatch<B>>::column_inner::<StencilScalar>(&stencil)
                .ok_or_else(|| Error::Launch {
                    message: "copy_if stencil must be backed by one DeviceVec".to_string(),
                })?;
        let inner = crate::detail::copy_if(self.into_inner(), (stencil,), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn remove_if_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::remove_if(self.into_inner(), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn count_if_dispatch<Pred>(self, pred: Pred) -> Result<usize, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::count_if(self.into_inner(), pred)
    }

    fn all_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::all_of(self.into_inner(), pred)
    }

    fn any_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::any_of(self.into_inner(), pred)
    }

    fn none_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::none_of(self.into_inner(), pred)
    }

    fn find_if_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::find_if(self.into_inner(), pred)
    }

    fn partition_dispatch<Pred, Output>(self, pred: Pred) -> Result<(Output, Output), Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (matching, failing) = crate::detail::partition(self.into_inner(), pred)?;
        Ok((
            array_from_inner::<B, (T,), Output>(matching),
            array_from_inner::<B, (T,), Output>(failing),
        ))
    }

    fn is_partitioned_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::is_partitioned(self.into_inner(), pred)
    }

    fn replace_if_dispatch<Stencil, StencilScalar, Pred, Output>(
        self,
        replacement: <Self as MIter<B>>::Item,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B, Item = (StencilScalar,)>,
        StencilScalar: Scalar<B> + 'static,
        Pred: op::PredicateOp<(StencilScalar,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let stencil =
            <Stencil as sealed::MIterDispatch<B>>::column_inner::<StencilScalar>(&stencil)
                .ok_or_else(|| Error::Launch {
                    message: "replace_if stencil must be backed by one DeviceVec".to_string(),
                })?;
        let inner = crate::detail::replace_if(self.into_inner(), replacement, (stencil,), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn unique_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
    where
        Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::unique(self.into_inner(), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }
}

macro_rules! impl_miter_tuple {
    ($( $ty:ident : $idx:tt ),+ => $transform:ident) => {
        impl<'a, B, $( $ty ),+> MIter<B> for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( &'a crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);

            fn len(&self) -> usize {
                self.0.inner.len()
            }

            fn into_inner(self) -> Self::Inner {
                ($( &self.$idx.inner, )+)
            }
        }

        impl<'a, B, $( $ty ),+> sealed::MIterDispatch<B> for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            fn transform_dispatch<Op, Output, Y>(self, op: Op) -> Result<Output, Error>
            where
                Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
                Y: StorageOutput<B>,
                Output: MVec<B, Item = Y>,
            {
                let inner_input = self.into_inner();
                let inner = <Y as sealed::StorageOutputDispatch<B>>::$transform(
                    inner_input.0.policy(),
                    $( inner_input.$idx, )+
                    op,
                )?;
                Ok(array_from_inner::<B, Y, Output>(inner))
            }

            fn reverse_dispatch<Output>(self) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::reverse(self.into_inner())?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn sort_dispatch<Less, Output>(self, less: Less) -> Result<Output, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::sort(self.into_inner(), less)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn gather_dispatch<Indices, Output>(self, indices: Indices) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices = gather_index_inner::<B, Indices>(&indices)?;
                let inner = crate::detail::gather(self.into_inner(), indices)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn reduce_dispatch<Op>(
                self,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<<Self as MIter<B>>::Item, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::reduce(self.into_inner(), init, op)
            }

            fn inclusive_scan_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::inclusive_scan(self.into_inner(), op)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn exclusive_scan_dispatch<Op, Output>(
                self,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::exclusive_scan(self.into_inner(), init, op)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn adjacent_difference_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::adjacent_difference(self.into_inner(), op)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn copy_if_dispatch<Stencil, StencilScalar, Pred, Output>(
                self,
                stencil: Stencil,
                pred: Pred,
            ) -> Result<Output, Error>
            where
                Stencil: MIter<B, Item = (StencilScalar,)>,
                StencilScalar: Scalar<B> + 'static,
                Pred: op::PredicateOp<(StencilScalar,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let stencil =
                    <Stencil as sealed::MIterDispatch<B>>::column_inner::<StencilScalar>(&stencil)
                        .ok_or_else(|| Error::Launch {
                            message: "copy_if stencil must be backed by one DeviceVec".to_string(),
                        })?;
                let inner = crate::detail::copy_if(self.into_inner(), stencil, pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn remove_if_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::remove_if(self.into_inner(), pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn count_if_dispatch<Pred>(self, pred: Pred) -> Result<usize, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::count_if(self.into_inner(), pred)
            }

            fn all_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::all_of(self.into_inner(), pred)
            }

            fn any_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::any_of(self.into_inner(), pred)
            }

            fn none_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::none_of(self.into_inner(), pred)
            }

            fn find_if_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::find_if(self.into_inner(), pred)
            }

            fn partition_dispatch<Pred, Output>(self, pred: Pred) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let (matching, failing) = crate::detail::partition(self.into_inner(), pred)?;
                Ok((
                    array_from_inner::<B, ($( $ty, )+), Output>(matching),
                    array_from_inner::<B, ($( $ty, )+), Output>(failing),
                ))
            }

            fn is_partitioned_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::is_partitioned(self.into_inner(), pred)
            }

            fn replace_if_dispatch<Stencil, StencilScalar, Pred, Output>(
                self,
                replacement: <Self as MIter<B>>::Item,
                stencil: Stencil,
                pred: Pred,
            ) -> Result<Output, Error>
            where
                Stencil: MIter<B, Item = (StencilScalar,)>,
                StencilScalar: Scalar<B> + 'static,
                Pred: op::PredicateOp<(StencilScalar,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let stencil =
                    <Stencil as sealed::MIterDispatch<B>>::column_inner::<StencilScalar>(&stencil)
                        .ok_or_else(|| Error::Launch {
                            message: "replace_if stencil must be backed by one DeviceVec".to_string(),
                        })?;
                let inner = crate::detail::replace_if(self.into_inner(), replacement, stencil, pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn unique_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::unique(self.into_inner(), pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }
        }
    };
}

impl_miter_tuple!(A: 0, C: 1 => transform_binary);
impl_miter_tuple!(A: 0, C: 1, D: 2 => transform_ternary);

/// Applies a unary transform to a massively iterator.
pub fn transform<B, Input, Output, Op>(source: Input, op: Op) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B>,
    Op: op::UnaryOp<Input::Item, Output = Output::Item>,
{
    <Input as sealed::MIterDispatch<B>>::transform_dispatch(source, op)
}

/// Reverses a massively iterator into an owned vector.
pub fn reverse<B, Input, Output>(source: Input) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::reverse_dispatch(source)
}

/// Sorts a massively iterator into an owned vector.
pub fn sort<B, Input, Output, Less>(source: Input, less: Less) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::sort_dispatch(source, less)
}

/// Gathers a massively iterator at index positions into an owned vector.
pub fn gather<B, Input, Indices, Output>(source: Input, indices: Indices) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::gather_dispatch(source, indices)
}

/// Reduces a massively iterator to one host item.
pub fn reduce<B, Input, Op>(source: Input, init: Input::Item, op: Op) -> Result<Input::Item, Error>
where
    B: Backend,
    Input: MIter<B>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::reduce_dispatch(source, init, op)
}

/// Computes an inclusive scan.
pub fn inclusive_scan<B, Input, Output, Op>(source: Input, op: Op) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::inclusive_scan_dispatch(source, op)
}

/// Computes an exclusive scan.
pub fn exclusive_scan<B, Input, Output, Op>(
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::exclusive_scan_dispatch(source, init, op)
}

/// Computes adjacent differences.
pub fn adjacent_difference<B, Input, Output, Op>(source: Input, op: Op) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::adjacent_difference_dispatch(source, op)
}

/// Copies elements whose stencil value satisfies `pred`.
pub fn copy_if<B, Input, Stencil, StencilScalar, Output, Pred>(
    source: Input,
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Stencil: MIter<B, Item = (StencilScalar,)>,
    StencilScalar: Scalar<B> + 'static,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<(StencilScalar,)>,
{
    <Input as sealed::MIterDispatch<B>>::copy_if_dispatch(source, stencil, pred)
}

/// Removes elements satisfying `pred`.
pub fn remove_if<B, Input, Output, Pred>(source: Input, pred: Pred) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::remove_if_dispatch(source, pred)
}

/// Counts elements satisfying `pred`.
pub fn count_if<B, Input, Pred>(source: Input, pred: Pred) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::count_if_dispatch(source, pred)
}

/// Returns whether all elements satisfy `pred`.
pub fn all_of<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::all_of_dispatch(source, pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::any_of_dispatch(source, pred)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::none_of_dispatch(source, pred)
}

/// Finds the first element satisfying `pred`.
pub fn find_if<B, Input, Pred>(source: Input, pred: Pred) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::find_if_dispatch(source, pred)
}

/// Partitions elements by `pred`.
pub fn partition<B, Input, Output, Pred>(
    source: Input,
    pred: Pred,
) -> Result<(Output, Output), Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::partition_dispatch(source, pred)
}

/// Returns whether input is partitioned by `pred`.
pub fn is_partitioned<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::is_partitioned_dispatch(source, pred)
}

/// Replaces elements whose stencil value satisfies `pred`.
pub fn replace_if<B, Input, Stencil, StencilScalar, Output, Pred>(
    source: Input,
    replacement: Input::Item,
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Stencil: MIter<B, Item = (StencilScalar,)>,
    StencilScalar: Scalar<B> + 'static,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<(StencilScalar,)>,
{
    <Input as sealed::MIterDispatch<B>>::replace_if_dispatch(source, replacement, stencil, pred)
}

/// Removes consecutive duplicates under `pred`.
pub fn unique<B, Input, Output, Pred>(source: Input, pred: Pred) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::unique_dispatch(source, pred)
}

/// Stable sort. The current lower implementation is stable.
pub fn stable_sort<B, Input, Output, Less>(source: Input, less: Less) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    sort(source, less)
}

/// Gathers selected elements.
pub fn gather_if<B, Input, Indices, Stencil, T, S, Output, Pred>(
    source: Input,
    indices: Indices,
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    Indices: MIter<B, Item = (u32,)>,
    Stencil: MIter<B, Item = (S,)>,
    T: Scalar<B> + Default + 'static,
    S: Scalar<B> + 'static,
    Output: MVec<B, Item = (T,)>,
    Pred: op::PredicateOp<(S,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "gather_if source must be backed by one DeviceVec".to_string(),
            }
        })?;
    let indices =
        <Indices as sealed::MIterDispatch<B>>::column_inner::<u32>(&indices).ok_or_else(|| {
            Error::Launch {
                message: "gather_if indices must be backed by one u32 DeviceVec".to_string(),
            }
        })?;
    let stencil =
        <Stencil as sealed::MIterDispatch<B>>::column_inner::<S>(&stencil).ok_or_else(|| {
            Error::Launch {
                message: "gather_if stencil must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::gather_if(source, indices, stencil, pred)?;
    Ok(array_from_inner::<B, (T,), Output>(inner))
}

/// Scatters values into a newly allocated output.
pub fn scatter<B, Input, Indices, T, Output>(
    source: Input,
    indices: Indices,
    len: usize,
    default: (T,),
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    Indices: MIter<B, Item = (u32,)>,
    T: Scalar<B> + Default + 'static,
    Output: MVec<B, Item = (T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "scatter source must be backed by one DeviceVec".to_string(),
            }
        })?;
    let indices =
        <Indices as sealed::MIterDispatch<B>>::column_inner::<u32>(&indices).ok_or_else(|| {
            Error::Launch {
                message: "scatter indices must be backed by one u32 DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::scatter(source, indices, len, default.0)?;
    Ok(array_from_inner::<B, (T,), Output>(inner))
}

/// Scatters selected values into a newly allocated output.
pub fn scatter_if<B, Input, Indices, Stencil, T, S, Output, Pred>(
    source: Input,
    indices: Indices,
    len: usize,
    default: (T,),
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    Indices: MIter<B, Item = (u32,)>,
    Stencil: MIter<B, Item = (S,)>,
    T: Scalar<B> + Default + 'static,
    S: Scalar<B> + 'static,
    Output: MVec<B, Item = (T,)>,
    Pred: op::PredicateOp<(S,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "scatter_if source must be backed by one DeviceVec".to_string(),
            }
        })?;
    let indices =
        <Indices as sealed::MIterDispatch<B>>::column_inner::<u32>(&indices).ok_or_else(|| {
            Error::Launch {
                message: "scatter_if indices must be backed by one u32 DeviceVec".to_string(),
            }
        })?;
    let stencil =
        <Stencil as sealed::MIterDispatch<B>>::column_inner::<S>(&stencil).ok_or_else(|| {
            Error::Launch {
                message: "scatter_if stencil must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::scatter_if(source, indices, len, default.0, stencil, pred)?;
    Ok(array_from_inner::<B, (T,), Output>(inner))
}

/// Finds the first adjacent pair satisfying `pred`.
pub fn adjacent_find<B, Input, T, Pred>(source: Input, pred: Pred) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Pred: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "adjacent_find source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::adjacent_find((source,), pred)
}

/// Returns whether two inputs are equal under `eq`.
pub fn equal<B, Left, Right, T, Eq>(left: Left, right: Right, eq: Eq) -> Result<bool, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Eq: op::BinaryPredicateOp<(T,)>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "equal left input must be backed by one DeviceVec".to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "equal right input must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::equal((left,), (right,), eq)
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<B, Left, Right, T, Eq>(
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Eq: op::BinaryPredicateOp<(T,)>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "mismatch left input must be backed by one DeviceVec".to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "mismatch right input must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::mismatch((left,), (right,), eq)
}

/// Finds the first input element equal to any needle.
pub fn find_first_of<B, Input, Needles, T, Eq>(
    source: Input,
    needles: Needles,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    Needles: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Eq: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "find_first_of source must be backed by one DeviceVec".to_string(),
            }
        })?;
    let needles =
        <Needles as sealed::MIterDispatch<B>>::column_inner::<T>(&needles).ok_or_else(|| {
            Error::Launch {
                message: "find_first_of needles must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::find_first_of((source,), (needles,), eq)
}

/// Finds the minimum element index.
pub fn min_element<B, Input, T, Less>(source: Input, less: Less) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "min_element source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::min_element((source,), less)
}

/// Finds the maximum element index.
pub fn max_element<B, Input, T, Less>(source: Input, less: Less) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "max_element source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::max_element((source,), less)
}

/// Finds both minimum and maximum element indices.
pub fn minmax_element<B, Input, T, Less>(
    source: Input,
    less: Less,
) -> Result<Option<(usize, usize)>, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "minmax_element source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::minmax_element((source,), less)
}

/// Finds the lower bound of `value` in a sorted input.
pub fn lower_bound<B, Input, T, Less>(
    source: Input,
    value: (T,),
    less: Less,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "lower_bound source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::lower_bound((source,), value, less)
}

/// Finds the upper bound of `value` in a sorted input.
pub fn upper_bound<B, Input, T, Less>(
    source: Input,
    value: (T,),
    less: Less,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "upper_bound source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::upper_bound((source,), value, less)
}

/// Finds the equal range of `value` in a sorted input.
pub fn equal_range<B, Input, T, Less>(
    source: Input,
    value: (T,),
    less: Less,
) -> Result<(usize, usize), Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "equal_range source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::equal_range((source,), value, less)
}

/// Returns the first position where sorted order is broken.
pub fn is_sorted_until<B, Input, T, Less>(source: Input, less: Less) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "is_sorted_until source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::is_sorted_until((source,), less)
}

/// Returns whether input is sorted.
pub fn is_sorted<B, Input, T, Less>(source: Input, less: Less) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let source =
        <Input as sealed::MIterDispatch<B>>::column_inner::<T>(&source).ok_or_else(|| {
            Error::Launch {
                message: "is_sorted source must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::is_sorted((source,), less)
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<B, Left, Right, T, Less>(
    left: Left,
    right: Right,
    less: Less,
) -> Result<bool, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "lexicographical_compare left input must be backed by one DeviceVec"
                .to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "lexicographical_compare right input must be backed by one DeviceVec"
                    .to_string(),
            }
        })?;
    crate::detail::lexicographical_compare((left,), (right,), less)
}

/// Merges two sorted inputs.
pub fn merge<B, Left, Right, T, Output, Less>(
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + Default + 'static,
    Output: MVec<B, Item = (T,)>,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "merge left input must be backed by one DeviceVec".to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "merge right input must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::merge((left,), (right,), less)?;
    Ok(array_from_inner::<B, (T,), Output>(inner))
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<B, Left, Right, T, Output, Less>(
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + Default + 'static,
    Output: MVec<B, Item = (T,)>,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "set_union left input must be backed by one DeviceVec".to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "set_union right input must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::set_union((left,), (right,), less)?;
    Ok(array_from_inner::<B, (T,), Output>(inner))
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<B, Left, Right, T, Output, Less>(
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + Default + 'static,
    Output: MVec<B, Item = (T,)>,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "set_intersection left input must be backed by one DeviceVec".to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "set_intersection right input must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::set_intersection((left,), (right,), less)?;
    Ok(array_from_inner::<B, (T,), Output>(inner))
}

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<B, Left, Right, T, Output, Less>(
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + Default + 'static,
    Output: MVec<B, Item = (T,)>,
    Less: op::BinaryPredicateOp<(T,)>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "set_difference left input must be backed by one DeviceVec".to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "set_difference right input must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::set_difference((left,), (right,), less)?;
    Ok(array_from_inner::<B, (T,), Output>(inner))
}

/// Applies a scalar binary transform over two inputs and reduces the result.
pub fn inner_product<B, Left, Right, T, TransformOp, ReduceOp>(
    left: Left,
    right: Right,
    transform_op: TransformOp,
    init: T,
    reduce_op: ReduceOp,
) -> Result<T, Error>
where
    B: Backend,
    Left: MIter<B, Item = (T,)>,
    Right: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
    TransformOp: op::BinaryOp<T>,
    ReduceOp: op::BinaryOp<T>,
{
    let left = <Left as sealed::MIterDispatch<B>>::column_inner::<T>(&left).ok_or_else(|| {
        Error::Launch {
            message: "inner_product left input must be backed by one DeviceVec".to_string(),
        }
    })?;
    let right =
        <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
            Error::Launch {
                message: "inner_product right input must be backed by one DeviceVec".to_string(),
            }
        })?;
    crate::detail::inner_product(left, right, transform_op, init, reduce_op)
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<B, Keys, Values, K, V, KeyEq, Op, Output>(
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B, Item = (V,)>,
    K: Scalar<B> + PartialEq + 'static,
    V: Scalar<B> + 'static,
    KeyEq: op::BinaryPredicateOp<(K,)>,
    Op: op::BinaryOp<(V,)>,
    Output: MVec<B, Item = (V,)>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "inclusive_scan_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    let values =
        <Values as sealed::MIterDispatch<B>>::column_inner::<V>(&values).ok_or_else(|| {
            Error::Launch {
                message: "inclusive_scan_by_key values must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::inclusive_scan_by_key((keys,), (values,), key_eq, op)?;
    Ok(array_from_inner::<B, (V,), Output>(inner))
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<B, Keys, Values, K, V, KeyEq, Op, Output>(
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: (V,),
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B, Item = (V,)>,
    K: Scalar<B> + PartialEq + 'static,
    V: Scalar<B> + 'static,
    KeyEq: op::BinaryPredicateOp<(K,)>,
    Op: op::BinaryOp<(V,)>,
    Output: MVec<B, Item = (V,)>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "exclusive_scan_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    let values =
        <Values as sealed::MIterDispatch<B>>::column_inner::<V>(&values).ok_or_else(|| {
            Error::Launch {
                message: "exclusive_scan_by_key values must be backed by one DeviceVec".to_string(),
            }
        })?;
    let inner = crate::detail::exclusive_scan_by_key((keys,), (values,), key_eq, init, op)?;
    Ok(array_from_inner::<B, (V,), Output>(inner))
}

/// Reduces consecutive values with equal keys.
pub fn reduce_by_key<B, Keys, Values, K, V, KeyEq, Op, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: (V,),
    op: Op,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B, Item = (V,)>,
    K: Scalar<B> + Default + 'static,
    V: Scalar<B> + Default + 'static,
    KeyEq: op::BinaryPredicateOp<(K,)>,
    Op: op::BinaryOp<(V,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = (V,)>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "reduce_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    let values =
        <Values as sealed::MIterDispatch<B>>::column_inner::<V>(&values).ok_or_else(|| {
            Error::Launch {
                message: "reduce_by_key values must be backed by one DeviceVec".to_string(),
            }
        })?;
    let (key_inner, value_inner) =
        crate::detail::reduce_by_key((keys,), (values,), key_eq, init, op)?;
    Ok((
        array_from_inner::<B, (K,), KeyOutput>(key_inner),
        array_from_inner::<B, (V,), ValueOutput>(value_inner),
    ))
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<B, Keys, Values, K, V, Eq, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    eq: Eq,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B, Item = (V,)>,
    K: Scalar<B> + Default + 'static,
    V: Scalar<B> + Default + 'static,
    Eq: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = (V,)>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "unique_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    let values =
        <Values as sealed::MIterDispatch<B>>::column_inner::<V>(&values).ok_or_else(|| {
            Error::Launch {
                message: "unique_by_key values must be backed by one DeviceVec".to_string(),
            }
        })?;
    let (key_inner, value_inner) = crate::detail::unique_by_key((keys,), (values,), eq)?;
    Ok((
        array_from_inner::<B, (K,), KeyOutput>(key_inner),
        array_from_inner::<B, (V,), ValueOutput>(value_inner),
    ))
}

/// Sorts key-value pairs by key.
pub fn sort_by_key<B, Keys, Values, K, V, Less, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B, Item = (V,)>,
    K: Scalar<B> + Default + 'static,
    V: Scalar<B> + Default + 'static,
    Less: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = (V,)>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "sort_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    let values =
        <Values as sealed::MIterDispatch<B>>::column_inner::<V>(&values).ok_or_else(|| {
            Error::Launch {
                message: "sort_by_key values must be backed by one DeviceVec".to_string(),
            }
        })?;
    let (key_inner, value_inner) = crate::detail::sort_by_key((keys,), (values,), less)?;
    Ok((
        array_from_inner::<B, (K,), KeyOutput>(key_inner),
        array_from_inner::<B, (V,), ValueOutput>(value_inner),
    ))
}

/// Stable key-value sort. The current lower implementation is stable.
pub fn stable_sort_by_key<B, Keys, Values, K, V, Less, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B, Item = (V,)>,
    K: Scalar<B> + Default + 'static,
    V: Scalar<B> + Default + 'static,
    Less: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = (V,)>,
{
    sort_by_key(keys, values, less)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<
    B,
    LeftKeys,
    LeftValues,
    RightKeys,
    RightValues,
    K,
    V,
    Less,
    KeyOutput,
    ValueOutput,
>(
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    LeftKeys: MIter<B, Item = (K,)>,
    RightKeys: MIter<B, Item = (K,)>,
    LeftValues: MIter<B, Item = (V,)>,
    RightValues: MIter<B, Item = (V,)>,
    K: Scalar<B> + Default + 'static,
    V: Scalar<B> + Default + 'static,
    Less: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = (V,)>,
{
    let left_keys = <LeftKeys as sealed::MIterDispatch<B>>::column_inner::<K>(&left_keys)
        .ok_or_else(|| Error::Launch {
            message: "merge_by_key left keys must be backed by one DeviceVec".to_string(),
        })?;
    let left_values = <LeftValues as sealed::MIterDispatch<B>>::column_inner::<V>(&left_values)
        .ok_or_else(|| Error::Launch {
            message: "merge_by_key left values must be backed by one DeviceVec".to_string(),
        })?;
    let right_keys = <RightKeys as sealed::MIterDispatch<B>>::column_inner::<K>(&right_keys)
        .ok_or_else(|| Error::Launch {
            message: "merge_by_key right keys must be backed by one DeviceVec".to_string(),
        })?;
    let right_values = <RightValues as sealed::MIterDispatch<B>>::column_inner::<V>(&right_values)
        .ok_or_else(|| Error::Launch {
            message: "merge_by_key right values must be backed by one DeviceVec".to_string(),
        })?;
    let (key_inner, value_inner) = crate::detail::merge_by_key(
        (left_keys,),
        (left_values,),
        (right_keys,),
        (right_values,),
        less,
    )?;
    Ok((
        array_from_inner::<B, (K,), KeyOutput>(key_inner),
        array_from_inner::<B, (V,), ValueOutput>(value_inner),
    ))
}
