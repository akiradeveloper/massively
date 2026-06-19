use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceBinaryMap, DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2,
        SoA3, SoAView1, SoAView2, SoAView3,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    op::{BinaryOp, BinaryPredicateOp, GpuOp},
    primitives::reduce as primitive_reduce,
};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// One-component key input accepted by by-key algorithms.
#[doc(hidden)]
pub trait KeyInput {
    /// CubeCL runtime used by keys.
    type Runtime: Runtime;
    /// Key scalar type.
    type Item;

    /// Materializes keys for primitive kernels.
    fn key_input(self) -> Result<DeviceVec<Self::Runtime, Self::Item>, Error>;
}

impl<Source> KeyInput for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;

    fn key_input(self) -> Result<DeviceVec<Self::Runtime, Self::Item>, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_collect(&self.source)
    }
}

impl<Source> KeyInput for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;

    fn key_input(self) -> Result<DeviceVec<Self::Runtime, Self::Item>, Error> {
        <SoAView1<Source> as KeyInput>::key_input(SoAView1 { source: self })
    }
}

/// Input accepted by [`reduce`].
#[doc(hidden)]
pub trait ReduceInput<Op> {
    /// Initial value type.
    type Init;
    /// Reduction output type.
    type Output;

    /// Reduces this input.
    fn reduce_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error>;
}

impl<Source, Op> ReduceInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Init = (Source::Item,);
    type Output = (Source::Item,);

    fn reduce_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let bindings = self.source.stage()?;
        primitive_reduce::reduce_tuple1_device_expr::<_, _, Source::Expr, Op>(
            self.source.policy(),
            &bindings,
            self.source.len(),
            init,
        )
    }
}

impl<Source, Op> ReduceInput<Op> for (Source,)
where
    SoAView1<Source>: ReduceInput<Op>,
{
    type Init = <SoAView1<Source> as ReduceInput<Op>>::Init;
    type Output = <SoAView1<Source> as ReduceInput<Op>>::Output;

    fn reduce_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView1<Source> as ReduceInput<Op>>::reduce_input(SoAView1 { source: self.0 }, init, op)
    }
}

macro_rules! impl_reduce_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $reduce_fn:ident) => {
        impl<$first, $( $rest ),+, Op> ReduceInput<Op> for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Op: BinaryOp<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
        {
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );

            fn reduce_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_reduce::$reduce_fn::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                    $( <$rest as KernelColumn>::Item, )+
                    <$first as KernelColumn>::Expr,
                    $( <$rest as KernelColumn>::Expr, )+
                    Op,
                >(
                    KernelColumn::policy(&self.$first_field),
                    &$first_field,
                    $( &$field, )+
                    KernelColumn::len(&self.$first_field),
                    init,
                )
            }
        }
    };
}

impl_reduce_input!(SoAView2<A, B> { left, right } => reduce_tuple2_device_expr);
impl_reduce_input!(SoAView3<A, B, C> { first, second, third } => reduce_tuple3_device_expr);

impl<Left, Right, Op> ReduceInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: ReduceInput<Op>,
{
    type Init = <SoAView2<Left, Right> as ReduceInput<Op>>::Init;
    type Output = <SoAView2<Left, Right> as ReduceInput<Op>>::Output;

    fn reduce_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ReduceInput<Op>>::reduce_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            init,
            op,
        )
    }
}

impl<First, Second, Third, Op> ReduceInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ReduceInput<Op>,
{
    type Init = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Init;
    type Output = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Output;

    fn reduce_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ReduceInput<Op>>::reduce_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            init,
            op,
        )
    }
}

macro_rules! impl_reduce_soa_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $reduce_fn:ident) => {
        impl<$first, $( $rest ),+, Op> ReduceInput<Op> for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Op: BinaryOp<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
        {
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );

            fn reduce_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_reduce::$reduce_fn::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                    $( <$rest as KernelColumn>::Item, )+
                    <$first as KernelColumn>::Expr,
                    $( <$rest as KernelColumn>::Expr, )+
                    Op,
                >(
                    KernelColumn::policy(&self.$first_field),
                    &$first_field,
                    $( &$field, )+
                    KernelColumn::len(&self.$first_field),
                    init,
                )
            }
        }
    };
}

impl_reduce_soa_input!(SoA2<A, B> { left, right } => reduce_tuple2_device_expr);
impl_reduce_soa_input!(SoA3<A, B, C> { first, second, third } => reduce_tuple3_device_expr);

/// Reduces read-only device input to a host tuple item.
///
/// This is a borrowing algorithm: pass `&DeviceVec` for one column or [`zip`]
/// for multiple read-only columns. No output device storage is allocated.
///
/// [`zip`]: crate::zip
pub fn reduce<Input, Op>(
    input: Input,
    init: <Input as ReduceInput<Op>>::Init,
    _op: Op,
) -> Result<<Input as ReduceInput<Op>>::Output, Error>
where
    Input: ReduceInput<Op>,
{
    input.reduce_input(init, GpuOp::<Op>::new())
}

/// Input accepted by [`inner_product`].
#[doc(hidden)]
pub trait InnerProductInput<Right, TransformOp, ReduceOp> {
    /// Reduced item type.
    type Item;

    /// Applies a binary transform and reduces the result.
    fn inner_product_input(
        self,
        right: Right,
        init: Self::Item,
        transform_op: GpuOp<TransformOp>,
        reduce_op: GpuOp<ReduceOp>,
    ) -> Result<Self::Item, Error>;
}

impl<Left, Right, TransformOp, ReduceOp> InnerProductInput<Right, TransformOp, ReduceOp> for Left
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    TransformOp: BinaryOp<Left::Item>,
    ReduceOp: BinaryOp<Left::Item>,
    DeviceBinaryMap<Left, Right, TransformOp>:
        KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    <DeviceBinaryMap<Left, Right, TransformOp> as KernelColumn>::Expr: DeviceGpuExpr<Left::Item>,
{
    type Item = Left::Item;

    fn inner_product_input(
        self,
        right: Right,
        init: Self::Item,
        _transform_op: GpuOp<TransformOp>,
        reduce_op: GpuOp<ReduceOp>,
    ) -> Result<Self::Item, Error> {
        let mapped = DeviceBinaryMap {
            left: self,
            right,
            _op: PhantomData::<fn() -> TransformOp>,
        };
        let _ = reduce_op;
        super::device_expr_reduce::<_, ReduceOp>(&mapped, init)
    }
}

macro_rules! impl_inner_product_tuple_input {
    (
        $left_name:ident < $first_left:ident, $( $left:ident ),+ >,
        $right_name:ident < $first_right:ident, $( $right:ident ),+ > {
            $first_field:ident, $( $field:ident ),+
        }
    ) => {
        impl<$first_left, $first_right, $( $left, $right ),+, TransformOp, ReduceOp>
            InnerProductInput<$right_name<$first_right, $( $right ),+>, TransformOp, ReduceOp>
            for $left_name<$first_left, $( $left ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first_left as KernelColumn>::Item>,
            $right_name<$first_right, $( $right ),+>: ReadOnlySoA<Scalar = <$first_right as KernelColumn>::Item>,
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $first_right:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<<$first_left as KernelColumnAt<S0>>::Next>,
            $(
                $left: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>,
                $right:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<<$left as KernelColumnAt<S0>>::Next>,
            )+
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$left as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            TransformOp: BinaryOp<<$first_left as KernelColumn>::Item>,
            $( TransformOp: BinaryOp<<$left as KernelColumn>::Item>, )+
            ReduceOp: BinaryOp<<$first_left as KernelColumn>::Item>,
            $( ReduceOp: BinaryOp<<$left as KernelColumn>::Item>, )+
            DeviceBinaryMap<$first_left, $first_right, TransformOp>:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<S0>,
            <DeviceBinaryMap<$first_left, $first_right, TransformOp> as KernelColumn>::Expr:
                DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            $(
                DeviceBinaryMap<$left, $right, TransformOp>:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
                <DeviceBinaryMap<$left, $right, TransformOp> as KernelColumn>::Expr:
                    DeviceGpuExpr<<$left as KernelColumn>::Item>,
            )+
        {
            type Item = (
                <$first_left as KernelColumn>::Item,
                $( <$left as KernelColumn>::Item ),+
            );

            fn inner_product_input(
                self,
                right: $right_name<$first_right, $( $right ),+>,
                init: Self::Item,
                _transform_op: GpuOp<TransformOp>,
                _reduce_op: GpuOp<ReduceOp>,
            ) -> Result<Self::Item, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&right)?;
                let ($first_field, $( $field ),+) = init;
                let first_mapped = DeviceBinaryMap {
                    left: self.$first_field,
                    right: right.$first_field,
                    _op: PhantomData::<fn() -> TransformOp>,
                };
                let $first_field = super::device_expr_reduce::<_, ReduceOp>(&first_mapped, $first_field)?;
                $(
                    let mapped = DeviceBinaryMap {
                        left: self.$field,
                        right: right.$field,
                        _op: PhantomData::<fn() -> TransformOp>,
                    };
                    let $field = super::device_expr_reduce::<_, ReduceOp>(&mapped, $field)?;
                )+
                Ok(($first_field, $( $field ),+))
            }
        }
    };
}

impl_inner_product_tuple_input!(SoAView2<A, B>, SoAView2<RA, RB> { left, right });
impl_inner_product_tuple_input!(SoAView3<A, B, C>, SoAView3<RA, RB, RC> { first, second, third });

macro_rules! impl_inner_product_owned_tuple_input {
    (
        $left_name:ident < $first_left:ident, $( $left:ident ),+ >,
        $right_name:ident < $first_right:ident, $( $right:ident ),+ > {
            $first_field:ident, $( $field:ident ),+
        }
    ) => {
        impl<$first_left, $first_right, $( $left, $right ),+, TransformOp, ReduceOp>
            InnerProductInput<$right_name<$first_right, $( $right ),+>, TransformOp, ReduceOp>
            for $left_name<$first_left, $( $left ),+>
        where
            Self: SoA<Scalar = <$first_left as KernelColumn>::Item>,
            $right_name<$first_right, $( $right ),+>: SoA<Scalar = <$first_right as KernelColumn>::Item>,
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $first_right:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<<$first_left as KernelColumnAt<S0>>::Next>,
            $(
                $left: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>,
                $right:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<<$left as KernelColumnAt<S0>>::Next>,
            )+
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$left as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            TransformOp: BinaryOp<<$first_left as KernelColumn>::Item>,
            $( TransformOp: BinaryOp<<$left as KernelColumn>::Item>, )+
            ReduceOp: BinaryOp<<$first_left as KernelColumn>::Item>,
            $( ReduceOp: BinaryOp<<$left as KernelColumn>::Item>, )+
            DeviceBinaryMap<$first_left, $first_right, TransformOp>:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<S0>,
            <DeviceBinaryMap<$first_left, $first_right, TransformOp> as KernelColumn>::Expr:
                DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            $(
                DeviceBinaryMap<$left, $right, TransformOp>:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
                <DeviceBinaryMap<$left, $right, TransformOp> as KernelColumn>::Expr:
                    DeviceGpuExpr<<$left as KernelColumn>::Item>,
            )+
        {
            type Item = (
                <$first_left as KernelColumn>::Item,
                $( <$left as KernelColumn>::Item ),+
            );

            fn inner_product_input(
                self,
                right: $right_name<$first_right, $( $right ),+>,
                init: Self::Item,
                _transform_op: GpuOp<TransformOp>,
                _reduce_op: GpuOp<ReduceOp>,
            ) -> Result<Self::Item, Error> {
                SoA::validate(&self)?;
                SoA::validate(&right)?;
                super::ensure_same_len(SoA::len(&right), SoA::len(&self))?;
                let ($first_field, $( $field ),+) = init;
                let first_mapped = DeviceBinaryMap {
                    left: self.$first_field,
                    right: right.$first_field,
                    _op: PhantomData::<fn() -> TransformOp>,
                };
                let $first_field = super::device_expr_reduce::<_, ReduceOp>(&first_mapped, $first_field)?;
                $(
                    let mapped = DeviceBinaryMap {
                        left: self.$field,
                        right: right.$field,
                        _op: PhantomData::<fn() -> TransformOp>,
                    };
                    let $field = super::device_expr_reduce::<_, ReduceOp>(&mapped, $field)?;
                )+
                Ok(($first_field, $( $field ),+))
            }
        }
    };
}

impl_inner_product_owned_tuple_input!(SoA2<A, B>, SoA2<RA, RB> { left, right });
impl_inner_product_owned_tuple_input!(SoA3<A, B, C>, SoA3<RA, RB, RC> { first, second, third });

macro_rules! impl_inner_product_mixed_tuple_input {
    (
        $left_trait:ident,
        $right_trait:ident,
        $left_name:ident < $first_left:ident, $( $left:ident ),+ >,
        $right_name:ident < $first_right:ident, $( $right:ident ),+ > {
            $first_field:ident, $( $field:ident ),+
        }
    ) => {
        impl<$first_left, $first_right, $( $left, $right ),+, TransformOp, ReduceOp>
            InnerProductInput<$right_name<$first_right, $( $right ),+>, TransformOp, ReduceOp>
            for $left_name<$first_left, $( $left ),+>
        where
            Self: $left_trait<Scalar = <$first_left as KernelColumn>::Item>,
            $right_name<$first_right, $( $right ),+>: $right_trait<Scalar = <$first_right as KernelColumn>::Item>,
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $first_right:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<<$first_left as KernelColumnAt<S0>>::Next>,
            $(
                $left: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>,
                $right:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<<$left as KernelColumnAt<S0>>::Next>,
            )+
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$left as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            TransformOp: BinaryOp<<$first_left as KernelColumn>::Item>,
            $( TransformOp: BinaryOp<<$left as KernelColumn>::Item>, )+
            ReduceOp: BinaryOp<<$first_left as KernelColumn>::Item>,
            $( ReduceOp: BinaryOp<<$left as KernelColumn>::Item>, )+
            DeviceBinaryMap<$first_left, $first_right, TransformOp>:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<S0>,
            <DeviceBinaryMap<$first_left, $first_right, TransformOp> as KernelColumn>::Expr:
                DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            $(
                DeviceBinaryMap<$left, $right, TransformOp>:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
                <DeviceBinaryMap<$left, $right, TransformOp> as KernelColumn>::Expr:
                    DeviceGpuExpr<<$left as KernelColumn>::Item>,
            )+
        {
            type Item = (
                <$first_left as KernelColumn>::Item,
                $( <$left as KernelColumn>::Item ),+
            );

            fn inner_product_input(
                self,
                right: $right_name<$first_right, $( $right ),+>,
                init: Self::Item,
                _transform_op: GpuOp<TransformOp>,
                _reduce_op: GpuOp<ReduceOp>,
            ) -> Result<Self::Item, Error> {
                $left_trait::validate(&self)?;
                $right_trait::validate(&right)?;
                super::ensure_same_len($right_trait::len(&right), $left_trait::len(&self))?;
                let ($first_field, $( $field ),+) = init;
                let first_mapped = DeviceBinaryMap {
                    left: self.$first_field,
                    right: right.$first_field,
                    _op: PhantomData::<fn() -> TransformOp>,
                };
                let $first_field = super::device_expr_reduce::<_, ReduceOp>(&first_mapped, $first_field)?;
                $(
                    let mapped = DeviceBinaryMap {
                        left: self.$field,
                        right: right.$field,
                        _op: PhantomData::<fn() -> TransformOp>,
                    };
                    let $field = super::device_expr_reduce::<_, ReduceOp>(&mapped, $field)?;
                )+
                Ok(($first_field, $( $field ),+))
            }
        }
    };
}

macro_rules! impl_inner_product_mixed_tuple_inputs {
    (
        $soa:ident < $first_left:ident, $( $left:ident ),+ >,
        $soa_view:ident < $first_right:ident, $( $right:ident ),+ > {
            $first_field:ident, $( $field:ident ),+
        }
    ) => {
        impl_inner_product_mixed_tuple_input!(
            SoA,
            ReadOnlySoA,
            $soa < $first_left, $( $left ),+ >,
            $soa_view < $first_right, $( $right ),+ > {
                $first_field, $( $field ),+
            }
        );
        impl_inner_product_mixed_tuple_input!(
            ReadOnlySoA,
            SoA,
            $soa_view < $first_left, $( $left ),+ >,
            $soa < $first_right, $( $right ),+ > {
                $first_field, $( $field ),+
            }
        );
    };
}

impl_inner_product_mixed_tuple_inputs!(SoA2<A, B>, SoAView2<RA, RB> { left, right });
impl_inner_product_mixed_tuple_inputs!(SoA3<A, B, C>, SoAView3<RA, RB, RC> { first, second, third });

/// Applies a binary transform over two read-only inputs and reduces the result.
///
/// This is a fused borrowing algorithm. It reads both inputs and returns a host
/// scalar.
pub fn inner_product<Left, Right, TransformOp, ReduceOp>(
    left: Left,
    right: Right,
    _transform_op: TransformOp,
    init: <Left as InnerProductInput<Right, TransformOp, ReduceOp>>::Item,
    _reduce_op: ReduceOp,
) -> Result<<Left as InnerProductInput<Right, TransformOp, ReduceOp>>::Item, Error>
where
    Left: InnerProductInput<Right, TransformOp, ReduceOp>,
{
    left.inner_product_input(
        right,
        init,
        GpuOp::<TransformOp>::new(),
        GpuOp::<ReduceOp>::new(),
    )
}

/// Input accepted by [`reduce_by_key`].
#[doc(hidden)]
pub trait ReduceByKeyInput<K, KeyEq, Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Initial value type.
    type Init;
    /// Reduced values output type.
    type Values;

    /// Reduces contiguous equal-key runs.
    fn reduce_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error>;
}

impl<Source, K, KeyEq, Op> ReduceByKeyInput<K, KeyEq, Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Init = Source::Item;
    type Values = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reduce_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        ReadOnlySoA::validate(&self)?;
        let (keys, source) =
            super::device_expr_reduce_by_key::<Source, K, KeyEq, Op>(&self.source, keys, init)?;
        Ok((keys, SoA1 { source }))
    }
}

impl<Source, K, KeyEq, Op> ReduceByKeyInput<K, KeyEq, Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    SoAView1<Source>: ReduceByKeyInput<K, KeyEq, Op>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
{
    type Runtime = <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::Runtime;
    type Values = <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::Values;
    type Init = <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::Init;

    fn reduce_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::reduce_by_key_input(
            SoAView1 { source: self },
            keys,
            init,
            op,
        )
    }
}

impl<Left, Right, K, KeyEq, Op> ReduceByKeyInput<K, KeyEq, Op> for SoAView2<Left, Right>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: GpuExpr<Left::Item>,
    Right::Expr: GpuExpr<Right::Item>,
    Op: BinaryOp<Left::Item>,
    Op: BinaryOp<Right::Item>,
{
    type Runtime = Left::Runtime;
    type Init = (Left::Item, Right::Item);
    type Values = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn reduce_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        self.left.validate()?;
        self.right.validate()?;
        super::ensure_same_len(self.right.len(), self.left.len())?;
        let (out_keys, left, control) =
            super::device_expr_reduce_by_key_with_control::<Left, K, KeyEq, Op>(
                &self.left, keys, init.0,
            )?;
        let right = super::device_expr_reduce_by_key_with_existing_control::<Right, K, KeyEq, Op>(
            &self.right,
            keys,
            init.1,
            &control,
        )?;
        Ok((out_keys, SoA2 { left, right }))
    }
}

impl<First, Second, Third, K, KeyEq, Op> ReduceByKeyInput<K, KeyEq, Op>
    for SoAView3<First, Second, Third>
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: GpuExpr<First::Item>,
    Second::Expr: GpuExpr<Second::Item>,
    Third::Expr: GpuExpr<Third::Item>,
    Op: BinaryOp<First::Item>,
    Op: BinaryOp<Second::Item>,
    Op: BinaryOp<Third::Item>,
{
    type Runtime = First::Runtime;
    type Init = (First::Item, Second::Item, Third::Item);
    type Values = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn reduce_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        self.first.validate()?;
        self.second.validate()?;
        self.third.validate()?;
        super::ensure_same_len(self.second.len(), self.first.len())?;
        super::ensure_same_len(self.third.len(), self.first.len())?;
        let (out_keys, first, control) = super::device_expr_reduce_by_key_with_control::<
            First,
            K,
            KeyEq,
            Op,
        >(&self.first, keys, init.0)?;
        let second = super::device_expr_reduce_by_key_with_existing_control::<Second, K, KeyEq, Op>(
            &self.second,
            keys,
            init.1,
            &control,
        )?;
        let third = super::device_expr_reduce_by_key_with_existing_control::<Third, K, KeyEq, Op>(
            &self.third,
            keys,
            init.2,
            &control,
        )?;
        Ok((
            out_keys,
            SoA3 {
                first,
                second,
                third,
            },
        ))
    }
}

macro_rules! impl_reduce_by_key_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Key, KeyEq, Op> ReduceByKeyInput<Key, KeyEq, Op> for $input<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            Key: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<Key>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: GpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: GpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Op: BinaryOp<<$first as KernelColumn>::Item>,
            $(
                Op: BinaryOp<<$rest as KernelColumn>::Item>,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Values = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn reduce_by_key_input(
                self,
                keys: &DeviceVec<Self::Runtime, Key>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<(DeviceVec<Self::Runtime, Key>, Self::Values), Error> {
                SoA::validate(&self)?;
                let ($first_field, $( $field ),+) = init;
                let (out_keys, $first_field, control) = super::device_expr_reduce_by_key_with_control::<$first, Key, KeyEq, Op>(
                    &self.$first_field,
                    keys,
                    $first_field,
                )?;
                $(
                    let $field = super::device_expr_reduce_by_key_with_existing_control::<$rest, Key, KeyEq, Op>(
                        &self.$field,
                        keys,
                        $field,
                        &control,
                    )?;
                )+
                Ok((out_keys, $output { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_reduce_by_key_soa_input!(SoA2 -> SoA2<A, B> { left, right });
impl_reduce_by_key_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third });

macro_rules! impl_reduce_by_key_tuple_values {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Key, KeyEq, Op> ReduceByKeyInput<Key, KeyEq, Op> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: ReduceByKeyInput<Key, KeyEq, Op>,
        {
            type Runtime = <$view<$( $ty ),+> as ReduceByKeyInput<Key, KeyEq, Op>>::Runtime;
            type Init = <$view<$( $ty ),+> as ReduceByKeyInput<Key, KeyEq, Op>>::Init;
            type Values = <$view<$( $ty ),+> as ReduceByKeyInput<Key, KeyEq, Op>>::Values;

            fn reduce_by_key_input(
                self,
                keys: &DeviceVec<Self::Runtime, Key>,
                init: Self::Init,
                op: GpuOp<Op>,
            ) -> Result<(DeviceVec<Self::Runtime, Key>, Self::Values), Error> {
                <$view<$( $ty ),+> as ReduceByKeyInput<Key, KeyEq, Op>>::reduce_by_key_input(
                    $view { $( $field: self.$index ),+ },
                    keys,
                    init,
                    op,
                )
            }
        }
    };
}

impl_reduce_by_key_tuple_values!(SoAView2<A, B> { left: 0, right: 1 });
impl_reduce_by_key_tuple_values!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

#[doc(hidden)]
pub trait ReduceByKeyCall<Values, KeyEq, Op> {
    type Init;
    type Output;

    fn reduce_by_key_call(
        self,
        values: Values,
        key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Values, Keys, KeyEq, Op> ReduceByKeyCall<Values, KeyEq, Op> for Keys
where
    Keys: KeyInput,
    Keys::Item: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<Keys::Item>,
    Values: ReduceByKeyInput<Keys::Item, KeyEq, Op, Runtime = Keys::Runtime>,
{
    type Init = <Values as ReduceByKeyInput<Keys::Item, KeyEq, Op>>::Init;
    type Output = (
        SoA1<DeviceVec<Keys::Runtime, Keys::Item>>,
        <Values as ReduceByKeyInput<Keys::Item, KeyEq, Op>>::Values,
    );

    fn reduce_by_key_call(
        self,
        values: Values,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.key_input()?;
        let (keys, values) = values.reduce_by_key_input(&keys, init, GpuOp::<Op>::new())?;
        Ok((SoA1 { source: keys }, values))
    }
}

impl<ValueSource, KeyA, KeyB, KeyEq, Op> ReduceByKeyCall<ValueSource, KeyEq, Op>
    for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    KeyA::Item: PartialEq,
    KeyB::Item: PartialEq,
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueSource::Item>,
{
    type Init = ValueSource::Item;
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn reduce_by_key_call(
        self,
        values: ValueSource,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let values = super::device_expr_collect(&values.source)?;
        let (left, right, source) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &values, init,
            )?;
        Ok((SoA2 { left, right }, SoA1 { source }))
    }
}

impl<ValueA, ValueB, KeyA, KeyB, KeyEq, Op> ReduceByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op>
    for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    SoAView2<ValueA, ValueB>:
        ReadOnlySoA<Item = (ValueA::Item, ValueB::Item), Scalar = ValueA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    KeyA::Item: PartialEq,
    KeyB::Item: PartialEq,
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
{
    type Init = (ValueA::Item, ValueB::Item);
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn reduce_by_key_call(
        self,
        values: SoAView2<ValueA, ValueB>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let (left, right, value_a) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &value_a, init.0,
            )?;
        let (_, _, value_b) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &value_b, init.1,
            )?;
        Ok((
            SoA2 { left, right },
            SoA2 {
                left: value_a,
                right: value_b,
            },
        ))
    }
}

impl<ValueA, ValueB, ValueC, KeyA, KeyB, KeyEq, Op>
    ReduceByKeyCall<SoAView3<ValueA, ValueB, ValueC>, KeyEq, Op> for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    SoAView3<ValueA, ValueB, ValueC>:
        ReadOnlySoA<Item = (ValueA::Item, ValueB::Item, ValueC::Item), Scalar = ValueA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement + PartialEq,
    KeyB::Item: CubePrimitive + CubeElement + PartialEq,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
    Op: BinaryOp<ValueC::Item>,
{
    type Init = (ValueA::Item, ValueB::Item, ValueC::Item);
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA3<
            DeviceVec<KeyA::Runtime, ValueA::Item>,
            DeviceVec<KeyA::Runtime, ValueB::Item>,
            DeviceVec<KeyA::Runtime, ValueC::Item>,
        >,
    );

    fn reduce_by_key_call(
        self,
        values: SoAView3<ValueA, ValueB, ValueC>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        let (left, right, value_a) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &value_a, init.0,
            )?;
        let (_, _, value_b) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &value_b, init.1,
            )?;
        let (_, _, value_c) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &value_c, init.2,
            )?;
        Ok((
            SoA2 { left, right },
            SoA3 {
                first: value_a,
                second: value_b,
                third: value_c,
            },
        ))
    }
}

impl<ValueSource, KeyA, KeyB, KeyC, KeyEq, Op> ReduceByKeyCall<ValueSource, KeyEq, Op>
    for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement + PartialEq,
    KeyB::Item: CubePrimitive + CubeElement + PartialEq,
    KeyC::Item: CubePrimitive + CubeElement + PartialEq,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueSource::Item>,
{
    type Init = ValueSource::Item;
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn reduce_by_key_call(
        self,
        values: ValueSource,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let values = super::device_expr_collect(&values.source)?;
        let (first, second, third, source) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &key_c, &values, init,
            )?;
        Ok((
            SoA3 {
                first,
                second,
                third,
            },
            SoA1 { source },
        ))
    }
}

impl<ValueA, ValueB, KeyA, KeyB, KeyC, KeyEq, Op>
    ReduceByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op> for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    SoAView2<ValueA, ValueB>:
        ReadOnlySoA<Item = (ValueA::Item, ValueB::Item), Scalar = ValueA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement + PartialEq,
    KeyB::Item: CubePrimitive + CubeElement + PartialEq,
    KeyC::Item: CubePrimitive + CubeElement + PartialEq,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
{
    type Init = (ValueA::Item, ValueB::Item);
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn reduce_by_key_call(
        self,
        values: SoAView2<ValueA, ValueB>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let (first, second, third, value_a) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &key_c, &value_a, init.0,
            )?;
        let (_, _, _, value_b) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &key_c, &value_b, init.1,
            )?;
        Ok((
            SoA3 {
                first,
                second,
                third,
            },
            SoA2 {
                left: value_a,
                right: value_b,
            },
        ))
    }
}

impl<ValueA, ValueB, ValueC, KeyA, KeyB, KeyC, KeyEq, Op>
    ReduceByKeyCall<SoAView3<ValueA, ValueB, ValueC>, KeyEq, Op> for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    SoAView3<ValueA, ValueB, ValueC>:
        ReadOnlySoA<Item = (ValueA::Item, ValueB::Item, ValueC::Item), Scalar = ValueA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement + PartialEq,
    KeyB::Item: CubePrimitive + CubeElement + PartialEq,
    KeyC::Item: CubePrimitive + CubeElement + PartialEq,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
    Op: BinaryOp<ValueC::Item>,
{
    type Init = (ValueA::Item, ValueB::Item, ValueC::Item);
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA3<
            DeviceVec<KeyA::Runtime, ValueA::Item>,
            DeviceVec<KeyA::Runtime, ValueB::Item>,
            DeviceVec<KeyA::Runtime, ValueC::Item>,
        >,
    );

    fn reduce_by_key_call(
        self,
        values: SoAView3<ValueA, ValueB, ValueC>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        let (first, second, third, value_a) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &key_c, &value_a, init.0,
            )?;
        let (_, _, _, value_b) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &key_c, &value_b, init.1,
            )?;
        let (_, _, _, value_c) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                &key_a, &key_b, &key_c, &value_c, init.2,
            )?;
        Ok((
            SoA3 {
                first,
                second,
                third,
            },
            SoA3 {
                first: value_a,
                second: value_b,
                third: value_c,
            },
        ))
    }
}

macro_rules! impl_reduce_by_tuple_key_scalar_value {
    (
        $keys:ident -> $out_keys:ident,
        $reduce_fn:ident,
        $eq:ident,
        ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )
    ) => {
        impl<ValueSource, $first, $( $key ),+, KeyEq, Op> ReduceByKeyCall<ValueSource, KeyEq, Op>
            for $keys<$first, $( $key ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement + PartialEq,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement + PartialEq, )+
            ValueSource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<ValueSource::Item>,
        {
            type Init = ValueSource::Item;
            type Output = (
                $out_keys<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                SoA1<DeviceVec<<$first as KernelColumn>::Runtime, ValueSource::Item>>,
            );

            fn reduce_by_key_call(
                self,
                values: ValueSource,
                _key_eq: GpuOp<KeyEq>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                values.validate()?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let values = super::device_expr_collect(&values)?;
                let ($first_out, $( $out_field, )+ source) =
                    primitive_reduce::$reduce_fn::<
                        <$first as KernelColumn>::Runtime,
                        <$first as KernelColumn>::Item,
                        $( <$key as KernelColumn>::Item, )+
                        ValueSource::Item,
                        KeyEq,
                        Op,
                    >(
                        &$first_field,
                        $( &$field, )+
                        &values,
                        init,
                    )?;
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out_field ),+ },
                    SoA1 { source },
                ))
            }
        }
    };
}

impl_reduce_by_tuple_key_scalar_value!(SoA2 -> SoA2, reduce_tuple2_by_key_device_vec, Tuple2Equal, (A: left: out_left, B: right: out_right));
impl_reduce_by_tuple_key_scalar_value!(SoA3 -> SoA3, reduce_tuple3_by_key_device_vec, Tuple3Equal, (A: first: out_first, B: second: out_second, C: third: out_third));

macro_rules! impl_reduce_by_tuple_key_soa_view_values {
    (@reduce $reduce_fn:ident, $eq:ident, ($first_field:ident, $( $field:ident ),+), $value_ty:ident, $value_field:ident, $init:expr, ($first:ident, $( $key:ident ),+)) => {
        primitive_reduce::$reduce_fn::<
            <$first as KernelColumn>::Runtime,
            <$first as KernelColumn>::Item,
            $( <$key as KernelColumn>::Item, )+
            <$value_ty as KernelColumn>::Item,
            $eq,
            Op,
        >(
            &$first_field,
            $( &$field, )+
            &$value_field,
            $init,
        )
    };
    (@reduce_values $reduce_fn:ident, $eq:ident, $value_index:tt, ($first_field:ident, $( $field:ident ),+), ($first:ident, $( $key:ident ),+), $init:ident, ) => {};
    (@reduce_values $reduce_fn:ident, $eq:ident, $value_index:tt, ($first_field:ident, $( $field:ident ),+), ($first:ident, $( $key:ident ),+), $init:ident, $value_ty:ident: $value_field:ident: $idx:tt $(, $tail_ty:ident: $tail_field:ident: $tail_idx:tt )*) => {
        let $value_field = impl_reduce_by_tuple_key_soa_view_values!(
            @reduce $reduce_fn,
            $eq,
            ($first_field, $( $field ),+),
            $value_ty,
            $value_field,
            $init.$idx,
            ($first, $( $key ),+)
        )?.$value_index;
        impl_reduce_by_tuple_key_soa_view_values!(
            @reduce_values $reduce_fn,
            $eq,
            $value_index,
            ($first_field, $( $field ),+),
            ($first, $( $key ),+),
            $init,
            $( $tail_ty: $tail_field: $tail_idx ),*
        );
    };

    (
        $key_storage:ident,
        $storage:ident,
        $values:ident -> $output:ident < $first_value:ident: $first_idx:tt, $( $value:ident: $idx:tt ),+ > { $first_value_field:ident, $( $value_field:ident ),+ },
        $keys:ident -> $out_keys:ident,
        $reduce_fn:ident,
        $eq:ident,
        $value_index:tt,
        ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )
    ) => {
        impl<$first_value, $( $value ),+, $first, $( $key ),+, KeyEq, Op>
            ReduceByKeyCall<$values<$first_value, $( $value ),+>, KeyEq, Op> for $keys<$first, $( $key ),+>
        where
            Self: $key_storage<Scalar = <$first as KernelColumn>::Item>,
            $values<$first_value, $( $value ),+>: $storage<Scalar = <$first_value as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $first_value: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            $( $value: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement + PartialEq,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement + PartialEq, )+
            <$first_value as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            <$first_value as KernelColumn>::Expr: DeviceGpuExpr<<$first_value as KernelColumn>::Item>,
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<<$first_value as KernelColumn>::Item>,
            $( Op: BinaryOp<<$value as KernelColumn>::Item>, )+
        {
            type Init = (<$first_value as KernelColumn>::Item, $( <$value as KernelColumn>::Item ),+);
            type Output = (
                $out_keys<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                $output<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first_value as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+
                >,
            );

            fn reduce_by_key_call(
                self,
                values: $values<$first_value, $( $value ),+>,
                _key_eq: GpuOp<KeyEq>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                $key_storage::validate(&self)?;
                $storage::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let $first_value_field = super::device_expr_collect(&values.$first_value_field)?;
                $( let $value_field = super::device_expr_collect(&values.$value_field)?; )+
                let ($first_out, $( $out_field, )+ $first_value_field) = impl_reduce_by_tuple_key_soa_view_values!(
                    @reduce $reduce_fn,
                    KeyEq,
                    ($first_field, $( $field ),+),
                    $first_value,
                    $first_value_field,
                    init.$first_idx,
                    ($first, $( $key ),+)
                )?;
                impl_reduce_by_tuple_key_soa_view_values!(
                    @reduce_values $reduce_fn,
                    KeyEq,
                    $value_index,
                    ($first_field, $( $field ),+),
                    ($first, $( $key ),+),
                    init,
                    $( $value: $value_field: $idx ),+
                );
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out_field ),+ },
                    $output { $first_value_field, $( $value_field ),+ },
                ))
            }
        }
    };
}

macro_rules! impl_reduce_by_tuple_key_soa_view_values_for_key {
    ($key_storage:ident, $keys:ident -> $out_keys:ident, $reduce_fn:ident, $eq:ident, $value_index:tt, ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )) => {
        impl_reduce_by_tuple_key_soa_view_values!($key_storage, SoA, SoA2 -> SoA2<A: 0, B: 1> { left, right }, $keys -> $out_keys, $reduce_fn, $eq, $value_index, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_reduce_by_tuple_key_soa_view_values!($key_storage, SoA, SoA3 -> SoA3<A: 0, B: 1, C: 2> { first, second, third }, $keys -> $out_keys, $reduce_fn, $eq, $value_index, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
    };
}

impl_reduce_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView2 -> SoA2, reduce_tuple2_by_key_device_vec, Tuple2Equal, 2, (KA: left: out_left, KB: right: out_right));
impl_reduce_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView3 -> SoA3, reduce_tuple3_by_key_device_vec, Tuple3Equal, 3, (KA: first: out_first, KB: second: out_second, KC: third: out_third));
impl_reduce_by_tuple_key_soa_view_values_for_key!(SoA, SoA2 -> SoA2, reduce_tuple2_by_key_device_vec, Tuple2Equal, 2, (KA: left: out_left, KB: right: out_right));
impl_reduce_by_tuple_key_soa_view_values_for_key!(SoA, SoA3 -> SoA3, reduce_tuple3_by_key_device_vec, Tuple3Equal, 3, (KA: first: out_first, KB: second: out_second, KC: third: out_third));

macro_rules! impl_reduce_by_key_tuple_keys {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Values, KeyEq, Op> ReduceByKeyCall<Values, KeyEq, Op> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: ReduceByKeyCall<Values, KeyEq, Op>,
        {
            type Init = <$view<$( $ty ),+> as ReduceByKeyCall<Values, KeyEq, Op>>::Init;
            type Output = <$view<$( $ty ),+> as ReduceByKeyCall<Values, KeyEq, Op>>::Output;

            fn reduce_by_key_call(
                self,
                values: Values,
                key_eq: GpuOp<KeyEq>,
                init: Self::Init,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as ReduceByKeyCall<Values, KeyEq, Op>>::reduce_by_key_call(
                    $view { $( $field: self.$index ),+ },
                    values,
                    key_eq,
                    init,
                    op,
                )
            }
        }
    };
}

impl_reduce_by_key_tuple_keys!(SoAView2<A, B> { left: 0, right: 1 });
impl_reduce_by_key_tuple_keys!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

impl<KeySource, ValueSource, KeyEq, Op> ReduceByKeyCall<(ValueSource,), KeyEq, Op> for (KeySource,)
where
    KeySource: KeyInput,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource: ReduceByKeyInput<
            KeySource::Item,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
            Runtime = KeySource::Runtime,
        >,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
{
    type Init = (
        <ValueSource as ReduceByKeyInput<
            KeySource::Item,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
        >>::Init,
    );
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        <ValueSource as ReduceByKeyInput<
            KeySource::Item,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
        >>::Values,
    );

    fn reduce_by_key_call(
        self,
        values: (ValueSource,),
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.0.key_input()?;
        let (keys, values) = values.0.reduce_by_key_input(
            &keys,
            init.0,
            GpuOp::<super::Tuple1BinaryOp<Op>>::new(),
        )?;
        Ok((SoA1 { source: keys }, values))
    }
}

/// Reduces contiguous equal-key runs using read-only keys and values.
///
/// This is a borrowing algorithm: values may be a borrowed column or a read-only
/// SoA from [`zip`](crate::zip). The returned keys and values are owned SoA
/// storage.
pub fn reduce_by_key<Keys, Values, KeyEq, Op>(
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    init: <Keys as ReduceByKeyCall<Values, KeyEq, Op>>::Init,
    _op: Op,
) -> Result<
    <<Keys as ReduceByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Keys: ReduceByKeyCall<Values, KeyEq, Op>,
    <Keys as ReduceByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput,
{
    materialize(keys.reduce_by_key_call(values, GpuOp::<KeyEq>::new(), init, GpuOp::<Op>::new())?)
}
