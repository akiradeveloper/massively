use crate::{
    device::{
        DeviceBinaryMap, DeviceVec, KernelColumn, KernelColumnAt, S0, SoA1, SoA2, SoA3, SoA4, SoA5,
        SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoVA, SoVA1, SoVA2, SoVA3, SoVA4, SoVA5,
        SoVA6, SoVA7, SoVA8, SoVA9, SoVA10, SoVA11, SoVA12,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    op::{BinaryOp, GpuOp},
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

impl<Source> KeyInput for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;

    fn key_input(self) -> Result<DeviceVec<Self::Runtime, Self::Item>, Error> {
        SoVA::validate(&self)?;
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
        <SoVA1<Source> as KeyInput>::key_input(SoVA1 { source: self })
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

impl<Source, Op> ReduceInput<Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Init = Source::Item;
    type Output = Source::Item;

    fn reduce_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        super::device_expr_reduce::<Source, Op>(&self.source, init)
    }
}

impl<Source, Op> ReduceInput<Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Init = Source::Item;
    type Output = Source::Item;

    fn reduce_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoVA1<Source> as ReduceInput<Op>>::reduce_input(SoVA1 { source: self }, init, op)
    }
}

macro_rules! impl_reduce_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> ReduceInput<Op> for $name<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
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
            Op: BinaryOp<<$first as KernelColumn>::Item>,
            $(
                Op: BinaryOp<<$rest as KernelColumn>::Item>,
            )+
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
                SoVA::validate(&self)?;
                let ($first_field, $( $field ),+) = init;
                let $first_field =
                    super::device_expr_reduce::<$first, Op>(&self.$first_field, $first_field)?;
                $(
                    let $field =
                        super::device_expr_reduce::<$rest, Op>(&self.$field, $field)?;
                )+
                Ok(($first_field, $( $field ),+))
            }
        }
    };
}

impl_reduce_input!(SoVA2<A, B> { left, right });
impl_reduce_input!(SoVA3<A, B, C> { first, second, third });
impl_reduce_input!(SoVA4<A, B, C, D> { a, b, c, d });
impl_reduce_input!(SoVA5<A, B, C, D, E> { a, b, c, d, e });
impl_reduce_input!(SoVA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_reduce_input!(SoVA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_reduce_input!(SoVA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_reduce_input!(SoVA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_reduce_input!(SoVA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_reduce_input!(SoVA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_reduce_input!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Reduces read-only device input to host scalar(s).
///
/// This is a borrowing algorithm: pass `&DeviceVec` for one column or [`vzip`]
/// for multiple read-only columns. No output device storage is allocated.
///
/// [`vzip`]: crate::vzip
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
    /// Reduced scalar type.
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

/// Applies a binary transform over two read-only inputs and reduces the result.
///
/// This is a fused borrowing algorithm. It reads both inputs and returns a host
/// scalar.
pub fn inner_product<Left, Right, TransformOp, ReduceOp>(
    left: Left,
    right: Right,
    init: <Left as InnerProductInput<Right, TransformOp, ReduceOp>>::Item,
    _transform_op: TransformOp,
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
pub trait ReduceByKeyInput<K, Op> {
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

impl<Source, K, Op> ReduceByKeyInput<K, Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    K: CubePrimitive + CubeElement,
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
        SoVA::validate(&self)?;
        let (keys, source) =
            super::device_expr_reduce_by_key::<Source, K, Op>(&self.source, keys, init)?;
        Ok((keys, SoA1 { source }))
    }
}

impl<Source, K, Op> ReduceByKeyInput<K, Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    SoVA1<Source>: ReduceByKeyInput<K, Op>,
    K: CubePrimitive + CubeElement,
{
    type Runtime = <SoVA1<Source> as ReduceByKeyInput<K, Op>>::Runtime;
    type Values = <SoVA1<Source> as ReduceByKeyInput<K, Op>>::Values;
    type Init = <SoVA1<Source> as ReduceByKeyInput<K, Op>>::Init;

    fn reduce_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        <SoVA1<Source> as ReduceByKeyInput<K, Op>>::reduce_by_key_input(
            SoVA1 { source: self },
            keys,
            init,
            op,
        )
    }
}

impl<Left, Right, K, Op> ReduceByKeyInput<K, Op> for SoVA2<Left, Right>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    K: CubePrimitive + CubeElement,
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
        if self.left.len() != self.right.len() {
            return Err(Error::LengthMismatch {
                input: self.left.len(),
                output: self.right.len(),
            });
        }
        let (out_keys, left) =
            super::device_expr_reduce_by_key::<Left, K, Op>(&self.left, keys, init.0)?;
        let (_, right) =
            super::device_expr_reduce_by_key::<Right, K, Op>(&self.right, keys, init.1)?;
        Ok((out_keys, SoA2 { left, right }))
    }
}

impl<First, Second, Third, K, Op> ReduceByKeyInput<K, Op> for SoVA3<First, Second, Third>
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    K: CubePrimitive + CubeElement,
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
        if self.first.len() != self.second.len() {
            return Err(Error::LengthMismatch {
                input: self.first.len(),
                output: self.second.len(),
            });
        }
        if self.first.len() != self.third.len() {
            return Err(Error::LengthMismatch {
                input: self.first.len(),
                output: self.third.len(),
            });
        }
        let (out_keys, first) =
            super::device_expr_reduce_by_key::<First, K, Op>(&self.first, keys, init.0)?;
        let (_, second) =
            super::device_expr_reduce_by_key::<Second, K, Op>(&self.second, keys, init.1)?;
        let (_, third) =
            super::device_expr_reduce_by_key::<Third, K, Op>(&self.third, keys, init.2)?;
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

macro_rules! impl_reduce_by_key_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Key, Op> ReduceByKeyInput<Key, Op> for $input<$first, $( $rest ),+>
        where
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            Key: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: GpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
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
                self.$first_field.validate()?;
                $(
                    self.$field.validate()?;
                    if self.$first_field.len() != self.$field.len() {
                        return Err(Error::LengthMismatch {
                            input: self.$first_field.len(),
                            output: self.$field.len(),
                        });
                    }
                )+
                let ($first_field, $( $field ),+) = init;
                let (out_keys, $first_field) = super::device_expr_reduce_by_key::<$first, Key, Op>(
                    &self.$first_field,
                    keys,
                    $first_field,
                )?;
                $(
                    let (_, $field) = super::device_expr_reduce_by_key::<$rest, Key, Op>(
                        &self.$field,
                        keys,
                        $field,
                    )?;
                )+
                Ok((out_keys, $output { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_reduce_by_key_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_reduce_by_key_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_reduce_by_key_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_reduce_by_key_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_reduce_by_key_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_reduce_by_key_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_reduce_by_key_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_reduce_by_key_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_reduce_by_key_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Reduces contiguous equal-key runs using read-only keys and values.
///
/// This is a borrowing algorithm: values may be a borrowed column or a read-only
/// SoVA from [`vzip`](crate::vzip). The returned keys and values are owned SoA
/// storage.
pub fn reduce_by_key<Values, Keys, R, K, Op>(
    values: Values,
    keys: Keys,
    init: Values::Init,
    _op: Op,
) -> Result<(SoA1<DeviceVec<R, K>>, Values::Values), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Keys: KeyInput<Runtime = R, Item = K>,
    Values: ReduceByKeyInput<K, Op, Runtime = R>,
{
    let keys = keys.key_input()?;
    let (keys, values) = values.reduce_by_key_input(&keys, init, GpuOp::<Op>::new())?;
    Ok((SoA1 { source: keys }, values))
}
