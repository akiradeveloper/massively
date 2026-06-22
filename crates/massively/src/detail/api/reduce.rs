use super::memory::{MaterializeOutput, materialize};
use crate::{
    detail::op::kernel::{BinaryOp2, PredicateOp2},
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoAView1,
        SoAView2, SoAView3,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::{reduce as primitive_reduce, scan as primitive_scan, segmented, select},
};
use cubecl::prelude::*;

/// One-component key input accepted by by-key algorithms.
#[doc(hidden)]
pub trait KeyInput {
    /// CubeCL runtime used by keys.
    type Runtime: Runtime;
    /// Key scalar type.
    type Item;

    /// Materializes keys for primitive kernels.
    fn key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<DeviceVec<Self::Runtime, Self::Item>, Error>;
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

    fn key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<DeviceVec<Self::Runtime, Self::Item>, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_collect_with_policy(policy, &self.source)
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

    fn key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<DeviceVec<Self::Runtime, Self::Item>, Error> {
        <SoAView1<Source> as KeyInput>::key_input(SoAView1 { source: self }, policy)
    }
}

/// Input accepted by [`reduce`].
#[doc(hidden)]
pub trait ReduceInput<Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Initial value type.
    type Init;
    /// Reduction output type.
    type Output;

    /// Reduces this input.
    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Op> ReduceInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp2<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Init = (Source::Item,);
    type Output = (Source::Item,);

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let bindings = self.source.stage(policy)?;
        primitive_reduce::reduce_tuple1_device_expr::<_, _, Source::Expr, Op>(
            policy,
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
    type Runtime = <SoAView1<Source> as ReduceInput<Op>>::Runtime;
    type Init = <SoAView1<Source> as ReduceInput<Op>>::Init;
    type Output = <SoAView1<Source> as ReduceInput<Op>>::Output;

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as ReduceInput<Op>>::reduce_input(
            SoAView1 { source: self.0 },
            policy,
            init,
            op,
        )
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
            Op: BinaryOp2<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );

            fn reduce_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = self.$first_field.stage(policy)?;
                $(
                    let $field = self.$field.stage(policy)?;
                )+
                primitive_reduce::$reduce_fn::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                    $( <$rest as KernelColumn>::Item, )+
                    <$first as KernelColumn>::Expr,
                    $( <$rest as KernelColumn>::Expr, )+
                    Op,
                >(
                    policy,
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
    type Runtime = <SoAView2<Left, Right> as ReduceInput<Op>>::Runtime;
    type Init = <SoAView2<Left, Right> as ReduceInput<Op>>::Init;
    type Output = <SoAView2<Left, Right> as ReduceInput<Op>>::Output;

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ReduceInput<Op>>::reduce_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            init,
            op,
        )
    }
}

impl<First, Second, Third, Op> ReduceInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ReduceInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Runtime;
    type Init = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Init;
    type Output = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Output;

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ReduceInput<Op>>::reduce_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
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
            Op: BinaryOp2<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );

            fn reduce_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = self.$first_field.stage(policy)?;
                $(
                    let $field = self.$field.stage(policy)?;
                )+
                primitive_reduce::$reduce_fn::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                    $( <$rest as KernelColumn>::Item, )+
                    <$first as KernelColumn>::Expr,
                    $( <$rest as KernelColumn>::Expr, )+
                    Op,
                >(
                    policy,
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
    policy: &CubePolicy<<Input as ReduceInput<Op>>::Runtime>,
    input: Input,
    init: <Input as ReduceInput<Op>>::Init,
    _op: Op,
) -> Result<<Input as ReduceInput<Op>>::Output, Error>
where
    Input: ReduceInput<Op>,
{
    input.reduce_input(policy, init, GpuOp::<Op>::new())
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
        policy: &CubePolicy<Self::Runtime>,
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
    KeyEq: PredicateOp2<K>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp2<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Init = Source::Item;
    type Values = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reduce_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        ReadOnlySoA::validate(&self)?;
        let (keys, source) = super::device_expr_reduce_by_key_with_policy::<Source, K, KeyEq, Op>(
            policy,
            &self.source,
            keys,
            init,
        )?;
        Ok((keys, SoA1 { source }))
    }
}

impl<Source, K, KeyEq, Op> ReduceByKeyInput<K, KeyEq, Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    SoAView1<Source>: ReduceByKeyInput<K, KeyEq, Op>,
    K: CubePrimitive + CubeElement,
    KeyEq: PredicateOp2<K>,
{
    type Runtime = <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::Runtime;
    type Values = <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::Values;
    type Init = <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::Init;

    fn reduce_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        <SoAView1<Source> as ReduceByKeyInput<K, KeyEq, Op>>::reduce_by_key_input(
            SoAView1 { source: self },
            policy,
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
    KeyEq: PredicateOp2<K>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: BinaryOp2<(Left::Item, Right::Item)>,
{
    type Runtime = Left::Runtime;
    type Init = (Left::Item, Right::Item);
    type Values = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn reduce_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        self.left.validate()?;
        self.right.validate()?;
        super::ensure_same_len(self.right.len(), self.left.len())?;
        let left = super::device_expr_collect_with_policy(policy, &self.left)?;
        let right = super::device_expr_collect_with_policy(policy, &self.right)?;
        let scanned = primitive_scan::inclusive_scan_tuple2_by_key_values_device_vec(
            policy,
            keys,
            &left,
            &right,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let control =
            segmented::key_run_end_control_with_policy::<Self::Runtime, K, KeyEq>(policy, keys)?;
        let out_keys = control.compact_first::<Self::Runtime, K>(policy)?;
        let left = control
            .compact_value::<Self::Runtime, Left::Item>(policy, scanned.left.handle.clone())?;
        let right = control
            .compact_value::<Self::Runtime, Right::Item>(policy, scanned.right.handle.clone())?;
        let (left, right) =
            primitive_reduce::apply_tuple2_init::<Self::Runtime, Left::Item, Right::Item, Op>(
                policy, &left, &right, init,
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
    KeyEq: PredicateOp2<K>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Op: BinaryOp2<(First::Item, Second::Item, Third::Item)>,
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
        policy: &CubePolicy<Self::Runtime>,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<(DeviceVec<Self::Runtime, K>, Self::Values), Error> {
        self.first.validate()?;
        self.second.validate()?;
        self.third.validate()?;
        super::ensure_same_len(self.second.len(), self.first.len())?;
        super::ensure_same_len(self.third.len(), self.first.len())?;
        let first = super::device_expr_collect_with_policy(policy, &self.first)?;
        let second = super::device_expr_collect_with_policy(policy, &self.second)?;
        let third = super::device_expr_collect_with_policy(policy, &self.third)?;
        let scanned = primitive_scan::inclusive_scan_tuple3_by_key_values_device_vec(
            policy,
            keys,
            &first,
            &second,
            &third,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let control =
            segmented::key_run_end_control_with_policy::<Self::Runtime, K, KeyEq>(policy, keys)?;
        let out_keys = control.compact_first::<Self::Runtime, K>(policy)?;
        let first = control
            .compact_value::<Self::Runtime, First::Item>(policy, scanned.first.handle.clone())?;
        let second = control
            .compact_value::<Self::Runtime, Second::Item>(policy, scanned.second.handle.clone())?;
        let third = control
            .compact_value::<Self::Runtime, Third::Item>(policy, scanned.third.handle.clone())?;
        let (first, second, third) = primitive_reduce::apply_tuple3_init::<
            Self::Runtime,
            First::Item,
            Second::Item,
            Third::Item,
            Op,
        >(policy, &first, &second, &third, init)?;
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
            KeyEq: PredicateOp2<Key>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Op: BinaryOp2<<$first as KernelColumn>::Item>,
            $(
                Op: BinaryOp2<<$rest as KernelColumn>::Item>,
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
                policy: &CubePolicy<Self::Runtime>,
                keys: &DeviceVec<Self::Runtime, Key>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<(DeviceVec<Self::Runtime, Key>, Self::Values), Error> {
                SoA::validate(&self)?;
                let _ = policy;
                let ($first_field, $( $field ),+) = init;
                let (out_keys, $first_field, control) = super::device_expr_reduce_by_key_with_control_with_policy::<$first, Key, KeyEq, Op>(
                    policy,
                    &self.$first_field,
                    keys,
                    $first_field,
                )?;
                $(
                    let $field = super::device_expr_reduce_by_key_with_existing_control_with_policy::<$rest, Key, KeyEq, Op>(
                        policy,
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
                policy: &CubePolicy<Self::Runtime>,
                keys: &DeviceVec<Self::Runtime, Key>,
                init: Self::Init,
                op: GpuOp<Op>,
            ) -> Result<(DeviceVec<Self::Runtime, Key>, Self::Values), Error> {
                <$view<$( $ty ),+> as ReduceByKeyInput<Key, KeyEq, Op>>::reduce_by_key_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
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
    type Runtime: Runtime;
    type Init;
    type Output;

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
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
    KeyEq: PredicateOp2<Keys::Item>,
    Values: ReduceByKeyInput<Keys::Item, KeyEq, Op, Runtime = Keys::Runtime>,
{
    type Runtime = Keys::Runtime;
    type Init = <Values as ReduceByKeyInput<Keys::Item, KeyEq, Op>>::Init;
    type Output = (
        SoA1<DeviceVec<Keys::Runtime, Keys::Item>>,
        <Values as ReduceByKeyInput<Keys::Item, KeyEq, Op>>::Values,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.key_input(policy)?;
        let (keys, values) = values.reduce_by_key_input(policy, &keys, init, GpuOp::<Op>::new())?;
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
    KeyEq: PredicateOp2<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp2<ValueSource::Item>,
{
    type Runtime = KeyA::Runtime;
    type Init = ValueSource::Item;
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: ValueSource,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.left)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.right)?;
        let values = super::device_expr_collect_with_policy(policy, &values.source)?;
        let (left, right, source) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &values, init,
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
    KeyEq: PredicateOp2<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp2<ValueA::Item>,
    Op: BinaryOp2<ValueB::Item>,
{
    type Runtime = KeyA::Runtime;
    type Init = (ValueA::Item, ValueB::Item);
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: SoAView2<ValueA, ValueB>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.left)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.right)?;
        let value_a = super::device_expr_collect_with_policy(policy, &values.left)?;
        let value_b = super::device_expr_collect_with_policy(policy, &values.right)?;
        let (left, right, value_a) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &value_a, init.0,
            )?;
        let (_, _, value_b) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &value_b, init.1,
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
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    KeyEq: PredicateOp2<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp2<ValueA::Item>,
    Op: BinaryOp2<ValueB::Item>,
    Op: BinaryOp2<ValueC::Item>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: SoAView3<ValueA, ValueB, ValueC>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.left)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.right)?;
        let value_a = super::device_expr_collect_with_policy(policy, &values.first)?;
        let value_b = super::device_expr_collect_with_policy(policy, &values.second)?;
        let value_c = super::device_expr_collect_with_policy(policy, &values.third)?;
        let (left, right, value_a) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &value_a, init.0,
            )?;
        let (_, _, value_b) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &value_b, init.1,
            )?;
        let (_, _, value_c) =
            primitive_reduce::reduce_tuple2_by_key_device_vec::<_, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &value_c, init.2,
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
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    KeyC::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    KeyEq: PredicateOp2<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp2<ValueSource::Item>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: ValueSource,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let key_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let values = super::device_expr_collect_with_policy(policy, &values.source)?;
        let (first, second, third, source) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &key_c, &values, init,
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
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    KeyC::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    KeyEq: PredicateOp2<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp2<ValueA::Item>,
    Op: BinaryOp2<ValueB::Item>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: SoAView2<ValueA, ValueB>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let key_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let value_a = super::device_expr_collect_with_policy(policy, &values.left)?;
        let value_b = super::device_expr_collect_with_policy(policy, &values.right)?;
        let (first, second, third, value_a) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &key_c, &value_a, init.0,
            )?;
        let (_, _, _, value_b) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &key_c, &value_b, init.1,
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
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    KeyC::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    KeyEq: PredicateOp2<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp2<ValueA::Item>,
    Op: BinaryOp2<ValueB::Item>,
    Op: BinaryOp2<ValueC::Item>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: SoAView3<ValueA, ValueB, ValueC>,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let key_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let value_a = super::device_expr_collect_with_policy(policy, &values.first)?;
        let value_b = super::device_expr_collect_with_policy(policy, &values.second)?;
        let value_c = super::device_expr_collect_with_policy(policy, &values.third)?;
        let (first, second, third, value_a) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &key_c, &value_a, init.0,
            )?;
        let (_, _, _, value_b) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &key_c, &value_b, init.1,
            )?;
        let (_, _, _, value_c) =
            primitive_reduce::reduce_tuple3_by_key_device_vec::<_, _, _, _, _, KeyEq, Op>(
                policy, &key_a, &key_b, &key_c, &value_c, init.2,
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
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueSource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
            KeyEq: PredicateOp2<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp2<ValueSource::Item>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
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
                policy: &CubePolicy<Self::Runtime>,
                values: ValueSource,
                _key_eq: GpuOp<KeyEq>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                values.validate()?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $( let $field = super::device_expr_collect_with_policy(policy, &self.$field)?; )+
                let values = super::device_expr_collect_with_policy(policy, &values)?;
                let ($first_out, $( $out_field, )+ source) =
                    primitive_reduce::$reduce_fn::<
                        <$first as KernelColumn>::Runtime,
                        <$first as KernelColumn>::Item,
                        $( <$key as KernelColumn>::Item, )+
                        ValueSource::Item,
                        KeyEq,
                        Op,
                    >(
                        policy,
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
    (@reduce $reduce_fn:ident, $eq:ident, $policy:ident, ($first_field:ident, $( $field:ident ),+), $value_ty:ident, $value_field:ident, $init:expr, ($first:ident, $( $key:ident ),+)) => {
        primitive_reduce::$reduce_fn::<
            <$first as KernelColumn>::Runtime,
            <$first as KernelColumn>::Item,
            $( <$key as KernelColumn>::Item, )+
            <$value_ty as KernelColumn>::Item,
            $eq,
            Op,
        >(
            $policy,
            &$first_field,
            $( &$field, )+
            &$value_field,
            $init,
        )
    };
    (@reduce_values $reduce_fn:ident, $eq:ident, $policy:ident, $value_index:tt, ($first_field:ident, $( $field:ident ),+), ($first:ident, $( $key:ident ),+), $init:ident, ) => {};
    (@reduce_values $reduce_fn:ident, $eq:ident, $policy:ident, $value_index:tt, ($first_field:ident, $( $field:ident ),+), ($first:ident, $( $key:ident ),+), $init:ident, $value_ty:ident: $value_field:ident: $idx:tt $(, $tail_ty:ident: $tail_field:ident: $tail_idx:tt )*) => {
        let $value_field = impl_reduce_by_tuple_key_soa_view_values!(
            @reduce $reduce_fn,
            $eq,
            $policy,
            ($first_field, $( $field ),+),
            $value_ty,
            $value_field,
            $init.$idx,
            ($first, $( $key ),+)
        )?.$value_index;
        impl_reduce_by_tuple_key_soa_view_values!(
            @reduce_values $reduce_fn,
            $eq,
            $policy,
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
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first_value as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            <$first_value as KernelColumn>::Expr: DeviceGpuExpr<<$first_value as KernelColumn>::Item>,
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            KeyEq: PredicateOp2<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp2<<$first_value as KernelColumn>::Item>,
            $( Op: BinaryOp2<<$value as KernelColumn>::Item>, )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
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
                policy: &CubePolicy<Self::Runtime>,
                values: $values<$first_value, $( $value ),+>,
                _key_eq: GpuOp<KeyEq>,
                init: Self::Init,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                $key_storage::validate(&self)?;
                $storage::validate(&values)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $( let $field = super::device_expr_collect_with_policy(policy, &self.$field)?; )+
                let $first_value_field = super::device_expr_collect_with_policy(policy, &values.$first_value_field)?;
                $( let $value_field = super::device_expr_collect_with_policy(policy, &values.$value_field)?; )+
                let ($first_out, $( $out_field, )+ $first_value_field) = impl_reduce_by_tuple_key_soa_view_values!(
                    @reduce $reduce_fn,
                    KeyEq,
                    policy,
                    ($first_field, $( $field ),+),
                    $first_value,
                    $first_value_field,
                    init.$first_idx,
                    ($first, $( $key ),+)
                )?;
                impl_reduce_by_tuple_key_soa_view_values!(
                    @reduce_values $reduce_fn,
                    KeyEq,
                    policy,
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
            type Runtime = <$view<$( $ty ),+> as ReduceByKeyCall<Values, KeyEq, Op>>::Runtime;
            type Init = <$view<$( $ty ),+> as ReduceByKeyCall<Values, KeyEq, Op>>::Init;
            type Output = <$view<$( $ty ),+> as ReduceByKeyCall<Values, KeyEq, Op>>::Output;

            fn reduce_by_key_call(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: Values,
                key_eq: GpuOp<KeyEq>,
                init: Self::Init,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as ReduceByKeyCall<Values, KeyEq, Op>>::reduce_by_key_call(
                    $view { $( $field: self.$index ),+ },
                    policy,
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
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    KeyEq: PredicateOp2<(KeySource::Item,)>,
    Op: BinaryOp2<(ValueSource::Item,)>,
{
    type Runtime = KeySource::Runtime;
    type Init = (ValueSource::Item,);
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueSource,),
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        self.0.validate()?;
        values.0.validate()?;
        super::ensure_same_len(values.0.len(), self.0.len())?;
        let len = self.0.len();
        if len == 0 {
            return Ok((
                SoA1 {
                    source: policy.empty_device_vec(),
                },
                SoA1 {
                    source: policy.empty_device_vec(),
                },
            ));
        }

        let client = policy.client();
        let key_bindings = self.0.stage(policy)?;
        let value_bindings = values.0.stage(policy)?;
        let inclusive_handle = primitive_scan::inclusive_scan_by_key_device_expr_handle::<
            KeySource::Runtime,
            KeySource::Item,
            ValueSource::Item,
            KeySource::Expr,
            ValueSource::Expr,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
        >(policy, &key_bindings, &value_bindings, len)?;

        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let init_handle = client.create_from_slice(ValueSource::Item::as_bytes(&[init.0]));
        let flag_handle = client.empty(len * std::mem::size_of::<u32>());
        let reduced_value_handle = client.empty(len * std::mem::size_of::<ValueSource::Item>());
        let key_slot0 = key_bindings.slots.first().unwrap();
        let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
        let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
        let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
        let key_offsets = key_bindings.slot_offsets_handle(client)?;
        let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_device_expr_end_flags_kernel::launch_unchecked::<
                KeySource::Item,
                ValueSource::Item,
                KeySource::Expr,
                super::Tuple1Less<KeyEq>,
                super::Tuple1BinaryOp<Op>,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(inclusive_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_value_handle.clone(), len) },
            );
        }

        let out_keys = super::device_expr_compact_with_flags_with_policy(
            policy,
            &self.0,
            flag_handle.clone(),
        )?;
        let handles =
            select::handles_from_flags(policy, len, len_u32, flag_handle, reduced_value_handle)?;
        let out_values = select::compact::<KeySource::Runtime, ValueSource::Item>(policy, handles)?;
        Ok((SoA1 { source: out_keys }, SoA1 { source: out_values }))
    }
}

impl<KeySource, ValueA, ValueB, KeyEq, Op> ReduceByKeyCall<(ValueA, ValueB), KeyEq, Op>
    for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    KeyEq: PredicateOp2<(KeySource::Item,)>,
    Op: BinaryOp2<(ValueA::Item, ValueB::Item)>,
{
    type Runtime = KeySource::Runtime;
    type Init = (ValueA::Item, ValueB::Item);
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA2<
            DeviceVec<KeySource::Runtime, ValueA::Item>,
            DeviceVec<KeySource::Runtime, ValueB::Item>,
        >,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueA, ValueB),
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        self.0.validate()?;
        values.0.validate()?;
        values.1.validate()?;
        super::ensure_same_len(values.0.len(), self.0.len())?;
        super::ensure_same_len(values.1.len(), self.0.len())?;
        let len = self.0.len();
        if len == 0 {
            return Ok((
                SoA1 {
                    source: policy.empty_device_vec(),
                },
                SoA2 {
                    left: policy.empty_device_vec(),
                    right: policy.empty_device_vec(),
                },
            ));
        }

        let client = policy.client();
        let key_bindings = self.0.stage(policy)?;
        let a_bindings = values.0.stage(policy)?;
        let b_bindings = values.1.stage(policy)?;
        let (inclusive_a, inclusive_b) =
            primitive_scan::inclusive_scan_tuple2_by_key_values_device_expr_handle::<
                KeySource::Runtime,
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                KeySource::Expr,
                ValueA::Expr,
                ValueB::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
            >(policy, &key_bindings, &a_bindings, &b_bindings, len)?;

        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let flag_handle = client.empty(len * std::mem::size_of::<u32>());
        let reduced_a_handle = client.empty(len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(len * std::mem::size_of::<ValueB::Item>());
        let key_slot0 = key_bindings.slots.first().unwrap();
        let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
        let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
        let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
        let key_offsets = key_bindings.slot_offsets_handle(client)?;
        let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple2_device_expr_end_flags_kernel::launch_unchecked::<
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                KeySource::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(inclusive_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(inclusive_b.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_b_handle.clone(), len) },
            );
        }

        let out_keys = super::device_expr_compact_with_flags_with_policy(
            policy,
            &self.0,
            flag_handle.clone(),
        )?;
        let left_handles = select::handles_from_flags(
            policy,
            len,
            len_u32,
            flag_handle.clone(),
            reduced_a_handle,
        )?;
        let right_handles =
            select::handles_from_flags(policy, len, len_u32, flag_handle, reduced_b_handle)?;
        let left = select::compact::<KeySource::Runtime, ValueA::Item>(policy, left_handles)?;
        let right = select::compact::<KeySource::Runtime, ValueB::Item>(policy, right_handles)?;

        Ok((SoA1 { source: out_keys }, SoA2 { left, right }))
    }
}

impl<KeySource, ValueA, ValueB, ValueC, KeyEq, Op>
    ReduceByKeyCall<(ValueA, ValueB, ValueC), KeyEq, Op> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    ValueC: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    KeyEq: PredicateOp2<(KeySource::Item,)>,
    Op: BinaryOp2<(ValueA::Item, ValueB::Item, ValueC::Item)>,
{
    type Runtime = KeySource::Runtime;
    type Init = (ValueA::Item, ValueB::Item, ValueC::Item);
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA3<
            DeviceVec<KeySource::Runtime, ValueA::Item>,
            DeviceVec<KeySource::Runtime, ValueB::Item>,
            DeviceVec<KeySource::Runtime, ValueC::Item>,
        >,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueA, ValueB, ValueC),
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        self.0.validate()?;
        values.0.validate()?;
        values.1.validate()?;
        values.2.validate()?;
        super::ensure_same_len(values.0.len(), self.0.len())?;
        super::ensure_same_len(values.1.len(), self.0.len())?;
        super::ensure_same_len(values.2.len(), self.0.len())?;
        let len = self.0.len();
        if len == 0 {
            return Ok((
                SoA1 {
                    source: policy.empty_device_vec(),
                },
                SoA3 {
                    first: policy.empty_device_vec(),
                    second: policy.empty_device_vec(),
                    third: policy.empty_device_vec(),
                },
            ));
        }

        let client = policy.client();
        let key_bindings = self.0.stage(policy)?;
        let a_bindings = values.0.stage(policy)?;
        let b_bindings = values.1.stage(policy)?;
        let c_bindings = values.2.stage(policy)?;
        let (inclusive_a, inclusive_b, inclusive_c) =
            primitive_scan::inclusive_scan_tuple3_by_key_values_device_expr_handle::<
                KeySource::Runtime,
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                ValueC::Item,
                KeySource::Expr,
                ValueA::Expr,
                ValueB::Expr,
                ValueC::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
            >(
                policy,
                &key_bindings,
                &a_bindings,
                &b_bindings,
                &c_bindings,
                len,
            )?;

        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let init_c = client.create_from_slice(ValueC::Item::as_bytes(&[init.2]));
        let flag_handle = client.empty(len * std::mem::size_of::<u32>());
        let reduced_a_handle = client.empty(len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(len * std::mem::size_of::<ValueB::Item>());
        let reduced_c_handle = client.empty(len * std::mem::size_of::<ValueC::Item>());
        let key_slot0 = key_bindings.slots.first().unwrap();
        let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
        let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
        let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
        let key_offsets = key_bindings.slot_offsets_handle(client)?;
        let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple3_device_expr_end_flags_kernel::launch_unchecked::<
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                ValueC::Item,
                KeySource::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(inclusive_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(inclusive_b.clone(), len) },
                unsafe { BufferArg::from_raw_parts(inclusive_c.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(init_c.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_c_handle.clone(), len) },
            );
        }

        let out_keys = super::device_expr_compact_with_flags_with_policy(
            policy,
            &self.0,
            flag_handle.clone(),
        )?;
        let first_handles = select::handles_from_flags(
            policy,
            len,
            len_u32,
            flag_handle.clone(),
            reduced_a_handle,
        )?;
        let second_handles = select::handles_from_flags(
            policy,
            len,
            len_u32,
            flag_handle.clone(),
            reduced_b_handle,
        )?;
        let third_handles =
            select::handles_from_flags(policy, len, len_u32, flag_handle, reduced_c_handle)?;
        let first = select::compact::<KeySource::Runtime, ValueA::Item>(policy, first_handles)?;
        let second = select::compact::<KeySource::Runtime, ValueB::Item>(policy, second_handles)?;
        let third = select::compact::<KeySource::Runtime, ValueC::Item>(policy, third_handles)?;

        Ok((
            SoA1 { source: out_keys },
            SoA3 {
                first,
                second,
                third,
            },
        ))
    }
}

/// Reduces contiguous equal-key runs using read-only keys and values.
///
/// This is a borrowing algorithm: values may be a borrowed column or a read-only
/// SoA from [`zip`](crate::zip). The returned keys and values are owned SoA
/// storage.
pub fn reduce_by_key<R, Keys, Values, KeyEq, Op>(
    policy: &CubePolicy<R>,
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
    R: Runtime,
    Keys: ReduceByKeyCall<Values, KeyEq, Op, Runtime = R>,
    <Keys as ReduceByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        keys.reduce_by_key_call(
            policy,
            values,
            GpuOp::<KeyEq>::new(),
            init,
            GpuOp::<Op>::new(),
        )?,
    )
}
