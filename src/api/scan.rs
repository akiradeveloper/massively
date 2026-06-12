use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, S0, SoA1, SoA2, SoA3, SoA4, SoA5, SoA6, SoA7,
        SoA8, SoA9, SoA10, SoA11, SoA12, SoVA, SoVA1, SoVA2, SoVA3, SoVA4, SoVA5, SoVA6, SoVA7,
        SoVA8, SoVA9, SoVA10, SoVA11, SoVA12,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr, Input},
    op::{BinaryOp, BinaryPredicateOp, GpuOp},
};
use cubecl::prelude::*;

/// One-component key input accepted by by-key scan algorithms.
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

/// Input accepted by [`inclusive_scan`].
#[doc(hidden)]
pub trait InclusiveScanInput<Op> {
    /// Scan output type.
    type Output;

    /// Computes an inclusive scan.
    fn inclusive_scan_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error>;
}

impl<Source, Op> InclusiveScanInput<Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
    Input<Source::Item>: GpuExpr<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn inclusive_scan_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        Ok(SoA1 {
            source: super::device_expr_inclusive_scan::<Source, Op>(&self.source)?,
        })
    }
}

impl<Source, Op> InclusiveScanInput<Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
    Input<Source::Item>: GpuExpr<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn inclusive_scan_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoVA1<Source> as InclusiveScanInput<Op>>::inclusive_scan_input(SoVA1 { source: self }, op)
    }
}

macro_rules! impl_inclusive_scan_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> InclusiveScanInput<Op> for $input<$first, $( $rest ),+>
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
            Input<<$first as KernelColumn>::Item>: GpuExpr<<$first as KernelColumn>::Item>,
            $(
                Op: BinaryOp<<$rest as KernelColumn>::Item>,
                Input<<$rest as KernelColumn>::Item>: GpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn inclusive_scan_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                let $first_field =
                    super::device_expr_inclusive_scan::<$first, Op>(&self.$first_field)?;
                $(
                    let $field =
                        super::device_expr_inclusive_scan::<$rest, Op>(&self.$field)?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_inclusive_scan_input!(SoVA2 -> SoA2<A, B> { left, right });
impl_inclusive_scan_input!(SoVA3 -> SoA3<A, B, C> { first, second, third });
impl_inclusive_scan_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_inclusive_scan_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_inclusive_scan_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_inclusive_scan_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_inclusive_scan_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_inclusive_scan_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_inclusive_scan_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_inclusive_scan_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_inclusive_scan_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Computes an inclusive scan from read-only input into device storage.
pub fn inclusive_scan<InputSource, Op>(
    source: InputSource,
    _op: Op,
) -> Result<<InputSource as InclusiveScanInput<Op>>::Output, Error>
where
    InputSource: InclusiveScanInput<Op>,
{
    source.inclusive_scan_input(GpuOp::<Op>::new())
}

/// Input accepted by [`exclusive_scan`].
#[doc(hidden)]
pub trait ExclusiveScanInput<Op> {
    /// Initial value type.
    type Init;
    /// Scan output type.
    type Output;

    /// Computes an exclusive scan.
    fn exclusive_scan_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error>;
}

impl<Source, Op> ExclusiveScanInput<Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
    Input<Source::Item>: GpuExpr<Source::Item>,
{
    type Init = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn exclusive_scan_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        Ok(SoA1 {
            source: super::device_expr_exclusive_scan::<Source, Op>(&self.source, init)?,
        })
    }
}

impl<Source, Op> ExclusiveScanInput<Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
    Input<Source::Item>: GpuExpr<Source::Item>,
{
    type Init = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn exclusive_scan_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoVA1<Source> as ExclusiveScanInput<Op>>::exclusive_scan_input(
            SoVA1 { source: self },
            init,
            op,
        )
    }
}

macro_rules! impl_exclusive_scan_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> ExclusiveScanInput<Op> for $input<$first, $( $rest ),+>
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
            Input<<$first as KernelColumn>::Item>: GpuExpr<<$first as KernelColumn>::Item>,
            $(
                Op: BinaryOp<<$rest as KernelColumn>::Item>,
                Input<<$rest as KernelColumn>::Item>: GpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn exclusive_scan_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                let ($first_field, $( $field ),+) = init;
                let $first_field =
                    super::device_expr_exclusive_scan::<$first, Op>(&self.$first_field, $first_field)?;
                $(
                    let $field =
                        super::device_expr_exclusive_scan::<$rest, Op>(&self.$field, $field)?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_exclusive_scan_input!(SoVA2 -> SoA2<A, B> { left, right });
impl_exclusive_scan_input!(SoVA3 -> SoA3<A, B, C> { first, second, third });
impl_exclusive_scan_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_exclusive_scan_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_exclusive_scan_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_exclusive_scan_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_exclusive_scan_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_exclusive_scan_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_exclusive_scan_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_exclusive_scan_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_exclusive_scan_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Computes an exclusive scan from read-only input into device storage.
pub fn exclusive_scan<InputSource, Op>(
    source: InputSource,
    init: <InputSource as ExclusiveScanInput<Op>>::Init,
    _op: Op,
) -> Result<<InputSource as ExclusiveScanInput<Op>>::Output, Error>
where
    InputSource: ExclusiveScanInput<Op>,
{
    source.exclusive_scan_input(init, GpuOp::<Op>::new())
}

/// Input accepted by [`adjacent_difference`].
#[doc(hidden)]
pub trait AdjacentDifferenceInput<Op> {
    /// Adjacent difference output type.
    type Output;

    /// Computes adjacent differences.
    fn adjacent_difference_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error>;
}

impl<Source, Op> AdjacentDifferenceInput<Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn adjacent_difference_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        let source = super::device_expr_adjacent_difference::<Source, Op>(&self)?;
        Ok(SoA1 { source })
    }
}

/// Computes adjacent differences into device storage.
pub fn adjacent_difference<Source, Op>(
    source: Source,
    _op: Op,
) -> Result<<Source as AdjacentDifferenceInput<Op>>::Output, Error>
where
    Source: AdjacentDifferenceInput<Op>,
{
    source.adjacent_difference_input(GpuOp::<Op>::new())
}

/// Input accepted by [`inclusive_scan_by_key`].
#[doc(hidden)]
pub trait InclusiveScanByKeyInput<K, KeyEq, Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Scan output type.
    type Output;

    /// Computes an inclusive scan by key.
    fn inclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, K, KeyEq, Op> InclusiveScanByKeyInput<K, KeyEq, Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    K: CubePrimitive + CubeElement + PartialEq,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn inclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        Ok(SoA1 {
            source: super::device_expr_inclusive_scan_by_key::<Source, K, KeyEq, Op>(
                &self.source,
                keys,
            )?,
        })
    }
}

impl<Source, K, KeyEq, Op> InclusiveScanByKeyInput<K, KeyEq, Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    SoVA1<Source>: InclusiveScanByKeyInput<K, KeyEq, Op>,
    K: CubePrimitive + CubeElement,
{
    type Runtime = <SoVA1<Source> as InclusiveScanByKeyInput<K, KeyEq, Op>>::Runtime;
    type Output = <SoVA1<Source> as InclusiveScanByKeyInput<K, KeyEq, Op>>::Output;

    fn inclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<Source> as InclusiveScanByKeyInput<K, KeyEq, Op>>::inclusive_scan_by_key_input(
            SoVA1 { source: self },
            keys,
            key_eq,
            op,
        )
    }
}

impl<Left, Right, K, KeyEq, Op> InclusiveScanByKeyInput<K, KeyEq, Op> for SoVA2<Left, Right>
where
    Self: SoVA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    K: CubePrimitive + CubeElement + PartialEq,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<Left::Item>,
    Op: BinaryOp<Right::Item>,
{
    type Runtime = Left::Runtime;
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn inclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let first =
            super::device_expr_inclusive_scan_by_key::<Left, K, KeyEq, Op>(&self.left, keys)?;
        let second =
            super::device_expr_inclusive_scan_by_key::<Right, K, KeyEq, Op>(&self.right, keys)?;
        Ok(SoA2 {
            left: first,
            right: second,
        })
    }
}

impl<First, Second, Third, K, KeyEq, Op> InclusiveScanByKeyInput<K, KeyEq, Op>
    for SoVA3<First, Second, Third>
where
    Self: SoVA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    K: CubePrimitive + CubeElement + PartialEq,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<First::Item>,
    Op: BinaryOp<Second::Item>,
    Op: BinaryOp<Third::Item>,
{
    type Runtime = First::Runtime;
    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn inclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let first =
            super::device_expr_inclusive_scan_by_key::<First, K, KeyEq, Op>(&self.first, keys)?;
        let second =
            super::device_expr_inclusive_scan_by_key::<Second, K, KeyEq, Op>(&self.second, keys)?;
        let third =
            super::device_expr_inclusive_scan_by_key::<Third, K, KeyEq, Op>(&self.third, keys)?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

macro_rules! impl_inclusive_scan_by_key_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Key, KeyEq, Op> InclusiveScanByKeyInput<Key, KeyEq, Op>
            for $input<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            Key: CubePrimitive + CubeElement + PartialEq,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            KeyEq: BinaryPredicateOp<Key>,
            Op: BinaryOp<<$first as KernelColumn>::Item>,
            $(
                Op: BinaryOp<<$rest as KernelColumn>::Item>,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn inclusive_scan_by_key_input(
                self,
                keys: &DeviceVec<Self::Runtime, Key>,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                let $first_field =
                    super::device_expr_inclusive_scan_by_key::<$first, Key, KeyEq, Op>(
                        &self.$first_field,
                        keys,
                    )?;
                $(
                    let $field =
                        super::device_expr_inclusive_scan_by_key::<$rest, Key, KeyEq, Op>(
                            &self.$field,
                            keys,
                        )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_inclusive_scan_by_key_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_inclusive_scan_by_key_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_inclusive_scan_by_key_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_inclusive_scan_by_key_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_inclusive_scan_by_key_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_inclusive_scan_by_key_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_inclusive_scan_by_key_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_inclusive_scan_by_key_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_inclusive_scan_by_key_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Computes an inclusive scan by key.
pub fn inclusive_scan_by_key<Values, Keys, R, K, KeyEq, Op>(
    values: Values,
    keys: Keys,
    _key_eq: KeyEq,
    _op: Op,
) -> Result<<Values as InclusiveScanByKeyInput<K, KeyEq, Op>>::Output, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Keys: KeyInput<Runtime = R, Item = K>,
    Values: InclusiveScanByKeyInput<K, KeyEq, Op, Runtime = R>,
{
    let keys = keys.key_input()?;
    values.inclusive_scan_by_key_input(&keys, GpuOp::<KeyEq>::new(), GpuOp::<Op>::new())
}

/// Input accepted by [`exclusive_scan_by_key`].
#[doc(hidden)]
pub trait ExclusiveScanByKeyInput<K, KeyEq, Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Initial value type.
    type Init;
    /// Scan output type.
    type Output;

    /// Computes an exclusive scan by key.
    fn exclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, K, KeyEq, Op> ExclusiveScanByKeyInput<K, KeyEq, Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    K: CubePrimitive + CubeElement + PartialEq,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Init = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn exclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        Ok(SoA1 {
            source: super::device_expr_exclusive_scan_by_key::<Source, K, KeyEq, Op>(
                &self.source,
                keys,
                init,
            )?,
        })
    }
}

impl<Source, K, KeyEq, Op> ExclusiveScanByKeyInput<K, KeyEq, Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    SoVA1<Source>: ExclusiveScanByKeyInput<K, KeyEq, Op>,
    K: CubePrimitive + CubeElement,
{
    type Runtime = <SoVA1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Runtime;
    type Init = <SoVA1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Init;
    type Output = <SoVA1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Output;

    fn exclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::exclusive_scan_by_key_input(
            SoVA1 { source: self },
            keys,
            init,
            key_eq,
            op,
        )
    }
}

impl<Left, Right, K, KeyEq, Op> ExclusiveScanByKeyInput<K, KeyEq, Op> for SoVA2<Left, Right>
where
    Self: SoVA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    K: CubePrimitive + CubeElement + PartialEq,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<Left::Item>,
    Op: BinaryOp<Right::Item>,
{
    type Runtime = Left::Runtime;
    type Init = (Left::Item, Right::Item);
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn exclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let first = super::device_expr_exclusive_scan_by_key::<Left, K, KeyEq, Op>(
            &self.left, keys, init.0,
        )?;
        let second = super::device_expr_exclusive_scan_by_key::<Right, K, KeyEq, Op>(
            &self.right,
            keys,
            init.1,
        )?;
        Ok(SoA2 {
            left: first,
            right: second,
        })
    }
}

impl<First, Second, Third, K, KeyEq, Op> ExclusiveScanByKeyInput<K, KeyEq, Op>
    for SoVA3<First, Second, Third>
where
    Self: SoVA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    K: CubePrimitive + CubeElement + PartialEq,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<First::Item>,
    Op: BinaryOp<Second::Item>,
    Op: BinaryOp<Third::Item>,
{
    type Runtime = First::Runtime;
    type Init = (First::Item, Second::Item, Third::Item);
    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn exclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let first = super::device_expr_exclusive_scan_by_key::<First, K, KeyEq, Op>(
            &self.first,
            keys,
            init.0,
        )?;
        let second = super::device_expr_exclusive_scan_by_key::<Second, K, KeyEq, Op>(
            &self.second,
            keys,
            init.1,
        )?;
        let third = super::device_expr_exclusive_scan_by_key::<Third, K, KeyEq, Op>(
            &self.third,
            keys,
            init.2,
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

macro_rules! impl_exclusive_scan_by_key_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Key, KeyEq, Op> ExclusiveScanByKeyInput<Key, KeyEq, Op>
            for $input<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            Key: CubePrimitive + CubeElement + PartialEq,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            KeyEq: BinaryPredicateOp<Key>,
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
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn exclusive_scan_by_key_input(
                self,
                keys: &DeviceVec<Self::Runtime, Key>,
                init: Self::Init,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                let ($first_field, $( $field ),+) = init;
                let $first_field =
                    super::device_expr_exclusive_scan_by_key::<$first, Key, KeyEq, Op>(
                        &self.$first_field,
                        keys,
                        $first_field,
                    )?;
                $(
                    let $field =
                        super::device_expr_exclusive_scan_by_key::<$rest, Key, KeyEq, Op>(
                            &self.$field,
                            keys,
                            $field,
                        )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_exclusive_scan_by_key_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_exclusive_scan_by_key_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_exclusive_scan_by_key_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_exclusive_scan_by_key_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_exclusive_scan_by_key_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_exclusive_scan_by_key_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_exclusive_scan_by_key_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_exclusive_scan_by_key_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_exclusive_scan_by_key_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Computes an exclusive scan by key.
pub fn exclusive_scan_by_key<Values, Keys, R, K, KeyEq, Op>(
    values: Values,
    keys: Keys,
    init: <Values as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Init,
    _key_eq: KeyEq,
    _op: Op,
) -> Result<<Values as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Output, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Keys: KeyInput<Runtime = R, Item = K>,
    Values: ExclusiveScanByKeyInput<K, KeyEq, Op, Runtime = R>,
{
    let keys = keys.key_input()?;
    values.exclusive_scan_by_key_input(&keys, init, GpuOp::<KeyEq>::new(), GpuOp::<Op>::new())
}
