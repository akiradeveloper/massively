use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoA4,
        SoA5, SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoAView1, SoAView2, SoAView3, SoAView4,
        SoAView5, SoAView6, SoAView7, SoAView8, SoAView9, SoAView10, SoAView11, SoAView12,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr, Input},
    op::{BinaryOp, BinaryPredicateOp, GpuOp},
    primitives::scan as primitive_scan,
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

impl<Source> KeyInput for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
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

/// Input accepted by [`inclusive_scan`].
#[doc(hidden)]
pub trait InclusiveScanInput<Op> {
    /// Scan output type.
    type Output;

    /// Computes an inclusive scan.
    fn inclusive_scan_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error>;
}

impl<Source, Op> InclusiveScanInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
    Input<Source::Item>: GpuExpr<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn inclusive_scan_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
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
        <SoAView1<Source> as InclusiveScanInput<Op>>::inclusive_scan_input(
            SoAView1 { source: self },
            op,
        )
    }
}

macro_rules! impl_inclusive_scan_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> InclusiveScanInput<Op> for $input<$first, $( $rest ),+>
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
                ReadOnlySoA::validate(&self)?;
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

impl_inclusive_scan_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_inclusive_scan_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_inclusive_scan_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_inclusive_scan_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_inclusive_scan_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_inclusive_scan_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_inclusive_scan_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_inclusive_scan_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_inclusive_scan_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_inclusive_scan_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_inclusive_scan_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_inclusive_scan_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> InclusiveScanInput<Op> for $input<$first, $( $rest ),+>
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
                SoA::validate(&self)?;
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

impl_inclusive_scan_soa_input!(SoA2 -> SoA2<A, B> { left, right });
impl_inclusive_scan_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_inclusive_scan_soa_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_inclusive_scan_soa_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_inclusive_scan_soa_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_inclusive_scan_soa_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_inclusive_scan_soa_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_inclusive_scan_soa_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_inclusive_scan_soa_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_inclusive_scan_soa_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_inclusive_scan_soa_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Computes an inclusive scan from read-only input into device storage.
pub fn inclusive_scan<InputSource, Op>(
    source: InputSource,
    _op: Op,
) -> Result<<<InputSource as InclusiveScanInput<Op>>::Output as MaterializeOutput>::Output, Error>
where
    InputSource: InclusiveScanInput<Op>,
    <InputSource as InclusiveScanInput<Op>>::Output: MaterializeOutput,
{
    materialize(source.inclusive_scan_input(GpuOp::<Op>::new())?)
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

impl<Source, Op> ExclusiveScanInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
    Input<Source::Item>: GpuExpr<Source::Item>,
{
    type Init = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn exclusive_scan_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
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
        <SoAView1<Source> as ExclusiveScanInput<Op>>::exclusive_scan_input(
            SoAView1 { source: self },
            init,
            op,
        )
    }
}

macro_rules! impl_exclusive_scan_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> ExclusiveScanInput<Op> for $input<$first, $( $rest ),+>
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
                ReadOnlySoA::validate(&self)?;
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

impl_exclusive_scan_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_exclusive_scan_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_exclusive_scan_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_exclusive_scan_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_exclusive_scan_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_exclusive_scan_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_exclusive_scan_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_exclusive_scan_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_exclusive_scan_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_exclusive_scan_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_exclusive_scan_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_exclusive_scan_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> ExclusiveScanInput<Op> for $input<$first, $( $rest ),+>
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
                SoA::validate(&self)?;
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

impl_exclusive_scan_soa_input!(SoA2 -> SoA2<A, B> { left, right });
impl_exclusive_scan_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_exclusive_scan_soa_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_exclusive_scan_soa_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_exclusive_scan_soa_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_exclusive_scan_soa_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_exclusive_scan_soa_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_exclusive_scan_soa_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_exclusive_scan_soa_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_exclusive_scan_soa_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_exclusive_scan_soa_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Computes an exclusive scan from read-only input into device storage.
pub fn exclusive_scan<InputSource, Op>(
    source: InputSource,
    init: <InputSource as ExclusiveScanInput<Op>>::Init,
    _op: Op,
) -> Result<<<InputSource as ExclusiveScanInput<Op>>::Output as MaterializeOutput>::Output, Error>
where
    InputSource: ExclusiveScanInput<Op>,
    <InputSource as ExclusiveScanInput<Op>>::Output: MaterializeOutput,
{
    materialize(source.exclusive_scan_input(init, GpuOp::<Op>::new())?)
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

impl<Source, Op> AdjacentDifferenceInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn adjacent_difference_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let source = super::device_expr_adjacent_difference::<Source, Op>(&self.source)?;
        Ok(SoA1 { source })
    }
}

macro_rules! impl_adjacent_difference_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> AdjacentDifferenceInput<Op> for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
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
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn adjacent_difference_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field =
                    super::device_expr_adjacent_difference::<$first, Op>(&self.$first_field)?;
                $(
                    let $field =
                        super::device_expr_adjacent_difference::<$rest, Op>(&self.$field)?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_adjacent_difference_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_adjacent_difference_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_adjacent_difference_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_adjacent_difference_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_adjacent_difference_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_adjacent_difference_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_adjacent_difference_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_adjacent_difference_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_adjacent_difference_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_adjacent_difference_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_adjacent_difference_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_adjacent_difference_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Op> AdjacentDifferenceInput<Op> for $input<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
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
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn adjacent_difference_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field =
                    super::device_expr_adjacent_difference::<$first, Op>(&self.$first_field)?;
                $(
                    let $field =
                        super::device_expr_adjacent_difference::<$rest, Op>(&self.$field)?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_adjacent_difference_soa_input!(SoA2 -> SoA2<A, B> { left, right });
impl_adjacent_difference_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_adjacent_difference_soa_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_adjacent_difference_soa_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_adjacent_difference_soa_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_adjacent_difference_soa_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_adjacent_difference_soa_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_adjacent_difference_soa_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_adjacent_difference_soa_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_adjacent_difference_soa_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_adjacent_difference_soa_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Computes adjacent differences into device storage.
pub fn adjacent_difference<Source, Op>(
    source: Source,
    _op: Op,
) -> Result<<<Source as AdjacentDifferenceInput<Op>>::Output as MaterializeOutput>::Output, Error>
where
    Source: AdjacentDifferenceInput<Op>,
    <Source as AdjacentDifferenceInput<Op>>::Output: MaterializeOutput,
{
    materialize(source.adjacent_difference_input(GpuOp::<Op>::new())?)
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

impl<Source, K, KeyEq, Op> InclusiveScanByKeyInput<K, KeyEq, Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
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
        ReadOnlySoA::validate(&self)?;
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
    SoAView1<Source>: InclusiveScanByKeyInput<K, KeyEq, Op>,
    K: CubePrimitive + CubeElement,
{
    type Runtime = <SoAView1<Source> as InclusiveScanByKeyInput<K, KeyEq, Op>>::Runtime;
    type Output = <SoAView1<Source> as InclusiveScanByKeyInput<K, KeyEq, Op>>::Output;

    fn inclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as InclusiveScanByKeyInput<K, KeyEq, Op>>::inclusive_scan_by_key_input(
            SoAView1 { source: self },
            keys,
            key_eq,
            op,
        )
    }
}

impl<Left, Right, K, KeyEq, Op> InclusiveScanByKeyInput<K, KeyEq, Op> for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
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
        ReadOnlySoA::validate(&self)?;
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
    for SoAView3<First, Second, Third>
where
    Self: ReadOnlySoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
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
        ReadOnlySoA::validate(&self)?;
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
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
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
                ReadOnlySoA::validate(&self)?;
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

impl_inclusive_scan_by_key_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_inclusive_scan_by_key_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_inclusive_scan_by_key_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_inclusive_scan_by_key_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_inclusive_scan_by_key_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_inclusive_scan_by_key_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_inclusive_scan_by_key_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_inclusive_scan_by_key_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_inclusive_scan_by_key_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_inclusive_scan_by_key_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Key, KeyEq, Op> InclusiveScanByKeyInput<Key, KeyEq, Op>
            for $input<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
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
                SoA::validate(&self)?;
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

impl_inclusive_scan_by_key_soa_input!(SoA2 -> SoA2<A, B> { left, right });
impl_inclusive_scan_by_key_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_inclusive_scan_by_key_soa_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_inclusive_scan_by_key_soa_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_inclusive_scan_by_key_soa_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_inclusive_scan_by_key_soa_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_inclusive_scan_by_key_soa_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_inclusive_scan_by_key_soa_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_inclusive_scan_by_key_soa_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_inclusive_scan_by_key_soa_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_inclusive_scan_by_key_soa_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

#[doc(hidden)]
pub trait InclusiveScanByKeyCall<Values, KeyEq, Op> {
    type Output;

    fn inclusive_scan_by_key_call(
        self,
        values: Values,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Values, Keys, KeyEq, Op> InclusiveScanByKeyCall<Values, KeyEq, Op> for Keys
where
    Keys: KeyInput,
    Keys::Item: CubePrimitive + CubeElement,
    Values: InclusiveScanByKeyInput<Keys::Item, KeyEq, Op, Runtime = Keys::Runtime>,
{
    type Output = <Values as InclusiveScanByKeyInput<Keys::Item, KeyEq, Op>>::Output;

    fn inclusive_scan_by_key_call(
        self,
        values: Values,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.key_input()?;
        values.inclusive_scan_by_key_input(&keys, GpuOp::<KeyEq>::new(), GpuOp::<Op>::new())
    }
}

impl<ValueSource, KeyA, KeyB, KeyEq, Op> InclusiveScanByKeyCall<ValueSource, KeyEq, Op>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueSource::Item>,
{
    type Output = SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>;

    fn inclusive_scan_by_key_call(
        self,
        values: ValueSource,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let values = super::device_expr_collect(&values.source)?;
        Ok(SoA1 {
            source: primitive_scan::inclusive_scan_tuple2_by_key_device_vec(
                &key_a,
                &key_b,
                &values,
                GpuOp::<KeyEq>::new(),
                GpuOp::<Op>::new(),
            )?,
        })
    }
}

impl<ValueA, ValueB, KeyA, KeyB, KeyEq, Op>
    InclusiveScanByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op> for SoAView2<KeyA, KeyB>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
{
    type Output =
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>;

    fn inclusive_scan_by_key_call(
        self,
        values: SoAView2<ValueA, ValueB>,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let left = primitive_scan::inclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_a,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let right = primitive_scan::inclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_b,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA2 { left, right })
    }
}

impl<ValueA, ValueB, ValueC, KeyA, KeyB, KeyEq, Op>
    InclusiveScanByKeyCall<SoAView3<ValueA, ValueB, ValueC>, KeyEq, Op> for SoAView2<KeyA, KeyB>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
    Op: BinaryOp<ValueC::Item>,
{
    type Output = SoA3<
        DeviceVec<KeyA::Runtime, ValueA::Item>,
        DeviceVec<KeyA::Runtime, ValueB::Item>,
        DeviceVec<KeyA::Runtime, ValueC::Item>,
    >;

    fn inclusive_scan_by_key_call(
        self,
        values: SoAView3<ValueA, ValueB, ValueC>,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        let first = primitive_scan::inclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_a,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let second = primitive_scan::inclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_b,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let third = primitive_scan::inclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_c,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

impl<ValueSource, KeyA, KeyB, KeyC, KeyEq, Op> InclusiveScanByKeyCall<ValueSource, KeyEq, Op>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueSource::Item>,
{
    type Output = SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>;

    fn inclusive_scan_by_key_call(
        self,
        values: ValueSource,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let values = super::device_expr_collect(&values.source)?;
        Ok(SoA1 {
            source: primitive_scan::inclusive_scan_tuple3_by_key_device_vec(
                &key_a,
                &key_b,
                &key_c,
                &values,
                GpuOp::<KeyEq>::new(),
                GpuOp::<Op>::new(),
            )?,
        })
    }
}

impl<ValueA, ValueB, KeyA, KeyB, KeyC, KeyEq, Op>
    InclusiveScanByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op> for SoAView3<KeyA, KeyB, KeyC>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
{
    type Output =
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>;

    fn inclusive_scan_by_key_call(
        self,
        values: SoAView2<ValueA, ValueB>,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let left = primitive_scan::inclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_a,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let right = primitive_scan::inclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_b,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA2 { left, right })
    }
}

impl<ValueA, ValueB, ValueC, KeyA, KeyB, KeyC, KeyEq, Op>
    InclusiveScanByKeyCall<SoAView3<ValueA, ValueB, ValueC>, KeyEq, Op>
    for SoAView3<KeyA, KeyB, KeyC>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
    Op: BinaryOp<ValueC::Item>,
{
    type Output = SoA3<
        DeviceVec<KeyA::Runtime, ValueA::Item>,
        DeviceVec<KeyA::Runtime, ValueB::Item>,
        DeviceVec<KeyA::Runtime, ValueC::Item>,
    >;

    fn inclusive_scan_by_key_call(
        self,
        values: SoAView3<ValueA, ValueB, ValueC>,
        _key_eq: GpuOp<KeyEq>,
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
        let first = primitive_scan::inclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_a,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let second = primitive_scan::inclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_b,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let third = primitive_scan::inclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_c,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

macro_rules! impl_inclusive_scan_by_tuple_key_scalar_value {
    (
        $keys:ident,
        $scan_fn:ident,
        ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )
    ) => {
        impl<ValueSource, $first, $( $key ),+, KeyEq, Op>
            InclusiveScanByKeyCall<ValueSource, KeyEq, Op> for $keys<$first, $( $key ),+>
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
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<ValueSource::Item>,
        {
            type Output = SoA1<DeviceVec<<$first as KernelColumn>::Runtime, ValueSource::Item>>;

            fn inclusive_scan_by_key_call(
                self,
                values: ValueSource,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                values.validate()?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let values = super::device_expr_collect(&values)?;
                Ok(SoA1 {
                    source: primitive_scan::$scan_fn(
                        &$first_field,
                        $( &$field, )+
                        &values,
                        GpuOp::<KeyEq>::new(),
                        GpuOp::<Op>::new(),
                    )?,
                })
            }
        }
    };
}

impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView4, inclusive_scan_tuple4_by_key_device_vec, (A: a, B: b, C: c, D: d));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView5, inclusive_scan_tuple5_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView6, inclusive_scan_tuple6_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView7, inclusive_scan_tuple7_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView8, inclusive_scan_tuple8_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView9, inclusive_scan_tuple9_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView10, inclusive_scan_tuple10_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView11, inclusive_scan_tuple11_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoAView12, inclusive_scan_tuple12_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA2, inclusive_scan_tuple2_by_key_device_vec, (A: left, B: right));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA3, inclusive_scan_tuple3_by_key_device_vec, (A: first, B: second, C: third));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA4, inclusive_scan_tuple4_by_key_device_vec, (A: a, B: b, C: c, D: d));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA5, inclusive_scan_tuple5_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA6, inclusive_scan_tuple6_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA7, inclusive_scan_tuple7_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA8, inclusive_scan_tuple8_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA9, inclusive_scan_tuple9_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA10, inclusive_scan_tuple10_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA11, inclusive_scan_tuple11_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA12, inclusive_scan_tuple12_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l));

macro_rules! impl_inclusive_scan_by_tuple_key_soa_view2_values {
    (
        $keys:ident,
        $scan_fn:ident,
        ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )
    ) => {
        impl<ValueA, ValueB, $first, $( $key ),+, KeyEq, Op>
            InclusiveScanByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op>
            for $keys<$first, $( $key ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            SoAView2<ValueA, ValueB>: ReadOnlySoA<Item = (ValueA::Item, ValueB::Item), Scalar = ValueA::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueA: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            ValueB: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueA::Item: CubePrimitive + CubeElement,
            ValueB::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
            ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<ValueA::Item>,
            Op: BinaryOp<ValueB::Item>,
        {
            type Output = SoA2<
                DeviceVec<<$first as KernelColumn>::Runtime, ValueA::Item>,
                DeviceVec<<$first as KernelColumn>::Runtime, ValueB::Item>,
            >;

            fn inclusive_scan_by_key_call(
                self,
                values: SoAView2<ValueA, ValueB>,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let value_a = super::device_expr_collect(&values.left)?;
                let value_b = super::device_expr_collect(&values.right)?;
                let left = primitive_scan::$scan_fn(
                    &$first_field,
                    $( &$field, )+
                    &value_a,
                    GpuOp::<KeyEq>::new(),
                    GpuOp::<Op>::new(),
                )?;
                let right = primitive_scan::$scan_fn(
                    &$first_field,
                    $( &$field, )+
                    &value_b,
                    GpuOp::<KeyEq>::new(),
                    GpuOp::<Op>::new(),
                )?;
                Ok(SoA2 { left, right })
            }
        }
    };
}

impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView4, inclusive_scan_tuple4_by_key_device_vec, (A: a, B: b, C: c, D: d));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView5, inclusive_scan_tuple5_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView6, inclusive_scan_tuple6_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView7, inclusive_scan_tuple7_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView8, inclusive_scan_tuple8_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView9, inclusive_scan_tuple9_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView10, inclusive_scan_tuple10_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView11, inclusive_scan_tuple11_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k));
impl_inclusive_scan_by_tuple_key_soa_view2_values!(SoAView12, inclusive_scan_tuple12_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l));

macro_rules! impl_inclusive_scan_by_tuple_key_soa_view3_values {
    (
        $keys:ident,
        $scan_fn:ident,
        ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )
    ) => {
        impl<ValueA, ValueB, ValueC, $first, $( $key ),+, KeyEq, Op>
            InclusiveScanByKeyCall<SoAView3<ValueA, ValueB, ValueC>, KeyEq, Op>
            for $keys<$first, $( $key ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            SoAView3<ValueA, ValueB, ValueC>:
                ReadOnlySoA<Item = (ValueA::Item, ValueB::Item, ValueC::Item), Scalar = ValueA::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueA: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            ValueB: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            ValueC: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueA::Item: CubePrimitive + CubeElement,
            ValueB::Item: CubePrimitive + CubeElement,
            ValueC::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
            ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
            ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<ValueA::Item>,
            Op: BinaryOp<ValueB::Item>,
            Op: BinaryOp<ValueC::Item>,
        {
            type Output = SoA3<
                DeviceVec<<$first as KernelColumn>::Runtime, ValueA::Item>,
                DeviceVec<<$first as KernelColumn>::Runtime, ValueB::Item>,
                DeviceVec<<$first as KernelColumn>::Runtime, ValueC::Item>,
            >;

            fn inclusive_scan_by_key_call(
                self,
                values: SoAView3<ValueA, ValueB, ValueC>,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let value_a = super::device_expr_collect(&values.first)?;
                let value_b = super::device_expr_collect(&values.second)?;
                let value_c = super::device_expr_collect(&values.third)?;
                let first = primitive_scan::$scan_fn(
                    &$first_field,
                    $( &$field, )+
                    &value_a,
                    GpuOp::<KeyEq>::new(),
                    GpuOp::<Op>::new(),
                )?;
                let second = primitive_scan::$scan_fn(
                    &$first_field,
                    $( &$field, )+
                    &value_b,
                    GpuOp::<KeyEq>::new(),
                    GpuOp::<Op>::new(),
                )?;
                let third = primitive_scan::$scan_fn(
                    &$first_field,
                    $( &$field, )+
                    &value_c,
                    GpuOp::<KeyEq>::new(),
                    GpuOp::<Op>::new(),
                )?;
                Ok(SoA3 { first, second, third })
            }
        }
    };
}

impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView4, inclusive_scan_tuple4_by_key_device_vec, (A: a, B: b, C: c, D: d));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView5, inclusive_scan_tuple5_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView6, inclusive_scan_tuple6_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView7, inclusive_scan_tuple7_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView8, inclusive_scan_tuple8_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView9, inclusive_scan_tuple9_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView10, inclusive_scan_tuple10_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView11, inclusive_scan_tuple11_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k));
impl_inclusive_scan_by_tuple_key_soa_view3_values!(SoAView12, inclusive_scan_tuple12_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l));

macro_rules! impl_inclusive_scan_by_tuple_key_soa_view_values {
    (@scan $scan_fn:ident, ($first_field:ident, $( $field:ident ),+), $value_field:ident) => {
        primitive_scan::$scan_fn(
            &$first_field,
            $( &$field, )+
            &$value_field,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )
    };
    (@scan_values $scan_fn:ident, ($first_field:ident, $( $field:ident ),+), ) => {};
    (@scan_values $scan_fn:ident, ($first_field:ident, $( $field:ident ),+), $value_field:ident $(, $tail:ident )*) => {
        let $value_field = impl_inclusive_scan_by_tuple_key_soa_view_values!(
            @scan $scan_fn,
            ($first_field, $( $field ),+),
            $value_field
        )?;
        impl_inclusive_scan_by_tuple_key_soa_view_values!(
            @scan_values $scan_fn,
            ($first_field, $( $field ),+),
            $( $tail ),*
        );
    };

    (
        $key_storage:ident,
        $storage:ident,
        $values:ident -> $output:ident < $first_value:ident, $( $value:ident ),+ > { $first_value_field:ident, $( $value_field:ident ),+ },
        $keys:ident,
        $scan_fn:ident,
        ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )
    ) => {
        impl<$first_value, $( $value ),+, $first, $( $key ),+, KeyEq, Op>
            InclusiveScanByKeyCall<$values<$first_value, $( $value ),+>, KeyEq, Op>
            for $keys<$first, $( $key ),+>
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
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<<$first_value as KernelColumn>::Item>,
            $( Op: BinaryOp<<$value as KernelColumn>::Item>, )+
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first_value as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+
            >;

            fn inclusive_scan_by_key_call(
                self,
                values: $values<$first_value, $( $value ),+>,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                $key_storage::validate(&self)?;
                $storage::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let $first_value_field = super::device_expr_collect(&values.$first_value_field)?;
                $(
                    let $value_field = super::device_expr_collect(&values.$value_field)?;
                )+
                let $first_value_field = impl_inclusive_scan_by_tuple_key_soa_view_values!(
                    @scan $scan_fn,
                    ($first_field, $( $field ),+),
                    $first_value_field
                )?;
                impl_inclusive_scan_by_tuple_key_soa_view_values!(
                    @scan_values $scan_fn,
                    ($first_field, $( $field ),+),
                    $( $value_field ),+
                );
                Ok($output { $first_value_field, $( $value_field ),+ })
            }
        }
    };
}

macro_rules! impl_inclusive_scan_by_tuple_key_soa_view_values_for_key {
    ($key_storage:ident, $keys:ident, $scan_fn:ident, ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )) => {
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView4 -> SoA4<A, B, C, D> { a, b, c, d }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA2 -> SoA2<A, B> { left, right }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA3 -> SoA3<A, B, C> { first, second, third }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA4 -> SoA4<A, B, C, D> { a, b, c, d }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
    };
}

impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView2, inclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView3, inclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView4, inclusive_scan_tuple4_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView5, inclusive_scan_tuple5_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView6, inclusive_scan_tuple6_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView7, inclusive_scan_tuple7_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView8, inclusive_scan_tuple8_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView9, inclusive_scan_tuple9_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView10, inclusive_scan_tuple10_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView11, inclusive_scan_tuple11_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView12, inclusive_scan_tuple12_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k, KL: l));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA2, inclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA3, inclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA4, inclusive_scan_tuple4_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA5, inclusive_scan_tuple5_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA6, inclusive_scan_tuple6_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA7, inclusive_scan_tuple7_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA8, inclusive_scan_tuple8_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA9, inclusive_scan_tuple9_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA10, inclusive_scan_tuple10_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA11, inclusive_scan_tuple11_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA12, inclusive_scan_tuple12_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k, KL: l));

/// Computes an inclusive scan by key.
pub fn inclusive_scan_by_key<Keys, Values, KeyEq, Op>(
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    _op: Op,
) -> Result<
    <<Keys as InclusiveScanByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Keys: InclusiveScanByKeyCall<Values, KeyEq, Op>,
    <Keys as InclusiveScanByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput,
{
    materialize(keys.inclusive_scan_by_key_call(
        values,
        GpuOp::<KeyEq>::new(),
        GpuOp::<Op>::new(),
    )?)
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

impl<Source, K, KeyEq, Op> ExclusiveScanByKeyInput<K, KeyEq, Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
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
        ReadOnlySoA::validate(&self)?;
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
    SoAView1<Source>: ExclusiveScanByKeyInput<K, KeyEq, Op>,
    K: CubePrimitive + CubeElement,
{
    type Runtime = <SoAView1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Runtime;
    type Init = <SoAView1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Init;
    type Output = <SoAView1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::Output;

    fn exclusive_scan_by_key_input(
        self,
        keys: &DeviceVec<Self::Runtime, K>,
        init: Self::Init,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as ExclusiveScanByKeyInput<K, KeyEq, Op>>::exclusive_scan_by_key_input(
            SoAView1 { source: self },
            keys,
            init,
            key_eq,
            op,
        )
    }
}

impl<Left, Right, K, KeyEq, Op> ExclusiveScanByKeyInput<K, KeyEq, Op> for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
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
        ReadOnlySoA::validate(&self)?;
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
    for SoAView3<First, Second, Third>
where
    Self: ReadOnlySoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
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
        ReadOnlySoA::validate(&self)?;
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
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
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
                ReadOnlySoA::validate(&self)?;
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

impl_exclusive_scan_by_key_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_exclusive_scan_by_key_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_exclusive_scan_by_key_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_exclusive_scan_by_key_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_exclusive_scan_by_key_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_exclusive_scan_by_key_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_exclusive_scan_by_key_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_exclusive_scan_by_key_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_exclusive_scan_by_key_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_exclusive_scan_by_key_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Key, KeyEq, Op> ExclusiveScanByKeyInput<Key, KeyEq, Op>
            for $input<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
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
                SoA::validate(&self)?;
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

impl_exclusive_scan_by_key_soa_input!(SoA2 -> SoA2<A, B> { left, right });
impl_exclusive_scan_by_key_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_exclusive_scan_by_key_soa_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_exclusive_scan_by_key_soa_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_exclusive_scan_by_key_soa_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_exclusive_scan_by_key_soa_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_exclusive_scan_by_key_soa_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_exclusive_scan_by_key_soa_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_exclusive_scan_by_key_soa_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_exclusive_scan_by_key_soa_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_exclusive_scan_by_key_soa_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

#[doc(hidden)]
pub trait ExclusiveScanByKeyCall<Values, KeyEq, Op> {
    type Init;
    type Output;

    fn exclusive_scan_by_key_call(
        self,
        values: Values,
        init: Self::Init,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Values, Keys, KeyEq, Op> ExclusiveScanByKeyCall<Values, KeyEq, Op> for Keys
where
    Keys: KeyInput,
    Keys::Item: CubePrimitive + CubeElement,
    Values: ExclusiveScanByKeyInput<Keys::Item, KeyEq, Op, Runtime = Keys::Runtime>,
{
    type Init = <Values as ExclusiveScanByKeyInput<Keys::Item, KeyEq, Op>>::Init;
    type Output = <Values as ExclusiveScanByKeyInput<Keys::Item, KeyEq, Op>>::Output;

    fn exclusive_scan_by_key_call(
        self,
        values: Values,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.key_input()?;
        values.exclusive_scan_by_key_input(&keys, init, GpuOp::<KeyEq>::new(), GpuOp::<Op>::new())
    }
}

impl<ValueSource, KeyA, KeyB, KeyEq, Op> ExclusiveScanByKeyCall<ValueSource, KeyEq, Op>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueSource::Item>,
{
    type Init = ValueSource::Item;
    type Output = SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>;

    fn exclusive_scan_by_key_call(
        self,
        values: ValueSource,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let values = super::device_expr_collect(&values.source)?;
        Ok(SoA1 {
            source: primitive_scan::exclusive_scan_tuple2_by_key_device_vec(
                &key_a,
                &key_b,
                &values,
                init,
                GpuOp::<KeyEq>::new(),
                GpuOp::<Op>::new(),
            )?,
        })
    }
}

impl<ValueA, ValueB, KeyA, KeyB, KeyEq, Op>
    ExclusiveScanByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op> for SoAView2<KeyA, KeyB>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
{
    type Init = (ValueA::Item, ValueB::Item);
    type Output =
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>;

    fn exclusive_scan_by_key_call(
        self,
        values: SoAView2<ValueA, ValueB>,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let left = primitive_scan::exclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_a,
            init.0,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let right = primitive_scan::exclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_b,
            init.1,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA2 { left, right })
    }
}

impl<ValueA, ValueB, ValueC, KeyA, KeyB, KeyEq, Op>
    ExclusiveScanByKeyCall<SoAView3<ValueA, ValueB, ValueC>, KeyEq, Op> for SoAView2<KeyA, KeyB>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
    Op: BinaryOp<ValueC::Item>,
{
    type Init = (ValueA::Item, ValueB::Item, ValueC::Item);
    type Output = SoA3<
        DeviceVec<KeyA::Runtime, ValueA::Item>,
        DeviceVec<KeyA::Runtime, ValueB::Item>,
        DeviceVec<KeyA::Runtime, ValueC::Item>,
    >;

    fn exclusive_scan_by_key_call(
        self,
        values: SoAView3<ValueA, ValueB, ValueC>,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        let first = primitive_scan::exclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_a,
            init.0,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let second = primitive_scan::exclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_b,
            init.1,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let third = primitive_scan::exclusive_scan_tuple2_by_key_device_vec(
            &key_a,
            &key_b,
            &value_c,
            init.2,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

impl<ValueSource, KeyA, KeyB, KeyC, KeyEq, Op> ExclusiveScanByKeyCall<ValueSource, KeyEq, Op>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueSource::Item>,
{
    type Init = ValueSource::Item;
    type Output = SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>;

    fn exclusive_scan_by_key_call(
        self,
        values: ValueSource,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let values = super::device_expr_collect(&values.source)?;
        Ok(SoA1 {
            source: primitive_scan::exclusive_scan_tuple3_by_key_device_vec(
                &key_a,
                &key_b,
                &key_c,
                &values,
                init,
                GpuOp::<KeyEq>::new(),
                GpuOp::<Op>::new(),
            )?,
        })
    }
}

impl<ValueA, ValueB, KeyA, KeyB, KeyC, KeyEq, Op>
    ExclusiveScanByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op> for SoAView3<KeyA, KeyB, KeyC>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
{
    type Init = (ValueA::Item, ValueB::Item);
    type Output =
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>;

    fn exclusive_scan_by_key_call(
        self,
        values: SoAView2<ValueA, ValueB>,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let left = primitive_scan::exclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_a,
            init.0,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let right = primitive_scan::exclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_b,
            init.1,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA2 { left, right })
    }
}

impl<ValueA, ValueB, ValueC, KeyA, KeyB, KeyC, KeyEq, Op>
    ExclusiveScanByKeyCall<SoAView3<ValueA, ValueB, ValueC>, KeyEq, Op>
    for SoAView3<KeyA, KeyB, KeyC>
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
    KeyEq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
    Op: BinaryOp<ValueA::Item>,
    Op: BinaryOp<ValueB::Item>,
    Op: BinaryOp<ValueC::Item>,
{
    type Init = (ValueA::Item, ValueB::Item, ValueC::Item);
    type Output = SoA3<
        DeviceVec<KeyA::Runtime, ValueA::Item>,
        DeviceVec<KeyA::Runtime, ValueB::Item>,
        DeviceVec<KeyA::Runtime, ValueC::Item>,
    >;

    fn exclusive_scan_by_key_call(
        self,
        values: SoAView3<ValueA, ValueB, ValueC>,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
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
        let first = primitive_scan::exclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_a,
            init.0,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let second = primitive_scan::exclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_b,
            init.1,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        let third = primitive_scan::exclusive_scan_tuple3_by_key_device_vec(
            &key_a,
            &key_b,
            &key_c,
            &value_c,
            init.2,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

macro_rules! impl_exclusive_scan_by_tuple_key_scalar_value {
    (
        $keys:ident,
        $scan_fn:ident,
        ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )
    ) => {
        impl<ValueSource, $first, $( $key ),+, KeyEq, Op>
            ExclusiveScanByKeyCall<ValueSource, KeyEq, Op> for $keys<$first, $( $key ),+>
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
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<ValueSource::Item>,
        {
            type Init = ValueSource::Item;
            type Output = SoA1<DeviceVec<<$first as KernelColumn>::Runtime, ValueSource::Item>>;

            fn exclusive_scan_by_key_call(
                self,
                values: ValueSource,
                init: Self::Init,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                values.validate()?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let values = super::device_expr_collect(&values)?;
                Ok(SoA1 {
                    source: primitive_scan::$scan_fn(
                        &$first_field,
                        $( &$field, )+
                        &values,
                        init,
                        GpuOp::<KeyEq>::new(),
                        GpuOp::<Op>::new(),
                    )?,
                })
            }
        }
    };
}

impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView4, exclusive_scan_tuple4_by_key_device_vec, (A: a, B: b, C: c, D: d));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView5, exclusive_scan_tuple5_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView6, exclusive_scan_tuple6_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView7, exclusive_scan_tuple7_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView8, exclusive_scan_tuple8_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView9, exclusive_scan_tuple9_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView10, exclusive_scan_tuple10_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView11, exclusive_scan_tuple11_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoAView12, exclusive_scan_tuple12_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA2, exclusive_scan_tuple2_by_key_device_vec, (A: left, B: right));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA3, exclusive_scan_tuple3_by_key_device_vec, (A: first, B: second, C: third));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA4, exclusive_scan_tuple4_by_key_device_vec, (A: a, B: b, C: c, D: d));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA5, exclusive_scan_tuple5_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA6, exclusive_scan_tuple6_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA7, exclusive_scan_tuple7_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA8, exclusive_scan_tuple8_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA9, exclusive_scan_tuple9_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA10, exclusive_scan_tuple10_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA11, exclusive_scan_tuple11_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA12, exclusive_scan_tuple12_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l));

macro_rules! impl_exclusive_scan_by_tuple_key_soa_view2_values {
    (
        $keys:ident,
        $scan_fn:ident,
        ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )
    ) => {
        impl<ValueA, ValueB, $first, $( $key ),+, KeyEq, Op>
            ExclusiveScanByKeyCall<SoAView2<ValueA, ValueB>, KeyEq, Op>
            for $keys<$first, $( $key ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            SoAView2<ValueA, ValueB>: ReadOnlySoA<Item = (ValueA::Item, ValueB::Item), Scalar = ValueA::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueA: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            ValueB: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueA::Item: CubePrimitive + CubeElement,
            ValueB::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
            ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<ValueA::Item>,
            Op: BinaryOp<ValueB::Item>,
        {
            type Init = (ValueA::Item, ValueB::Item);
            type Output = SoA2<
                DeviceVec<<$first as KernelColumn>::Runtime, ValueA::Item>,
                DeviceVec<<$first as KernelColumn>::Runtime, ValueB::Item>,
            >;

            fn exclusive_scan_by_key_call(
                self,
                values: SoAView2<ValueA, ValueB>,
                init: Self::Init,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let value_a = super::device_expr_collect(&values.left)?;
                let value_b = super::device_expr_collect(&values.right)?;
                let left = primitive_scan::$scan_fn(
                    &$first_field,
                    $( &$field, )+
                    &value_a,
                    init.0,
                    GpuOp::<KeyEq>::new(),
                    GpuOp::<Op>::new(),
                )?;
                let right = primitive_scan::$scan_fn(
                    &$first_field,
                    $( &$field, )+
                    &value_b,
                    init.1,
                    GpuOp::<KeyEq>::new(),
                    GpuOp::<Op>::new(),
                )?;
                Ok(SoA2 { left, right })
            }
        }
    };
}

impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView4, exclusive_scan_tuple4_by_key_device_vec, (A: a, B: b, C: c, D: d));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView5, exclusive_scan_tuple5_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView6, exclusive_scan_tuple6_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView7, exclusive_scan_tuple7_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView8, exclusive_scan_tuple8_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView9, exclusive_scan_tuple9_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView10, exclusive_scan_tuple10_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView11, exclusive_scan_tuple11_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k));
impl_exclusive_scan_by_tuple_key_soa_view2_values!(SoAView12, exclusive_scan_tuple12_by_key_device_vec, (A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l));

macro_rules! impl_exclusive_scan_by_tuple_key_soa_view_values {
    (@scan $scan_fn:ident, ($first_field:ident, $( $field:ident ),+), $value_field:ident, $init:expr) => {
        primitive_scan::$scan_fn(
            &$first_field,
            $( &$field, )+
            &$value_field,
            $init,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )
    };
    (@scan_values $scan_fn:ident, ($first_field:ident, $( $field:ident ),+), $init:ident, ) => {};
    (@scan_values $scan_fn:ident, ($first_field:ident, $( $field:ident ),+), $init:ident, $value_field:ident: $idx:tt $(, $tail_field:ident: $tail_idx:tt )*) => {
        let $value_field = impl_exclusive_scan_by_tuple_key_soa_view_values!(
            @scan $scan_fn,
            ($first_field, $( $field ),+),
            $value_field,
            $init.$idx
        )?;
        impl_exclusive_scan_by_tuple_key_soa_view_values!(
            @scan_values $scan_fn,
            ($first_field, $( $field ),+),
            $init,
            $( $tail_field: $tail_idx ),*
        );
    };

    (
        $key_storage:ident,
        $storage:ident,
        $values:ident -> $output:ident < $first_value:ident: $first_idx:tt, $( $value:ident: $idx:tt ),+ > { $first_value_field:ident, $( $value_field:ident ),+ },
        $keys:ident,
        $scan_fn:ident,
        ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )
    ) => {
        impl<$first_value, $( $value ),+, $first, $( $key ),+, KeyEq, Op>
            ExclusiveScanByKeyCall<$values<$first_value, $( $value ),+>, KeyEq, Op>
            for $keys<$first, $( $key ),+>
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
            KeyEq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
            Op: BinaryOp<<$first_value as KernelColumn>::Item>,
            $( Op: BinaryOp<<$value as KernelColumn>::Item>, )+
        {
            type Init = (<$first_value as KernelColumn>::Item, $( <$value as KernelColumn>::Item ),+);
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first_value as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+
            >;

            fn exclusive_scan_by_key_call(
                self,
                values: $values<$first_value, $( $value ),+>,
                init: Self::Init,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                $key_storage::validate(&self)?;
                $storage::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let $first_value_field = super::device_expr_collect(&values.$first_value_field)?;
                $( let $value_field = super::device_expr_collect(&values.$value_field)?; )+
                let $first_value_field = impl_exclusive_scan_by_tuple_key_soa_view_values!(
                    @scan $scan_fn,
                    ($first_field, $( $field ),+),
                    $first_value_field,
                    init.$first_idx
                )?;
                impl_exclusive_scan_by_tuple_key_soa_view_values!(
                    @scan_values $scan_fn,
                    ($first_field, $( $field ),+),
                    init,
                    $( $value_field: $idx ),+
                );
                Ok($output { $first_value_field, $( $value_field ),+ })
            }
        }
    };
}

macro_rules! impl_exclusive_scan_by_tuple_key_soa_view_values_for_key {
    ($key_storage:ident, $keys:ident, $scan_fn:ident, ( $first:ident: $first_field:ident, $( $key:ident: $field:ident ),+ )) => {
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView4 -> SoA4<A: 0, B: 1, C: 2, D: 3> { a, b, c, d }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView5 -> SoA5<A: 0, B: 1, C: 2, D: 3, E: 4> { a, b, c, d, e }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView6 -> SoA6<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5> { a, b, c, d, e, f }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView7 -> SoA7<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6> { a, b, c, d, e, f, g }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView8 -> SoA8<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7> { a, b, c, d, e, f, g, h }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView9 -> SoA9<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8> { a, b, c, d, e, f, g, h, i }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView10 -> SoA10<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9> { a, b, c, d, e, f, g, h, i, j }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView11 -> SoA11<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10> { a, b, c, d, e, f, g, h, i, j, k }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, ReadOnlySoA, SoAView12 -> SoA12<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11> { a, b, c, d, e, f, g, h, i, j, k, l }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA2 -> SoA2<A: 0, B: 1> { left, right }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA3 -> SoA3<A: 0, B: 1, C: 2> { first, second, third }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA4 -> SoA4<A: 0, B: 1, C: 2, D: 3> { a, b, c, d }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA5 -> SoA5<A: 0, B: 1, C: 2, D: 3, E: 4> { a, b, c, d, e }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA6 -> SoA6<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5> { a, b, c, d, e, f }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA7 -> SoA7<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6> { a, b, c, d, e, f, g }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA8 -> SoA8<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7> { a, b, c, d, e, f, g, h }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA9 -> SoA9<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8> { a, b, c, d, e, f, g, h, i }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA10 -> SoA10<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9> { a, b, c, d, e, f, g, h, i, j }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA11 -> SoA11<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10> { a, b, c, d, e, f, g, h, i, j, k }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA12 -> SoA12<A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11> { a, b, c, d, e, f, g, h, i, j, k, l }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
    };
}

impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView2, exclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView3, exclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView4, exclusive_scan_tuple4_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView5, exclusive_scan_tuple5_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView6, exclusive_scan_tuple6_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView7, exclusive_scan_tuple7_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView8, exclusive_scan_tuple8_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView9, exclusive_scan_tuple9_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView10, exclusive_scan_tuple10_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView11, exclusive_scan_tuple11_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView12, exclusive_scan_tuple12_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k, KL: l));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA2, exclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA3, exclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA4, exclusive_scan_tuple4_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA5, exclusive_scan_tuple5_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA6, exclusive_scan_tuple6_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA7, exclusive_scan_tuple7_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA8, exclusive_scan_tuple8_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA9, exclusive_scan_tuple9_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA10, exclusive_scan_tuple10_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA11, exclusive_scan_tuple11_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA12, exclusive_scan_tuple12_by_key_device_vec, (KA: a, KB: b, KC: c, KD: d, KE: e, KF: f, KG: g, KH: h, KI: i, KJ: j, KK: k, KL: l));

/// Computes an exclusive scan by key.
pub fn exclusive_scan_by_key<Keys, Values, KeyEq, Op>(
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    init: <Keys as ExclusiveScanByKeyCall<Values, KeyEq, Op>>::Init,
    _op: Op,
) -> Result<
    <<Keys as ExclusiveScanByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Keys: ExclusiveScanByKeyCall<Values, KeyEq, Op>,
    <Keys as ExclusiveScanByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput,
{
    materialize(keys.exclusive_scan_by_key_call(
        values,
        init,
        GpuOp::<KeyEq>::new(),
        GpuOp::<Op>::new(),
    )?)
}
