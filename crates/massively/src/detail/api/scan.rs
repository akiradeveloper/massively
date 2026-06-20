use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoAView1,
        SoAView2, SoAView3,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
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
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn inclusive_scan_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let bindings = self.source.stage()?;
        primitive_scan::inclusive_scan_tuple1_device_expr::<_, _, Source::Expr, Op>(
            self.source.policy(),
            &bindings,
            self.source.len(),
        )
    }
}

impl<Source, Op> InclusiveScanInput<Op> for (Source,)
where
    SoAView1<Source>: InclusiveScanInput<Op>,
{
    type Output = <SoAView1<Source> as InclusiveScanInput<Op>>::Output;

    fn inclusive_scan_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView1<Source> as InclusiveScanInput<Op>>::inclusive_scan_input(
            SoAView1 { source: self.0 },
            op,
        )
    }
}

macro_rules! impl_inclusive_scan_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $scan_fn:ident) => {
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
            Op: BinaryOp<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn inclusive_scan_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_scan::$scan_fn::<
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
                )
            }
        }
    };
}

impl_inclusive_scan_input!(SoAView2 -> SoA2<A, B> { left, right } => inclusive_scan_tuple2_device_expr);
impl_inclusive_scan_input!(SoAView3 -> SoA3<A, B, C> { first, second, third } => inclusive_scan_tuple3_device_expr);

impl<Left, Right, Op> InclusiveScanInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: InclusiveScanInput<Op>,
{
    type Output = <SoAView2<Left, Right> as InclusiveScanInput<Op>>::Output;

    fn inclusive_scan_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as InclusiveScanInput<Op>>::inclusive_scan_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            op,
        )
    }
}

impl<First, Second, Third, Op> InclusiveScanInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: InclusiveScanInput<Op>,
{
    type Output = <SoAView3<First, Second, Third> as InclusiveScanInput<Op>>::Output;

    fn inclusive_scan_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as InclusiveScanInput<Op>>::inclusive_scan_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            op,
        )
    }
}

macro_rules! impl_inclusive_scan_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $scan_fn:ident) => {
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
            Op: BinaryOp<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn inclusive_scan_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_scan::$scan_fn::<
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
                )
            }
        }
    };
}

impl_inclusive_scan_soa_input!(SoA2 -> SoA2<A, B> { left, right } => inclusive_scan_tuple2_device_expr);
impl_inclusive_scan_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third } => inclusive_scan_tuple3_device_expr);

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
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Init = (Source::Item,);
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn exclusive_scan_input(self, init: Self::Init, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let bindings = self.source.stage()?;
        primitive_scan::exclusive_scan_tuple1_device_expr::<_, _, Source::Expr, Op>(
            self.source.policy(),
            &bindings,
            self.source.len(),
            init,
        )
    }
}

impl<Source, Op> ExclusiveScanInput<Op> for (Source,)
where
    SoAView1<Source>: ExclusiveScanInput<Op>,
{
    type Init = <SoAView1<Source> as ExclusiveScanInput<Op>>::Init;
    type Output = <SoAView1<Source> as ExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView1<Source> as ExclusiveScanInput<Op>>::exclusive_scan_input(
            SoAView1 { source: self.0 },
            init,
            op,
        )
    }
}

macro_rules! impl_exclusive_scan_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $scan_fn:ident) => {
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
            Op: BinaryOp<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
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
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_scan::$scan_fn::<
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

impl_exclusive_scan_input!(SoAView2 -> SoA2<A, B> { left, right } => exclusive_scan_tuple2_device_expr);
impl_exclusive_scan_input!(SoAView3 -> SoA3<A, B, C> { first, second, third } => exclusive_scan_tuple3_device_expr);

impl<Left, Right, Op> ExclusiveScanInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: ExclusiveScanInput<Op>,
{
    type Init = <SoAView2<Left, Right> as ExclusiveScanInput<Op>>::Init;
    type Output = <SoAView2<Left, Right> as ExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ExclusiveScanInput<Op>>::exclusive_scan_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            init,
            op,
        )
    }
}

impl<First, Second, Third, Op> ExclusiveScanInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ExclusiveScanInput<Op>,
{
    type Init = <SoAView3<First, Second, Third> as ExclusiveScanInput<Op>>::Init;
    type Output = <SoAView3<First, Second, Third> as ExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_input(self, init: Self::Init, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ExclusiveScanInput<Op>>::exclusive_scan_input(
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

macro_rules! impl_exclusive_scan_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $scan_fn:ident) => {
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
            Op: BinaryOp<(<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+)>,
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
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_scan::$scan_fn::<
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

impl_exclusive_scan_soa_input!(SoA2 -> SoA2<A, B> { left, right } => exclusive_scan_tuple2_device_expr);
impl_exclusive_scan_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third } => exclusive_scan_tuple3_device_expr);

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
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
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
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $scan_fn:ident) => {
        impl<$first, $( $rest ),+, Op> AdjacentDifferenceInput<Op> for $input<$first, $( $rest ),+>
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
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn adjacent_difference_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_scan::$scan_fn::<
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
                )
            }
        }
    };
}

impl_adjacent_difference_input!(SoAView2 -> SoA2<A, B> { left, right } => adjacent_difference_tuple2_device_expr);
impl_adjacent_difference_input!(SoAView3 -> SoA3<A, B, C> { first, second, third } => adjacent_difference_tuple3_device_expr);

impl<Source, Op> AdjacentDifferenceInput<Op> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn adjacent_difference_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <Source as AdjacentDifferenceInput<super::Tuple1BinaryOp<Op>>>::adjacent_difference_input(
            self.0,
            GpuOp::<super::Tuple1BinaryOp<Op>>::new(),
        )
    }
}

impl<Left, Right, Op> AdjacentDifferenceInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: AdjacentDifferenceInput<Op>,
{
    type Output = <SoAView2<Left, Right> as AdjacentDifferenceInput<Op>>::Output;

    fn adjacent_difference_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as AdjacentDifferenceInput<Op>>::adjacent_difference_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            op,
        )
    }
}

impl<First, Second, Third, Op> AdjacentDifferenceInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: AdjacentDifferenceInput<Op>,
{
    type Output = <SoAView3<First, Second, Third> as AdjacentDifferenceInput<Op>>::Output;

    fn adjacent_difference_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as AdjacentDifferenceInput<Op>>::adjacent_difference_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            op,
        )
    }
}

macro_rules! impl_adjacent_difference_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $scan_fn:ident) => {
        impl<$first, $( $rest ),+, Op> AdjacentDifferenceInput<Op> for $input<$first, $( $rest ),+>
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
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn adjacent_difference_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = self.$first_field.stage()?;
                $(
                    let $field = self.$field.stage()?;
                )+
                primitive_scan::$scan_fn::<
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
                )
            }
        }
    };
}

impl_adjacent_difference_soa_input!(SoA2 -> SoA2<A, B> { left, right } => adjacent_difference_tuple2_device_expr);
impl_adjacent_difference_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third } => adjacent_difference_tuple3_device_expr);

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
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    K: CubePrimitive + CubeElement,
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
    K: CubePrimitive + CubeElement,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(Left::Item, Right::Item)>,
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
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        primitive_scan::inclusive_scan_tuple2_by_key_values_device_vec(
            keys,
            &left,
            &right,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )
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
    K: CubePrimitive + CubeElement,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(First::Item, Second::Item, Third::Item)>,
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
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        primitive_scan::inclusive_scan_tuple3_by_key_values_device_vec(
            keys,
            &first,
            &second,
            &third,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )
    }
}

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
            Key: CubePrimitive + CubeElement,
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

macro_rules! impl_inclusive_scan_by_key_tuple_values {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Key, KeyEq, Op> InclusiveScanByKeyInput<Key, KeyEq, Op> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: InclusiveScanByKeyInput<Key, KeyEq, Op>,
        {
            type Runtime = <$view<$( $ty ),+> as InclusiveScanByKeyInput<Key, KeyEq, Op>>::Runtime;
            type Output = <$view<$( $ty ),+> as InclusiveScanByKeyInput<Key, KeyEq, Op>>::Output;

            fn inclusive_scan_by_key_input(
                self,
                keys: &DeviceVec<Self::Runtime, Key>,
                key_eq: GpuOp<KeyEq>,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as InclusiveScanByKeyInput<Key, KeyEq, Op>>::inclusive_scan_by_key_input(
                    $view { $( $field: self.$index ),+ },
                    keys,
                    key_eq,
                    op,
                )
            }
        }
    };
}

impl_inclusive_scan_by_key_tuple_values!(SoAView2<A, B> { left: 0, right: 1 });
impl_inclusive_scan_by_key_tuple_values!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

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

impl_inclusive_scan_by_tuple_key_scalar_value!(SoA2, inclusive_scan_tuple2_by_key_device_vec, (A: left, B: right));
impl_inclusive_scan_by_tuple_key_scalar_value!(SoA3, inclusive_scan_tuple3_by_key_device_vec, (A: first, B: second, C: third));

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
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA2 -> SoA2<A, B> { left, right }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_inclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA3 -> SoA3<A, B, C> { first, second, third }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
    };
}

impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView2, inclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView3, inclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA2, inclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_inclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA3, inclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));

macro_rules! impl_inclusive_scan_by_key_tuple_keys {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Values, KeyEq, Op> InclusiveScanByKeyCall<Values, KeyEq, Op> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: InclusiveScanByKeyCall<Values, KeyEq, Op>,
        {
            type Output = <$view<$( $ty ),+> as InclusiveScanByKeyCall<Values, KeyEq, Op>>::Output;

            fn inclusive_scan_by_key_call(
                self,
                values: Values,
                key_eq: GpuOp<KeyEq>,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as InclusiveScanByKeyCall<Values, KeyEq, Op>>::inclusive_scan_by_key_call(
                    $view { $( $field: self.$index ),+ },
                    values,
                    key_eq,
                    op,
                )
            }
        }
    };
}

impl_inclusive_scan_by_key_tuple_keys!(SoAView2<A, B> { left: 0, right: 1 });
impl_inclusive_scan_by_key_tuple_keys!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

impl<KeySource, ValueSource, KeyEq, Op> InclusiveScanByKeyCall<(ValueSource,), KeyEq, Op>
    for (KeySource,)
where
    KeySource: KeyInput,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource: InclusiveScanByKeyInput<
            KeySource::Item,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
            Runtime = KeySource::Runtime,
        >,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
{
    type Output = <ValueSource as InclusiveScanByKeyInput<
        KeySource::Item,
        super::Tuple1Less<KeyEq>,
        super::Tuple1BinaryOp<Op>,
    >>::Output;

    fn inclusive_scan_by_key_call(
        self,
        values: (ValueSource,),
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.0.key_input()?;
        values.0.inclusive_scan_by_key_input(
            &keys,
            GpuOp::<super::Tuple1Less<KeyEq>>::new(),
            GpuOp::<super::Tuple1BinaryOp<Op>>::new(),
        )
    }
}

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
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    K: CubePrimitive + CubeElement,
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
    K: CubePrimitive + CubeElement,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(Left::Item, Right::Item)>,
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
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        primitive_scan::exclusive_scan_tuple2_by_key_values_device_vec(
            keys,
            &left,
            &right,
            init,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )
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
    K: CubePrimitive + CubeElement,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(First::Item, Second::Item, Third::Item)>,
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
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        primitive_scan::exclusive_scan_tuple3_by_key_values_device_vec(
            keys,
            &first,
            &second,
            &third,
            init,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )
    }
}

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
            Key: CubePrimitive + CubeElement,
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

macro_rules! impl_exclusive_scan_by_key_tuple_values {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Key, KeyEq, Op> ExclusiveScanByKeyInput<Key, KeyEq, Op> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: ExclusiveScanByKeyInput<Key, KeyEq, Op>,
        {
            type Runtime = <$view<$( $ty ),+> as ExclusiveScanByKeyInput<Key, KeyEq, Op>>::Runtime;
            type Init = <$view<$( $ty ),+> as ExclusiveScanByKeyInput<Key, KeyEq, Op>>::Init;
            type Output = <$view<$( $ty ),+> as ExclusiveScanByKeyInput<Key, KeyEq, Op>>::Output;

            fn exclusive_scan_by_key_input(
                self,
                keys: &DeviceVec<Self::Runtime, Key>,
                init: Self::Init,
                key_eq: GpuOp<KeyEq>,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as ExclusiveScanByKeyInput<Key, KeyEq, Op>>::exclusive_scan_by_key_input(
                    $view { $( $field: self.$index ),+ },
                    keys,
                    init,
                    key_eq,
                    op,
                )
            }
        }
    };
}

impl_exclusive_scan_by_key_tuple_values!(SoAView2<A, B> { left: 0, right: 1 });
impl_exclusive_scan_by_key_tuple_values!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

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

impl_exclusive_scan_by_tuple_key_scalar_value!(SoA2, exclusive_scan_tuple2_by_key_device_vec, (A: left, B: right));
impl_exclusive_scan_by_tuple_key_scalar_value!(SoA3, exclusive_scan_tuple3_by_key_device_vec, (A: first, B: second, C: third));

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
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA2 -> SoA2<A: 0, B: 1> { left, right }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
        impl_exclusive_scan_by_tuple_key_soa_view_values!($key_storage, SoA, SoA3 -> SoA3<A: 0, B: 1, C: 2> { first, second, third }, $keys, $scan_fn, ( $first: $first_field, $( $key: $field ),+ ));
    };
}

impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView2, exclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(ReadOnlySoA, SoAView3, exclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA2, exclusive_scan_tuple2_by_key_device_vec, (KA: left, KB: right));
impl_exclusive_scan_by_tuple_key_soa_view_values_for_key!(SoA, SoA3, exclusive_scan_tuple3_by_key_device_vec, (KA: first, KB: second, KC: third));

macro_rules! impl_exclusive_scan_by_key_tuple_keys {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Values, KeyEq, Op> ExclusiveScanByKeyCall<Values, KeyEq, Op> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: ExclusiveScanByKeyCall<Values, KeyEq, Op>,
        {
            type Init = <$view<$( $ty ),+> as ExclusiveScanByKeyCall<Values, KeyEq, Op>>::Init;
            type Output = <$view<$( $ty ),+> as ExclusiveScanByKeyCall<Values, KeyEq, Op>>::Output;

            fn exclusive_scan_by_key_call(
                self,
                values: Values,
                init: Self::Init,
                key_eq: GpuOp<KeyEq>,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as ExclusiveScanByKeyCall<Values, KeyEq, Op>>::exclusive_scan_by_key_call(
                    $view { $( $field: self.$index ),+ },
                    values,
                    init,
                    key_eq,
                    op,
                )
            }
        }
    };
}

impl_exclusive_scan_by_key_tuple_keys!(SoAView2<A, B> { left: 0, right: 1 });
impl_exclusive_scan_by_key_tuple_keys!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

impl<KeySource, ValueSource, KeyEq, Op> ExclusiveScanByKeyCall<(ValueSource,), KeyEq, Op>
    for (KeySource,)
where
    KeySource: KeyInput,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource: ExclusiveScanByKeyInput<
            KeySource::Item,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
            Runtime = KeySource::Runtime,
        >,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
{
    type Init = (
        <ValueSource as ExclusiveScanByKeyInput<
            KeySource::Item,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
        >>::Init,
    );
    type Output = <ValueSource as ExclusiveScanByKeyInput<
        KeySource::Item,
        super::Tuple1Less<KeyEq>,
        super::Tuple1BinaryOp<Op>,
    >>::Output;

    fn exclusive_scan_by_key_call(
        self,
        values: (ValueSource,),
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.0.key_input()?;
        values.0.exclusive_scan_by_key_input(
            &keys,
            init.0,
            GpuOp::<super::Tuple1Less<KeyEq>>::new(),
            GpuOp::<super::Tuple1BinaryOp<Op>>::new(),
        )
    }
}

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
