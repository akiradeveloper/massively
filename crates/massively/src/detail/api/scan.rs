use super::memory::{MaterializeOutput, materialize};
use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp},
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoAView1,
        SoAView2, SoAView3,
    },
    error::Error,
    expr::DeviceGpuExpr,
    op::GpuOp,
    policy::CubePolicy,
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
    /// Key column source.
    type Source: KernelColumn<Runtime = Self::Runtime, Item = Self::Item> + KernelColumnAt<S0>;

    /// Lowers keys to the column source consumed by primitive kernels.
    fn key_source(self) -> Result<Self::Source, Error>;
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
    type Source = Source;

    fn key_source(self) -> Result<Self::Source, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(self.source)
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
    type Source = Source;

    fn key_source(self) -> Result<Self::Source, Error> {
        <SoAView1<Source> as KeyInput>::key_source(SoAView1 { source: self })
    }
}

/// Input accepted by [`inclusive_scan`].
#[doc(hidden)]
pub trait InclusiveScanInput<Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Scan output type.
    type Output;

    /// Computes an inclusive scan.
    fn inclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Op> InclusiveScanInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn inclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let bindings = self.source.stage(policy)?;
        primitive_scan::inclusive_scan_tuple1_device_expr::<_, _, Source::Expr, Op>(
            policy,
            &bindings,
            self.source.len(),
        )
    }
}

impl<Source, Op> InclusiveScanInput<Op> for (Source,)
where
    SoAView1<Source>: InclusiveScanInput<Op>,
{
    type Runtime = <SoAView1<Source> as InclusiveScanInput<Op>>::Runtime;
    type Output = <SoAView1<Source> as InclusiveScanInput<Op>>::Output;

    fn inclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as InclusiveScanInput<Op>>::inclusive_scan_input(
            SoAView1 { source: self.0 },
            policy,
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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn inclusive_scan_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = self.$first_field.stage(policy)?;
                $(
                    let $field = self.$field.stage(policy)?;
                )+
                primitive_scan::$scan_fn::<
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
    type Runtime = <SoAView2<Left, Right> as InclusiveScanInput<Op>>::Runtime;
    type Output = <SoAView2<Left, Right> as InclusiveScanInput<Op>>::Output;

    fn inclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as InclusiveScanInput<Op>>::inclusive_scan_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            op,
        )
    }
}

impl<First, Second, Third, Op> InclusiveScanInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: InclusiveScanInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as InclusiveScanInput<Op>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as InclusiveScanInput<Op>>::Output;

    fn inclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as InclusiveScanInput<Op>>::inclusive_scan_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn inclusive_scan_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = self.$first_field.stage(policy)?;
                $(
                    let $field = self.$field.stage(policy)?;
                )+
                primitive_scan::$scan_fn::<
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
                )
            }
        }
    };
}

impl_inclusive_scan_soa_input!(SoA2 -> SoA2<A, B> { left, right } => inclusive_scan_tuple2_device_expr);
impl_inclusive_scan_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third } => inclusive_scan_tuple3_device_expr);

/// Computes an inclusive scan from read-only input into device storage.
pub fn inclusive_scan<InputSource, Op>(
    policy: &CubePolicy<<InputSource as InclusiveScanInput<Op>>::Runtime>,
    source: InputSource,
    _op: Op,
) -> Result<<<InputSource as InclusiveScanInput<Op>>::Output as MaterializeOutput>::Output, Error>
where
    InputSource: InclusiveScanInput<Op>,
    <InputSource as InclusiveScanInput<Op>>::Output:
        MaterializeOutput<Runtime = <InputSource as InclusiveScanInput<Op>>::Runtime>,
{
    materialize(
        policy,
        source.inclusive_scan_input(policy, GpuOp::<Op>::new())?,
    )
}

/// Input accepted by [`exclusive_scan`].
#[doc(hidden)]
pub trait ExclusiveScanInput<Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Initial value type.
    type Init;
    /// Scan output type.
    type Output;

    /// Computes an exclusive scan.
    fn exclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Op> ExclusiveScanInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Init = (Source::Item,);
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn exclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let bindings = self.source.stage(policy)?;
        primitive_scan::exclusive_scan_tuple1_device_expr::<_, _, Source::Expr, Op>(
            policy,
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
    type Runtime = <SoAView1<Source> as ExclusiveScanInput<Op>>::Runtime;
    type Init = <SoAView1<Source> as ExclusiveScanInput<Op>>::Init;
    type Output = <SoAView1<Source> as ExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as ExclusiveScanInput<Op>>::exclusive_scan_input(
            SoAView1 { source: self.0 },
            policy,
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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn exclusive_scan_input(
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
                primitive_scan::$scan_fn::<
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

impl_exclusive_scan_input!(SoAView2 -> SoA2<A, B> { left, right } => exclusive_scan_tuple2_device_expr);
impl_exclusive_scan_input!(SoAView3 -> SoA3<A, B, C> { first, second, third } => exclusive_scan_tuple3_device_expr);

impl<Left, Right, Op> ExclusiveScanInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: ExclusiveScanInput<Op>,
{
    type Runtime = <SoAView2<Left, Right> as ExclusiveScanInput<Op>>::Runtime;
    type Init = <SoAView2<Left, Right> as ExclusiveScanInput<Op>>::Init;
    type Output = <SoAView2<Left, Right> as ExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ExclusiveScanInput<Op>>::exclusive_scan_input(
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

impl<First, Second, Third, Op> ExclusiveScanInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ExclusiveScanInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as ExclusiveScanInput<Op>>::Runtime;
    type Init = <SoAView3<First, Second, Third> as ExclusiveScanInput<Op>>::Init;
    type Output = <SoAView3<First, Second, Third> as ExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ExclusiveScanInput<Op>>::exclusive_scan_input(
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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Init = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn exclusive_scan_input(
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
                primitive_scan::$scan_fn::<
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

impl_exclusive_scan_soa_input!(SoA2 -> SoA2<A, B> { left, right } => exclusive_scan_tuple2_device_expr);
impl_exclusive_scan_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third } => exclusive_scan_tuple3_device_expr);

/// Computes an exclusive scan from read-only input into device storage.
pub fn exclusive_scan<InputSource, Op>(
    policy: &CubePolicy<<InputSource as ExclusiveScanInput<Op>>::Runtime>,
    source: InputSource,
    init: <InputSource as ExclusiveScanInput<Op>>::Init,
    _op: Op,
) -> Result<<<InputSource as ExclusiveScanInput<Op>>::Output as MaterializeOutput>::Output, Error>
where
    InputSource: ExclusiveScanInput<Op>,
    <InputSource as ExclusiveScanInput<Op>>::Output:
        MaterializeOutput<Runtime = <InputSource as ExclusiveScanInput<Op>>::Runtime>,
{
    materialize(
        policy,
        source.exclusive_scan_input(policy, init, GpuOp::<Op>::new())?,
    )
}

/// Input accepted by [`adjacent_difference`].
#[doc(hidden)]
pub trait AdjacentDifferenceInput<Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Adjacent difference output type.
    type Output;

    /// Computes adjacent differences.
    fn adjacent_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Op> AdjacentDifferenceInput<Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn adjacent_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let source =
            super::device_expr_adjacent_difference_with_policy::<Source, Op>(policy, &self)?;
        Ok(SoA1 { source })
    }
}

impl<Source, Op> AdjacentDifferenceInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn adjacent_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let source =
            super::device_expr_adjacent_difference_with_policy::<Source, Op>(policy, &self.source)?;
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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn adjacent_difference_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = self.$first_field.stage(policy)?;
                $(
                    let $field = self.$field.stage(policy)?;
                )+
                primitive_scan::$scan_fn::<
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
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn adjacent_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <Source as AdjacentDifferenceInput<super::Tuple1BinaryOp<Op>>>::adjacent_difference_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1BinaryOp<Op>>::new(),
        )
    }
}

impl<Left, Right, Op> AdjacentDifferenceInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: AdjacentDifferenceInput<Op>,
{
    type Runtime = <SoAView2<Left, Right> as AdjacentDifferenceInput<Op>>::Runtime;
    type Output = <SoAView2<Left, Right> as AdjacentDifferenceInput<Op>>::Output;

    fn adjacent_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as AdjacentDifferenceInput<Op>>::adjacent_difference_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            op,
        )
    }
}

impl<First, Second, Third, Op> AdjacentDifferenceInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: AdjacentDifferenceInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as AdjacentDifferenceInput<Op>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as AdjacentDifferenceInput<Op>>::Output;

    fn adjacent_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as AdjacentDifferenceInput<Op>>::adjacent_difference_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn adjacent_difference_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = self.$first_field.stage(policy)?;
                $(
                    let $field = self.$field.stage(policy)?;
                )+
                primitive_scan::$scan_fn::<
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
                )
            }
        }
    };
}

impl_adjacent_difference_soa_input!(SoA2 -> SoA2<A, B> { left, right } => adjacent_difference_tuple2_device_expr);
impl_adjacent_difference_soa_input!(SoA3 -> SoA3<A, B, C> { first, second, third } => adjacent_difference_tuple3_device_expr);

/// Computes adjacent differences into device storage.
pub fn adjacent_difference<Source, Op>(
    policy: &CubePolicy<<Source as AdjacentDifferenceInput<Op>>::Runtime>,
    source: Source,
    _op: Op,
) -> Result<<<Source as AdjacentDifferenceInput<Op>>::Output as MaterializeOutput>::Output, Error>
where
    Source: AdjacentDifferenceInput<Op>,
    <Source as AdjacentDifferenceInput<Op>>::Output:
        MaterializeOutput<Runtime = <Source as AdjacentDifferenceInput<Op>>::Runtime>,
{
    materialize(
        policy,
        source.adjacent_difference_input(policy, GpuOp::<Op>::new())?,
    )
}

/// Input accepted by [`inclusive_scan_by_key`].
#[doc(hidden)]
pub trait InclusiveScanByKeyInput<KeySource, KeyEq, Op>
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
{
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Scan output type.
    type Output;

    /// Computes an inclusive scan by key.
    fn inclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, KeySource, KeyEq, Op> InclusiveScanByKeyInput<KeySource, KeyEq, Op>
    for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    KeySource: KernelColumn<Runtime = Source::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn inclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(SoA1 {
            source: super::device_expr_inclusive_scan_by_key_expr_keys_with_policy::<
                KeySource,
                Source,
                KeyEq,
                Op,
            >(policy, keys, &self.source)?,
        })
    }
}

impl<Source, KeySource, KeyEq, Op> InclusiveScanByKeyInput<KeySource, KeyEq, Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    KeySource: KernelColumn + KernelColumnAt<S0>,
    SoAView1<Source>: InclusiveScanByKeyInput<KeySource, KeyEq, Op>,
{
    type Runtime = <SoAView1<Source> as InclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Runtime;
    type Output = <SoAView1<Source> as InclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Output;

    fn inclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as InclusiveScanByKeyInput<KeySource, KeyEq, Op>>::inclusive_scan_by_key_input(
            SoAView1 { source: self },
            policy,
            keys,
            key_eq,
            op,
        )
    }
}

impl<Left, Right, KeySource, KeyEq, Op> InclusiveScanByKeyInput<KeySource, KeyEq, Op>
    for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    KeySource: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
    Op: BinaryOp<(Left::Item, Right::Item)>,
{
    type Runtime = Left::Runtime;
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn inclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        keys.validate()?;
        super::ensure_same_len(keys.len(), self.left.len())?;
        let key_bindings = keys.stage(policy)?;
        let left = self.left.stage(policy)?;
        let right = self.right.stage(policy)?;
        primitive_scan::inclusive_scan_tuple2_by_key_values_device_expr::<
            Left::Runtime,
            KeySource::Item,
            Left::Item,
            Right::Item,
            KeySource::Expr,
            Left::Expr,
            Right::Expr,
            KeyEq,
            Op,
        >(policy, &key_bindings, &left, &right, self.left.len())
    }
}

impl<First, Second, Third, KeySource, KeyEq, Op> InclusiveScanByKeyInput<KeySource, KeyEq, Op>
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
    KeySource: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
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
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        keys.validate()?;
        super::ensure_same_len(keys.len(), self.first.len())?;
        let key_bindings = keys.stage(policy)?;
        let first = self.first.stage(policy)?;
        let second = self.second.stage(policy)?;
        let third = self.third.stage(policy)?;
        primitive_scan::inclusive_scan_tuple3_by_key_values_device_expr::<
            First::Runtime,
            KeySource::Item,
            First::Item,
            Second::Item,
            Third::Item,
            KeySource::Expr,
            First::Expr,
            Second::Expr,
            Third::Expr,
            KeyEq,
            Op,
        >(
            policy,
            &key_bindings,
            &first,
            &second,
            &third,
            self.first.len(),
        )
    }
}

macro_rules! impl_inclusive_scan_by_key_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, KeySource, KeyEq, Op> InclusiveScanByKeyInput<KeySource, KeyEq, Op>
            for $input<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            KeySource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            KeySource::Item: CubePrimitive + CubeElement,
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            KeyEq: BinaryPredicateOp<KeySource::Item>,
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
                policy: &CubePolicy<Self::Runtime>,
                keys: &KeySource,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field =
                    super::device_expr_inclusive_scan_by_key_expr_keys_with_policy::<KeySource, $first, KeyEq, Op>(
                        policy,
                        keys,
                        &self.$first_field,
                    )?;
                $(
                    let $field =
                        super::device_expr_inclusive_scan_by_key_expr_keys_with_policy::<KeySource, $rest, KeyEq, Op>(
                            policy,
                            keys,
                            &self.$field,
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
        impl<$( $ty ),+, KeySource, KeyEq, Op> InclusiveScanByKeyInput<KeySource, KeyEq, Op> for ($( $ty ),+)
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            $view<$( $ty ),+>: InclusiveScanByKeyInput<KeySource, KeyEq, Op>,
        {
            type Runtime = <$view<$( $ty ),+> as InclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Runtime;
            type Output = <$view<$( $ty ),+> as InclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Output;

            fn inclusive_scan_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                keys: &KeySource,
                key_eq: GpuOp<KeyEq>,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as InclusiveScanByKeyInput<KeySource, KeyEq, Op>>::inclusive_scan_by_key_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
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
    type Runtime: Runtime;
    type Output;

    fn inclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Values, Keys, KeyEq, Op> InclusiveScanByKeyCall<Values, KeyEq, Op> for Keys
where
    Keys: KeyInput,
    Keys::Item: CubePrimitive + CubeElement,
    Values: InclusiveScanByKeyInput<Keys::Source, KeyEq, Op, Runtime = Keys::Runtime>,
{
    type Runtime = Keys::Runtime;
    type Output = <Values as InclusiveScanByKeyInput<Keys::Source, KeyEq, Op>>::Output;

    fn inclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.key_source()?;
        values.inclusive_scan_by_key_input(policy, &keys, GpuOp::<KeyEq>::new(), GpuOp::<Op>::new())
    }
}

pub fn inclusive_scan_by_key<R, Keys, Values, KeyEq, Op>(
    _policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    _op: Op,
) -> Result<
    <<Keys as InclusiveScanByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Keys: InclusiveScanByKeyCall<Values, KeyEq, Op, Runtime = R>,
    <Keys as InclusiveScanByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        _policy,
        keys.inclusive_scan_by_key_call(
            _policy,
            values,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?,
    )
}

/// Input accepted by [`exclusive_scan_by_key`].
#[doc(hidden)]
pub trait ExclusiveScanByKeyInput<KeySource, KeyEq, Op>
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
{
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Initial value type.
    type Init;
    /// Scan output type.
    type Output;

    /// Computes an exclusive scan by key.
    fn exclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        init: Self::Init,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, KeySource, KeyEq, Op> ExclusiveScanByKeyInput<KeySource, KeyEq, Op>
    for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    KeySource: KernelColumn<Runtime = Source::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
    Op: BinaryOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Init = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn exclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(SoA1 {
            source: super::device_expr_exclusive_scan_by_key_expr_keys_with_policy::<
                KeySource,
                Source,
                KeyEq,
                Op,
            >(policy, keys, &self.source, init)?,
        })
    }
}

impl<Source, KeySource, KeyEq, Op> ExclusiveScanByKeyInput<KeySource, KeyEq, Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    KeySource: KernelColumn + KernelColumnAt<S0>,
    SoAView1<Source>: ExclusiveScanByKeyInput<KeySource, KeyEq, Op>,
{
    type Runtime = <SoAView1<Source> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Runtime;
    type Init = <SoAView1<Source> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Init;
    type Output = <SoAView1<Source> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Output;

    fn exclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        init: Self::Init,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::exclusive_scan_by_key_input(
            SoAView1 { source: self },
            policy,
            keys,
            init,
            key_eq,
            op,
        )
    }
}

impl<Left, Right, KeySource, KeyEq, Op> ExclusiveScanByKeyInput<KeySource, KeyEq, Op>
    for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    KeySource: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
    Op: BinaryOp<(Left::Item, Right::Item)>,
{
    type Runtime = Left::Runtime;
    type Init = (Left::Item, Right::Item);
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn exclusive_scan_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        keys.validate()?;
        super::ensure_same_len(keys.len(), self.left.len())?;
        let key_bindings = keys.stage(policy)?;
        let left = self.left.stage(policy)?;
        let right = self.right.stage(policy)?;
        primitive_scan::exclusive_scan_tuple2_by_key_values_device_expr::<
            Left::Runtime,
            KeySource::Item,
            Left::Item,
            Right::Item,
            KeySource::Expr,
            Left::Expr,
            Right::Expr,
            KeyEq,
            Op,
        >(policy, &key_bindings, &left, &right, self.left.len(), init)
    }
}

impl<First, Second, Third, KeySource, KeyEq, Op> ExclusiveScanByKeyInput<KeySource, KeyEq, Op>
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
    KeySource: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
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
        policy: &CubePolicy<Self::Runtime>,
        keys: &KeySource,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        keys.validate()?;
        super::ensure_same_len(keys.len(), self.first.len())?;
        let key_bindings = keys.stage(policy)?;
        let first = self.first.stage(policy)?;
        let second = self.second.stage(policy)?;
        let third = self.third.stage(policy)?;
        primitive_scan::exclusive_scan_tuple3_by_key_values_device_expr::<
            First::Runtime,
            KeySource::Item,
            First::Item,
            Second::Item,
            Third::Item,
            KeySource::Expr,
            First::Expr,
            Second::Expr,
            Third::Expr,
            KeyEq,
            Op,
        >(
            policy,
            &key_bindings,
            &first,
            &second,
            &third,
            self.first.len(),
            init,
        )
    }
}

macro_rules! impl_exclusive_scan_by_key_soa_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, KeySource, KeyEq, Op> ExclusiveScanByKeyInput<KeySource, KeyEq, Op>
            for $input<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            KeySource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            KeySource::Item: CubePrimitive + CubeElement,
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            KeyEq: BinaryPredicateOp<KeySource::Item>,
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
                policy: &CubePolicy<Self::Runtime>,
                keys: &KeySource,
                init: Self::Init,
                _key_eq: GpuOp<KeyEq>,
                _op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let ($first_field, $( $field ),+) = init;
                let $first_field =
                    super::device_expr_exclusive_scan_by_key_expr_keys_with_policy::<KeySource, $first, KeyEq, Op>(
                        policy,
                        keys,
                        &self.$first_field,
                        $first_field,
                    )?;
                $(
                    let $field =
                        super::device_expr_exclusive_scan_by_key_expr_keys_with_policy::<KeySource, $rest, KeyEq, Op>(
                            policy,
                            keys,
                            &self.$field,
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
        impl<$( $ty ),+, KeySource, KeyEq, Op> ExclusiveScanByKeyInput<KeySource, KeyEq, Op> for ($( $ty ),+)
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            $view<$( $ty ),+>: ExclusiveScanByKeyInput<KeySource, KeyEq, Op>,
        {
            type Runtime = <$view<$( $ty ),+> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Runtime;
            type Init = <$view<$( $ty ),+> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Init;
            type Output = <$view<$( $ty ),+> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::Output;

            fn exclusive_scan_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                keys: &KeySource,
                init: Self::Init,
                key_eq: GpuOp<KeyEq>,
                op: GpuOp<Op>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as ExclusiveScanByKeyInput<KeySource, KeyEq, Op>>::exclusive_scan_by_key_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
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
    type Runtime: Runtime;
    type Init;
    type Output;

    fn exclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
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
    Values: ExclusiveScanByKeyInput<Keys::Source, KeyEq, Op, Runtime = Keys::Runtime>,
{
    type Runtime = Keys::Runtime;
    type Init = <Values as ExclusiveScanByKeyInput<Keys::Source, KeyEq, Op>>::Init;
    type Output = <Values as ExclusiveScanByKeyInput<Keys::Source, KeyEq, Op>>::Output;

    fn exclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let keys = self.key_source()?;
        values.exclusive_scan_by_key_input(
            policy,
            &keys,
            init,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )
    }
}

pub fn exclusive_scan_by_key<R, Keys, Values, KeyEq, Op>(
    _policy: &CubePolicy<R>,
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
    R: Runtime,
    Keys: ExclusiveScanByKeyCall<Values, KeyEq, Op, Runtime = R>,
    <Keys as ExclusiveScanByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        _policy,
        keys.exclusive_scan_by_key_call(
            _policy,
            values,
            init,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?,
    )
}
