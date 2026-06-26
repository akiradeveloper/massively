use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA1, SoA2, SoA3, SoAView1,
        SoAView2, SoAView3,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::range,
};
use cubecl::prelude::*;

fn scatter_one<ValueSource, IndexSource>(
    policy: &CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    len: usize,
    default: ValueSource::Item,
) -> Result<DeviceVec<ValueSource::Runtime, ValueSource::Item>, Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    let initial = range::filled(policy, len, default)?;
    super::device_expr_scatter_with_policy::<
        ValueSource,
        IndexSource,
        DeviceVec<ValueSource::Runtime, ValueSource::Item>,
    >(policy, values, indices, &initial)
}

fn scatter_where_one<ValueSource, IndexSource, Stencil, Pred>(
    policy: &CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    stencil: &Stencil,
    len: usize,
    default: ValueSource::Item,
) -> Result<DeviceVec<ValueSource::Runtime, ValueSource::Item>, Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    values.validate()?;
    indices.validate()?;
    super::ensure_same_len(values.len(), indices.len())?;
    super::ensure_same_len(values.len(), stencil.len())?;
    let flags = stencil.selection_handles_with_policy(policy, false)?;
    let initial = range::filled(policy, len, default)?;
    let input_len = values.len();
    let block_count = input_len.div_ceil(256);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    let value_bindings = values.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let value_slot0 = value_bindings.slot_or_first(0);
    let value_slot1 = value_bindings.slot_or_first(1);
    let value_slot2 = value_bindings.slot_or_first(2);
    let value_slot3 = value_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let value_slot_offsets = value_bindings.slot_offsets_handle(policy.client())?;
    let index_slot_offsets = index_bindings.slot_offsets_handle(policy.client())?;
    if input_len != 0 {
        unsafe {
            scatter_if_flags_kernel::launch_unchecked::<
                ValueSource::Item,
                ValueSource::Expr,
                IndexSource::Expr,
                ValueSource::Runtime,
            >(
                policy.client(),
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(256),
                unsafe { BufferArg::from_raw_parts(value_slot0.0.clone(), value_slot0.1) },
                unsafe { BufferArg::from_raw_parts(value_slot1.0.clone(), value_slot1.1) },
                unsafe { BufferArg::from_raw_parts(value_slot2.0.clone(), value_slot2.1) },
                unsafe { BufferArg::from_raw_parts(value_slot3.0.clone(), value_slot3.1) },
                unsafe { BufferArg::from_raw_parts(value_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1) },
                unsafe { BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1) },
                unsafe { BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1) },
                unsafe { BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1) },
                unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(flags.flag.clone(), flags.len) },
                unsafe { BufferArg::from_raw_parts(initial.handle.clone(), initial.len) },
            );
        }
    }
    Ok(initial)
}

/// Input accepted by [`scatter`].
#[doc(hidden)]
pub trait ScatterInput<Indices> {
    /// Runtime used by this input.
    type Runtime: Runtime;

    /// Default value accepted by scatter.
    type Default;
    /// Output produced by scatter.
    type Output;

    /// Scatters `self[i]` into default-initialized output at `indices[i]`.
    fn scatter_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: Indices,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error>;
}

impl<ValueSource, IndexSource> ScatterInput<SoAView1<IndexSource>> for SoAView1<ValueSource>
where
    Self: ReadOnlySoA<Item = (ValueSource::Item,), Scalar = ValueSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = (u32,), Scalar = u32>,
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_input(
        self,
        policy: &CubePolicy<ValueSource::Runtime>,
        indices: SoAView1<IndexSource>,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&indices)?;
        Ok(SoA1 {
            source: scatter_one::<ValueSource, IndexSource>(
                policy,
                &self.source,
                &indices.source,
                len,
                default,
            )?,
        })
    }
}

impl<ValueSource, IndexSource> ScatterInput<IndexSource> for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_input(
        self,
        policy: &CubePolicy<ValueSource::Runtime>,
        indices: IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        Ok(SoA1 {
            source: scatter_one::<ValueSource, IndexSource>(policy, &self, &indices, len, default)?,
        })
    }
}

impl<ValueSource, IndexSource> ScatterInput<(IndexSource,)> for (ValueSource,)
where
    SoAView1<ValueSource>: ScatterInput<SoAView1<IndexSource>>,
{
    type Runtime = <SoAView1<ValueSource> as ScatterInput<SoAView1<IndexSource>>>::Runtime;
    type Default = <SoAView1<ValueSource> as ScatterInput<SoAView1<IndexSource>>>::Default;
    type Output = <SoAView1<ValueSource> as ScatterInput<SoAView1<IndexSource>>>::Output;

    fn scatter_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView1<ValueSource> as ScatterInput<SoAView1<IndexSource>>>::scatter_input(
            SoAView1 { source: self.0 },
            policy,
            SoAView1 { source: indices.0 },
            len,
            default,
        )
    }
}

impl<Left, Right, IndexSource> ScatterInput<(IndexSource,)> for (Left, Right)
where
    SoAView2<Left, Right>: ScatterInput<SoAView1<IndexSource>>,
{
    type Runtime = <SoAView2<Left, Right> as ScatterInput<SoAView1<IndexSource>>>::Runtime;
    type Default = <SoAView2<Left, Right> as ScatterInput<SoAView1<IndexSource>>>::Default;
    type Output = <SoAView2<Left, Right> as ScatterInput<SoAView1<IndexSource>>>::Output;

    fn scatter_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ScatterInput<SoAView1<IndexSource>>>::scatter_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            SoAView1 { source: indices.0 },
            len,
            default,
        )
    }
}

impl<First, Second, Third, IndexSource> ScatterInput<(IndexSource,)> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ScatterInput<SoAView1<IndexSource>>,
{
    type Runtime = <SoAView3<First, Second, Third> as ScatterInput<SoAView1<IndexSource>>>::Runtime;
    type Default = <SoAView3<First, Second, Third> as ScatterInput<SoAView1<IndexSource>>>::Default;
    type Output = <SoAView3<First, Second, Third> as ScatterInput<SoAView1<IndexSource>>>::Output;

    fn scatter_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ScatterInput<SoAView1<IndexSource>>>::scatter_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            SoAView1 { source: indices.0 },
            len,
            default,
        )
    }
}

macro_rules! impl_scatter_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, IndexSource> ScatterInput<SoAView1<IndexSource>>
            for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            IndexSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = u32> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: GpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: GpuExpr<<$rest as KernelColumn>::Item>,
            )+
            IndexSource::Expr: GpuExpr<u32>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            type Default = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn scatter_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                indices: SoAView1<IndexSource>,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let ($first_field, $( $field ),+) = default;
                let $first_field = scatter_one::<$first, IndexSource>(
                    policy,
                    &self.$first_field,
                    &indices.source,
                    len,
                    $first_field,
                )?;
                $(
                    let $field = scatter_one::<$rest, IndexSource>(
                        policy,
                        &self.$field,
                        &indices.source,
                        len,
                        $field,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_scatter_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_scatter_input!(SoA2 -> SoA2<A, B> { left, right });
impl_scatter_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_input!(SoA3 -> SoA3<A, B, C> { first, second, third });

macro_rules! impl_scatter_input_index_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource> ScatterInput<IndexSource>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: ScatterInput<SoAView1<IndexSource>>,
        {
            type Runtime = <Self as ScatterInput<SoAView1<IndexSource>>>::Runtime;
            type Default = <Self as ScatterInput<SoAView1<IndexSource>>>::Default;
            type Output = <Self as ScatterInput<SoAView1<IndexSource>>>::Output;

            fn scatter_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: IndexSource,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                <Self as ScatterInput<SoAView1<IndexSource>>>::scatter_input(
                    self,
                    policy,
                    SoAView1 { source: indices },
                    len,
                    default,
                )
            }
        }
    };
}

impl_scatter_input_index_source!(SoAView2<A, B>);
impl_scatter_input_index_source!(SoA2<A, B>);
impl_scatter_input_index_source!(SoAView3<A, B, C>);
impl_scatter_input_index_source!(SoA3<A, B, C>);

/// Input accepted by [`scatter_where`].
#[doc(hidden)]
pub trait ScatterWhereInput<Indices, Stencil, Pred> {
    /// Runtime used by this input.
    type Runtime: Runtime;

    /// Default value accepted by scatter-if.
    type Default;
    /// Output produced by scatter-if.
    type Output;

    /// Scatters selected values into default-initialized output at `indices[i]`.
    fn scatter_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: Indices,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<ValueSource, IndexSource, Stencil, Pred>
    ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred> for SoAView1<ValueSource>
where
    Self: ReadOnlySoA<Item = (ValueSource::Item,), Scalar = ValueSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = (u32,), Scalar = u32>,
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_where_input(
        self,
        policy: &CubePolicy<ValueSource::Runtime>,
        indices: SoAView1<IndexSource>,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&indices)?;
        Ok(SoA1 {
            source: scatter_where_one::<ValueSource, IndexSource, Stencil, Pred>(
                policy,
                &self.source,
                &indices.source,
                &stencil,
                len,
                default,
            )?,
        })
    }
}

impl<ValueSource, IndexSource, Stencil, Pred> ScatterWhereInput<IndexSource, Stencil, Pred>
    for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_where_input(
        self,
        policy: &CubePolicy<ValueSource::Runtime>,
        indices: IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        let _ = pred;
        Ok(SoA1 {
            source: scatter_where_one::<ValueSource, IndexSource, Stencil, Pred>(
                policy, &self, &indices, &stencil, len, default,
            )?,
        })
    }
}

impl<ValueSource, IndexSource, Stencil, Pred> ScatterWhereInput<(IndexSource,), Stencil, Pred>
    for (ValueSource,)
where
    SoAView1<ValueSource>: ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>,
{
    type Runtime =
        <SoAView1<ValueSource> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Runtime;
    type Default =
        <SoAView1<ValueSource> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;
    type Output =
        <SoAView1<ValueSource> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;

    fn scatter_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<ValueSource> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::scatter_where_input(
            SoAView1 { source: self.0 },
            policy,
            SoAView1 { source: indices.0 },
            stencil,
            len,
            default,
            pred,
        )
    }
}

impl<Left, Right, IndexSource, Stencil, Pred> ScatterWhereInput<(IndexSource,), Stencil, Pred>
    for (Left, Right)
where
    SoAView2<Left, Right>: ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>,
{
    type Runtime =
        <SoAView2<Left, Right> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Runtime;
    type Default =
        <SoAView2<Left, Right> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;
    type Output =
        <SoAView2<Left, Right> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;

    fn scatter_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::scatter_where_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            SoAView1 { source: indices.0 },
            stencil,
            len,
            default,
            pred,
        )
    }
}

impl<First, Second, Third, IndexSource, Stencil, Pred>
    ScatterWhereInput<(IndexSource,), Stencil, Pred> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as ScatterWhereInput<
        SoAView1<IndexSource>,
        Stencil,
        Pred,
    >>::Runtime;
    type Default = <SoAView3<First, Second, Third> as ScatterWhereInput<
        SoAView1<IndexSource>,
        Stencil,
        Pred,
    >>::Default;
    type Output = <SoAView3<First, Second, Third> as ScatterWhereInput<
        SoAView1<IndexSource>,
        Stencil,
        Pred,
    >>::Output;

    fn scatter_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ScatterWhereInput<
            SoAView1<IndexSource>,
            Stencil,
            Pred,
        >>::scatter_where_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            SoAView1 { source: indices.0 },
            stencil,
            len,
            default,
            pred,
        )
    }
}

macro_rules! impl_scatter_where_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, IndexSource, Stencil, Pred> ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>
            for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            IndexSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = u32> + KernelColumnAt<S0>,
            Stencil: super::SelectionStencil<Pred, Runtime = <$first as KernelColumn>::Runtime>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            IndexSource::Expr: DeviceGpuExpr<u32>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            type Default = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn scatter_where_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                indices: SoAView1<IndexSource>,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let ($first_field, $( $field ),+) = default;
                let $first_field = scatter_where_one::<$first, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$first_field,
                    &indices.source,
                    &stencil,
                    len,
                    $first_field,
                )?;
                $(
                    let $field = scatter_where_one::<$rest, IndexSource, Stencil, Pred>(
                        policy,
                        &self.$field,
                        &indices.source,
                        &stencil,
                        len,
                        $field,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_scatter_where_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_scatter_where_input!(SoA2 -> SoA2<A, B> { left, right });
impl_scatter_where_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_where_input!(SoA3 -> SoA3<A, B, C> { first, second, third });

macro_rules! impl_scatter_where_input_sources {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource, Stencil, Pred> ScatterWhereInput<IndexSource, Stencil, Pred>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>,
        {
            type Runtime = <Self as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Runtime;
            type Default = <Self as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;
            type Output = <Self as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;

            fn scatter_where_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: IndexSource,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <Self as ScatterWhereInput<SoAView1<IndexSource>, Stencil, Pred>>::scatter_where_input(
                    self,
                    policy,
                    SoAView1 { source: indices },
                    stencil,
                    len,
                    default,
                    pred,
                )
            }
        }
    };
}

impl_scatter_where_input_sources!(SoAView2<A, B>);
impl_scatter_where_input_sources!(SoA2<A, B>);
impl_scatter_where_input_sources!(SoAView3<A, B, C>);
impl_scatter_where_input_sources!(SoA3<A, B, C>);

/// Scatters `values[i]` into a new output at `indices[i]`.
///
/// The output is allocated with `len` elements, initialized with `default`, and
/// then updated by the scatter. For multiple value columns, pass borrowed
/// columns as a tuple, such as `(values.slice(..), tags.slice(..))`, and use
/// the same tuple shape for `default`.
pub fn scatter<Values, Indices>(
    policy: &CubePolicy<<Values as ScatterInput<Indices>>::Runtime>,
    values: Values,
    indices: Indices,
    len: usize,
    default: <Values as ScatterInput<Indices>>::Default,
) -> Result<<<Values as ScatterInput<Indices>>::Output as MaterializeOutput>::Output, Error>
where
    Values: ScatterInput<Indices>,
    <Values as ScatterInput<Indices>>::Output:
        MaterializeOutput<Runtime = <Values as ScatterInput<Indices>>::Runtime>,
{
    materialize(policy, values.scatter_input(policy, indices, len, default)?)
}

/// Scatters selected values into a new output at `indices[i]`.
///
/// The output is allocated with `len` elements, initialized with `default`, and
/// then updated for values satisfying `Pred`.
pub fn scatter_where<Values, Indices, Stencil, Pred>(
    policy: &CubePolicy<<Values as ScatterWhereInput<Indices, Stencil, Pred>>::Runtime>,
    values: Values,
    indices: Indices,
    len: usize,
    default: <Values as ScatterWhereInput<Indices, Stencil, Pred>>::Default,
    stencil: Stencil,
    _pred: Pred,
) -> Result<
    <<Values as ScatterWhereInput<Indices, Stencil, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Values: ScatterWhereInput<Indices, Stencil, Pred>,
    <Values as ScatterWhereInput<Indices, Stencil, Pred>>::Output:
        MaterializeOutput<Runtime = <Values as ScatterWhereInput<Indices, Stencil, Pred>>::Runtime>,
{
    materialize(
        policy,
        values.scatter_where_input(policy, indices, stencil, len, default, GpuOp::<Pred>::new())?,
    )
}
