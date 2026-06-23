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
    primitives::range as primitive_range,
};
use cubecl::prelude::*;

const BLOCK_API_SIZE: u32 = 256;

fn gather_if_one<InputSource, IndexSource, Stencil, Pred>(
    policy: &CubePolicy<InputSource::Runtime>,
    input: &InputSource,
    indices: &IndexSource,
    stencil: &Stencil,
    default: InputSource::Item,
) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = InputSource::Runtime>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    input.validate()?;
    indices.validate()?;
    super::ensure_same_len(indices.len(), stencil.len())?;
    let flags = stencil.selection_handles_with_policy(policy, false)?;

    let len = indices.len();
    let output = primitive_range::filled(policy, len, default)?;
    let num_blocks = len.div_ceil(BLOCK_API_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = policy.client();
    let input_bindings = input.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let input_slot0 = input_bindings.slot_or_first(0);
    let input_slot1 = input_bindings.slot_or_first(1);
    let input_slot2 = input_bindings.slot_or_first(2);
    let input_slot3 = input_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let input_slot_offsets = input_bindings.slot_offsets_handle(client)?;
    let index_slot_offsets = index_bindings.slot_offsets_handle(client)?;

    if len != 0 {
        unsafe {
            gather_if_flags_kernel::launch_unchecked::<
                InputSource::Item,
                InputSource::Expr,
                IndexSource::Expr,
                InputSource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_API_SIZE),
                unsafe { BufferArg::from_raw_parts(input_slot0.0.clone(), input_slot0.1) },
                unsafe { BufferArg::from_raw_parts(input_slot1.0.clone(), input_slot1.1) },
                unsafe { BufferArg::from_raw_parts(input_slot2.0.clone(), input_slot2.1) },
                unsafe { BufferArg::from_raw_parts(input_slot3.0.clone(), input_slot3.1) },
                unsafe { BufferArg::from_raw_parts(input_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1) },
                unsafe { BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1) },
                unsafe { BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1) },
                unsafe { BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1) },
                unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(flags.flag.clone(), flags.len) },
                unsafe { BufferArg::from_raw_parts(output.handle.clone(), output.len) },
            );
        }
    }

    Ok(output)
}

/// Input accepted by [`gather`].
#[doc(hidden)]
pub trait GatherInput<Indices> {
    /// Runtime used by this input.
    type Runtime: Runtime;

    /// Output produced by gather.
    type Output;

    /// Gathers `self[indices[i]]`.
    fn gather_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: Indices,
    ) -> Result<Self::Output, Error>;
}

impl<InputSource, IndexSource> GatherInput<SoAView1<IndexSource>> for SoAView1<InputSource>
where
    Self: ReadOnlySoA<Item = (InputSource::Item,), Scalar = InputSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = (u32,), Scalar = u32>,
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Runtime = InputSource::Runtime;
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_input(
        self,
        policy: &CubePolicy<InputSource::Runtime>,
        indices: SoAView1<IndexSource>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&indices)?;
        Ok(SoA1 {
            source: super::device_expr_gather_with_policy::<InputSource, IndexSource>(
                policy,
                &self.source,
                &indices.source,
            )?,
        })
    }
}

impl<InputSource, IndexSource> GatherInput<IndexSource> for InputSource
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Runtime = InputSource::Runtime;
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_input(
        self,
        policy: &CubePolicy<InputSource::Runtime>,
        indices: IndexSource,
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: indices },
        )
    }
}

impl<InputSource, IndexSource> GatherInput<(IndexSource,)> for (InputSource,)
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Runtime = InputSource::Runtime;
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_input(
        self,
        policy: &CubePolicy<InputSource::Runtime>,
        indices: (IndexSource,),
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView1 { source: self.0 },
            policy,
            SoAView1 { source: indices.0 },
        )
    }
}

impl<Left, Right, IndexSource> GatherInput<(IndexSource,)> for (Left, Right)
where
    SoAView2<Left, Right>: GatherInput<SoAView1<IndexSource>>,
{
    type Runtime = <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::Runtime;
    type Output = <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            SoAView1 { source: indices.0 },
        )
    }
}

impl<Left, Right, IndexSource> GatherInput<IndexSource> for (Left, Right)
where
    SoAView2<Left, Right>: GatherInput<SoAView1<IndexSource>>,
{
    type Runtime = <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::Runtime;
    type Output = <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: IndexSource,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            SoAView1 { source: indices },
        )
    }
}

impl<First, Second, Third, IndexSource> GatherInput<(IndexSource,)> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: GatherInput<SoAView1<IndexSource>>,
{
    type Runtime = <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            SoAView1 { source: indices.0 },
        )
    }
}

impl<First, Second, Third, IndexSource> GatherInput<IndexSource> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: GatherInput<SoAView1<IndexSource>>,
{
    type Runtime = <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: IndexSource,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            SoAView1 { source: indices },
        )
    }
}

macro_rules! impl_gather_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, IndexSource> GatherInput<SoAView1<IndexSource>>
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

            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn gather_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                indices: SoAView1<IndexSource>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let $first_field = super::device_expr_gather_with_policy::<$first, IndexSource>(
                    policy,
                    &self.$first_field,
                    &indices.source,
                )?;
                $(
                    let $field = super::device_expr_gather_with_policy::<$rest, IndexSource>(
                        policy,
                        &self.$field,
                        &indices.source,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_gather_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_gather_input!(SoA2 -> SoA2<A, B> { left, right });
impl_gather_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_gather_input!(SoA3 -> SoA3<A, B, C> { first, second, third });

macro_rules! impl_gather_input_index_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource> GatherInput<IndexSource>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: GatherInput<SoAView1<IndexSource>>,
        {
            type Runtime = <Self as GatherInput<SoAView1<IndexSource>>>::Runtime;
            type Output = <Self as GatherInput<SoAView1<IndexSource>>>::Output;

            fn gather_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: IndexSource,
            ) -> Result<Self::Output, Error> {
                <Self as GatherInput<SoAView1<IndexSource>>>::gather_input(
                    self,
                    policy,
                    SoAView1 { source: indices },
                )
            }
        }
    };
}

impl_gather_input_index_source!(SoAView2<A, B>);
impl_gather_input_index_source!(SoA2<A, B>);
impl_gather_input_index_source!(SoAView3<A, B, C>);
impl_gather_input_index_source!(SoA3<A, B, C>);

/// Input accepted by [`gather_if`].
#[doc(hidden)]
pub trait GatherIfInput<Indices, Stencil, Pred> {
    /// Runtime used by this input.
    type Runtime: Runtime;

    /// Output produced by gather-if.
    type Output;
    /// Default value used for positions that are not selected.
    type Default;

    /// Gathers selected elements into default-initialized output.
    fn gather_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: Indices,
        stencil: Stencil,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<InputSource, IndexSource, Stencil, Pred> GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>
    for SoAView1<InputSource>
where
    Self: ReadOnlySoA<Item = (InputSource::Item,), Scalar = InputSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = (u32,), Scalar = u32>,
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = InputSource::Runtime>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = InputSource::Runtime;
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;
    type Default = (InputSource::Item,);

    fn gather_if_input(
        self,
        policy: &CubePolicy<InputSource::Runtime>,
        indices: SoAView1<IndexSource>,
        stencil: Stencil,
        default: Self::Default,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        Ok(SoA1 {
            source: gather_if_one::<InputSource, IndexSource, Stencil, Pred>(
                policy,
                &self.source,
                &indices.source,
                &stencil,
                default.0,
            )?,
        })
    }
}

impl<InputSource, IndexSource, Stencil, Pred> GatherIfInput<IndexSource, Stencil, Pred>
    for InputSource
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = InputSource::Runtime>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = InputSource::Runtime;
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;
    type Default = InputSource::Item;

    fn gather_if_input(
        self,
        policy: &CubePolicy<InputSource::Runtime>,
        indices: IndexSource,
        stencil: Stencil,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: indices },
            stencil,
            (default,),
            pred,
        )
    }
}

impl<InputSource, IndexSource, Stencil, Pred> GatherIfInput<(IndexSource,), Stencil, Pred>
    for (InputSource,)
where
    SoAView1<InputSource>: GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>,
{
    type Runtime =
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Runtime;
    type Output =
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;
    type Default =
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;

    fn gather_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        stencil: Stencil,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
            SoAView1 { source: self.0 },
            policy,
            SoAView1 { source: indices.0 },
            stencil,
            default,
            pred,
        )
    }
}

impl<Left, Right, IndexSource, Stencil, Pred> GatherIfInput<(IndexSource,), Stencil, Pred>
    for (Left, Right)
where
    SoAView2<Left, Right>: GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>,
{
    type Runtime =
        <SoAView2<Left, Right> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Runtime;
    type Output =
        <SoAView2<Left, Right> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;
    type Default =
        <SoAView2<Left, Right> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;

    fn gather_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        stencil: Stencil,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            SoAView1 { source: indices.0 },
            stencil,
            default,
            pred,
        )
    }
}

impl<First, Second, Third, IndexSource, Stencil, Pred> GatherIfInput<(IndexSource,), Stencil, Pred>
    for (First, Second, Third)
where
    SoAView3<First, Second, Third>: GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as GatherIfInput<
        SoAView1<IndexSource>,
        Stencil,
        Pred,
    >>::Runtime;
    type Output = <SoAView3<First, Second, Third> as GatherIfInput<
        SoAView1<IndexSource>,
        Stencil,
        Pred,
    >>::Output;
    type Default = <SoAView3<First, Second, Third> as GatherIfInput<
        SoAView1<IndexSource>,
        Stencil,
        Pred,
    >>::Default;

    fn gather_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: (IndexSource,),
        stencil: Stencil,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as GatherIfInput<
            SoAView1<IndexSource>,
            Stencil,
            Pred,
        >>::gather_if_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            SoAView1 { source: indices.0 },
            stencil,
            default,
            pred,
        )
    }
}

macro_rules! impl_gather_if_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        #[allow(non_camel_case_types)]
        impl<$first, $( $rest ),+, IndexSource, Stencil, Pred>
            GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>
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

            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;
            type Default = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );

            fn gather_if_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                indices: SoAView1<IndexSource>,
                stencil: Stencil,
                default: Self::Default,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let ($first_field, $( $field ),+) = default;
                let $first_field = gather_if_one::<$first, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$first_field,
                    &indices.source,
                    &stencil,
                    $first_field,
                )?;
                $(
                    let $field = gather_if_one::<$rest, IndexSource, Stencil, Pred>(
                        policy,
                        &self.$field,
                        &indices.source,
                        &stencil,
                        $field,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_gather_if_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_gather_if_input!(SoA2 -> SoA2<A, B> { left, right });
impl_gather_if_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_gather_if_input!(SoA3 -> SoA3<A, B, C> { first, second, third });

macro_rules! impl_gather_if_input_sources {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource, Stencil, Pred>
            GatherIfInput<IndexSource, Stencil, Pred>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>,
        {
            type Runtime = <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Runtime;
            type Output = <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;
            type Default = <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;

            fn gather_if_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: IndexSource,
                stencil: Stencil,
                default: Self::Default,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
                    self,
                    policy,
                    SoAView1 { source: indices },
                    stencil,
                    default,
                    pred,
                )
            }
        }
    };
}

impl_gather_if_input_sources!(SoAView2<A, B>);
impl_gather_if_input_sources!(SoA2<A, B>);
impl_gather_if_input_sources!(SoAView3<A, B, C>);
impl_gather_if_input_sources!(SoA3<A, B, C>);

/// Gathers `input[indices[i]]` into new owned device storage.
///
/// This is a borrowing algorithm: `input` and `indices` are read-only. For
/// multiple value columns, pass borrowed columns as `SoA2` or `SoA3`.
/// Indices may be passed as `SoA1(indices.slice(..))`.
pub fn gather<Input, Indices>(
    policy: &CubePolicy<<Input as GatherInput<Indices>>::Runtime>,
    input: Input,
    indices: Indices,
) -> Result<<<Input as GatherInput<Indices>>::Output as MaterializeOutput>::Output, Error>
where
    Input: GatherInput<Indices>,
    <Input as GatherInput<Indices>>::Output:
        MaterializeOutput<Runtime = <Input as GatherInput<Indices>>::Runtime>,
{
    materialize(policy, input.gather_input(policy, indices)?)
}

/// Gathers elements whose staged stencil flag satisfies `Pred`.
///
/// This is a borrowing algorithm: `input` and `indices` are read-only.
pub fn gather_if<Input, Indices, Stencil, Pred>(
    policy: &CubePolicy<<Input as GatherIfInput<Indices, Stencil, Pred>>::Runtime>,
    input: Input,
    indices: Indices,
    stencil: Stencil,
    default: <Input as GatherIfInput<Indices, Stencil, Pred>>::Default,
    _pred: Pred,
) -> Result<
    <<Input as GatherIfInput<Indices, Stencil, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Input: GatherIfInput<Indices, Stencil, Pred>,
    <Input as GatherIfInput<Indices, Stencil, Pred>>::Output:
        MaterializeOutput<Runtime = <Input as GatherIfInput<Indices, Stencil, Pred>>::Runtime>,
{
    materialize(
        policy,
        input.gather_if_input(policy, indices, stencil, default, GpuOp::<Pred>::new())?,
    )
}
