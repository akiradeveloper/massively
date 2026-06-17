use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, S0, SoA1, SoA2, SoA3, SoA4, SoA5, SoA6, SoA7,
        SoA8, SoA9, SoA10, SoA11, SoA12, SoVA, SoVA1, SoVA2, SoVA3, SoVA4, SoVA5, SoVA6, SoVA7,
        SoVA8, SoVA9, SoVA10, SoVA11, SoVA12, StorageKernelColumn,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::{GpuOp, PredicateOp},
    primitives::range as primitive_range,
};
use cubecl::prelude::*;

const BLOCK_API_SIZE: u32 = 256;

fn gather_if_one<InputSource, IndexSource, StencilSource, InitialSource, Pred>(
    input: &InputSource,
    indices: &IndexSource,
    stencil: &StencilSource,
    initial: &InitialSource,
) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    StencilSource: KernelColumn<Runtime = InputSource::Runtime> + KernelColumnAt<S0>,
    InitialSource:
        KernelColumn<Runtime = InputSource::Runtime, Item = InputSource::Item> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    StencilSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    StencilSource::Expr: DeviceGpuExpr<StencilSource::Item>,
    InitialSource::Expr: DeviceGpuExpr<InputSource::Item>,
    Pred: PredicateOp<StencilSource::Item>,
{
    input.validate()?;
    indices.validate()?;
    stencil.validate()?;
    initial.validate()?;

    let input = super::device_expr_collect(input)?;
    let indices = super::device_expr_collect(indices)?;
    let stencil = super::device_expr_collect(stencil)?;
    let initial = super::device_expr_collect(initial)?;

    super::ensure_same_len(stencil.len, indices.len)?;
    super::ensure_same_len(initial.len, indices.len)?;

    let output = primitive_range::copy_device(&initial)?;
    let num_blocks = indices.len.div_ceil(BLOCK_API_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = input.policy.client();

    if indices.len != 0 {
        unsafe {
            gather_if_kernel::launch_unchecked::<
                InputSource::Item,
                StencilSource::Item,
                Pred,
                InputSource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_API_SIZE),
                ArrayArg::from_raw_parts::<InputSource::Item>(&input.handle, input.len, 1),
                ArrayArg::from_raw_parts::<u32>(&indices.handle, indices.len, 1),
                ArrayArg::from_raw_parts::<StencilSource::Item>(&stencil.handle, stencil.len, 1),
                ArrayArg::from_raw_parts::<InputSource::Item>(&output.handle, output.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(output)
}

/// Input accepted by [`gather`].
#[doc(hidden)]
pub trait GatherInput<Indices> {
    /// Output produced by gather.
    type Output;

    /// Gathers `self[indices[i]]`.
    fn gather_input(self, indices: Indices) -> Result<Self::Output, Error>;
}

impl<InputSource, IndexSource> GatherInput<SoVA1<IndexSource>> for SoVA1<InputSource>
where
    Self: SoVA<Item = InputSource::Item, Scalar = InputSource::Item>,
    SoVA1<IndexSource>: SoVA<Item = u32, Scalar = u32>,
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_input(self, indices: SoVA1<IndexSource>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        SoVA::validate(&indices)?;
        Ok(SoA1 {
            source: super::device_expr_gather::<InputSource, IndexSource>(
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
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_input(self, indices: IndexSource) -> Result<Self::Output, Error> {
        <SoVA1<InputSource> as GatherInput<SoVA1<IndexSource>>>::gather_input(
            SoVA1 { source: self },
            SoVA1 { source: indices },
        )
    }
}

macro_rules! impl_gather_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, IndexSource> GatherInput<SoVA1<IndexSource>>
            for $input<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
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
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn gather_input(self, indices: SoVA1<IndexSource>) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                SoVA::validate(&indices)?;
                let $first_field = super::device_expr_gather::<$first, IndexSource>(
                    &self.$first_field,
                    &indices.source,
                )?;
                $(
                    let $field = super::device_expr_gather::<$rest, IndexSource>(
                        &self.$field,
                        &indices.source,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_gather_input!(SoVA2 -> SoA2<A, B> { left, right });
impl_gather_input!(SoA2 -> SoA2<A, B> { left, right });
impl_gather_input!(SoVA3 -> SoA3<A, B, C> { first, second, third });
impl_gather_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_gather_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_gather_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_gather_input_index_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource> GatherInput<IndexSource>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: GatherInput<SoVA1<IndexSource>>,
        {
            type Output = <Self as GatherInput<SoVA1<IndexSource>>>::Output;

            fn gather_input(self, indices: IndexSource) -> Result<Self::Output, Error> {
                <Self as GatherInput<SoVA1<IndexSource>>>::gather_input(
                    self,
                    SoVA1 { source: indices },
                )
            }
        }
    };
}

impl_gather_input_index_source!(SoVA2<A, B>);
impl_gather_input_index_source!(SoA2<A, B>);
impl_gather_input_index_source!(SoVA3<A, B, C>);
impl_gather_input_index_source!(SoA3<A, B, C>);
impl_gather_input_index_source!(SoVA4<A, B, C, D>);
impl_gather_input_index_source!(SoA4<A, B, C, D>);
impl_gather_input_index_source!(SoVA5<A, B, C, D, E>);
impl_gather_input_index_source!(SoA5<A, B, C, D, E>);
impl_gather_input_index_source!(SoVA6<A, B, C, D, E, F>);
impl_gather_input_index_source!(SoA6<A, B, C, D, E, F>);
impl_gather_input_index_source!(SoVA7<A, B, C, D, E, F, G>);
impl_gather_input_index_source!(SoA7<A, B, C, D, E, F, G>);
impl_gather_input_index_source!(SoVA8<A, B, C, D, E, F, G, H>);
impl_gather_input_index_source!(SoA8<A, B, C, D, E, F, G, H>);
impl_gather_input_index_source!(SoVA9<A, B, C, D, E, F, G, H, I>);
impl_gather_input_index_source!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_gather_input_index_source!(SoVA10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_input_index_source!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_input_index_source!(SoVA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_input_index_source!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_input_index_source!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_gather_input_index_source!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Input accepted by [`gather_if`].
#[doc(hidden)]
pub trait GatherIfInput<Indices, Stencil, Initial, Pred> {
    /// Output produced by gather-if.
    type Output;

    /// Gathers selected elements into a copy of `initial`.
    fn gather_if_input(
        self,
        indices: Indices,
        stencil: Stencil,
        initial: Initial,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<InputSource, IndexSource, StencilSource, InitialSource, Pred>
    GatherIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, SoVA1<InitialSource>, Pred>
    for SoVA1<InputSource>
where
    Self: SoVA<Item = InputSource::Item, Scalar = InputSource::Item>,
    SoVA1<IndexSource>: SoVA<Item = u32, Scalar = u32>,
    SoVA1<StencilSource>: SoVA<Item = StencilSource::Item, Scalar = StencilSource::Item>,
    SoVA1<InitialSource>: SoVA<Item = InputSource::Item, Scalar = InputSource::Item>,
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    StencilSource: KernelColumn<Runtime = InputSource::Runtime> + KernelColumnAt<S0>,
    InitialSource:
        KernelColumn<Runtime = InputSource::Runtime, Item = InputSource::Item> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    StencilSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    StencilSource::Expr: DeviceGpuExpr<StencilSource::Item>,
    InitialSource::Expr: DeviceGpuExpr<InputSource::Item>,
    Pred: PredicateOp<StencilSource::Item>,
{
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_if_input(
        self,
        indices: SoVA1<IndexSource>,
        stencil: SoVA1<StencilSource>,
        initial: SoVA1<InitialSource>,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        Ok(SoA1 {
            source: gather_if_one::<InputSource, IndexSource, StencilSource, InitialSource, Pred>(
                &self.source,
                &indices.source,
                &stencil.source,
                &initial.source,
            )?,
        })
    }
}

impl<InputSource, IndexSource, StencilSource, InitialSource, Pred>
    GatherIfInput<IndexSource, StencilSource, InitialSource, Pred> for InputSource
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    StencilSource: KernelColumn<Runtime = InputSource::Runtime> + KernelColumnAt<S0>,
    InitialSource:
        KernelColumn<Runtime = InputSource::Runtime, Item = InputSource::Item> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    StencilSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    StencilSource::Expr: DeviceGpuExpr<StencilSource::Item>,
    InitialSource::Expr: DeviceGpuExpr<InputSource::Item>,
    Pred: PredicateOp<StencilSource::Item>,
{
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_if_input(
        self,
        indices: IndexSource,
        stencil: StencilSource,
        initial: InitialSource,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<InputSource> as GatherIfInput<
            SoVA1<IndexSource>,
            SoVA1<StencilSource>,
            SoVA1<InitialSource>,
            Pred,
        >>::gather_if_input(
            SoVA1 { source: self },
            SoVA1 { source: indices },
            SoVA1 { source: stencil },
            SoVA1 { source: initial },
            pred,
        )
    }
}

impl<InputSource, IndexSource, StencilSource, R, T, Pred>
    GatherIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, DeviceVec<R, T>, Pred>
    for SoVA1<InputSource>
where
    SoVA1<InputSource>:
        GatherIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, SoA1<DeviceVec<R, T>>, Pred>,
    R: Runtime,
{
    type Output = <SoVA1<InputSource> as GatherIfInput<
        SoVA1<IndexSource>,
        SoVA1<StencilSource>,
        SoA1<DeviceVec<R, T>>,
        Pred,
    >>::Output;

    fn gather_if_input(
        self,
        indices: SoVA1<IndexSource>,
        stencil: SoVA1<StencilSource>,
        initial: DeviceVec<R, T>,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<InputSource> as GatherIfInput<
            SoVA1<IndexSource>,
            SoVA1<StencilSource>,
            SoA1<DeviceVec<R, T>>,
            Pred,
        >>::gather_if_input(self, indices, stencil, SoA1 { source: initial }, pred)
    }
}

macro_rules! impl_gather_if_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        #[allow(non_camel_case_types)]
        impl<$first, $( $rest ),+, IndexSource, StencilSource, InitialFirst, $( $field ),+, Pred>
            GatherIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, $output<InitialFirst, $( $field ),+>, Pred>
            for $input<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            IndexSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = u32> + KernelColumnAt<S0>,
            StencilSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            InitialFirst: KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$first as KernelColumn>::Item> + KernelColumnAt<S0>,
            $(
                $field: KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$rest as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            StencilSource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            IndexSource::Expr: DeviceGpuExpr<u32>,
            StencilSource::Expr: DeviceGpuExpr<StencilSource::Item>,
            InitialFirst::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$field as KernelColumn>::Expr: DeviceGpuExpr<<$field as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<StencilSource::Item>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn gather_if_input(
                self,
                indices: SoVA1<IndexSource>,
                stencil: SoVA1<StencilSource>,
                initial: $output<InitialFirst, $( $field ),+>,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                SoVA::validate(&indices)?;
                SoVA::validate(&stencil)?;
                let $first_field = gather_if_one::<
                    $first,
                    IndexSource,
                    StencilSource,
                    InitialFirst,
                    Pred,
                >(
                    &self.$first_field,
                    &indices.source,
                    &stencil.source,
                    &initial.$first_field,
                )?;
                $(
                    let $field = gather_if_one::<
                        $rest,
                        IndexSource,
                        StencilSource,
                        $field,
                        Pred,
                    >(
                        &self.$field,
                        &indices.source,
                        &stencil.source,
                        &initial.$field,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_gather_if_input!(SoVA2 -> SoA2<A, B> { left, right });
impl_gather_if_input!(SoA2 -> SoA2<A, B> { left, right });
impl_gather_if_input!(SoVA3 -> SoA3<A, B, C> { first, second, third });
impl_gather_if_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_gather_if_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_if_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_if_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_if_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_if_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_if_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_if_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_if_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_if_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_if_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_if_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_if_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_if_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_if_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_if_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_if_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_if_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_gather_if_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_gather_if_input_sources {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource, StencilSource, Initial, Pred>
            GatherIfInput<IndexSource, StencilSource, Initial, Pred>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            StencilSource: KernelColumn + KernelColumnAt<S0>,
            Self: GatherIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, Initial, Pred>,
        {
            type Output = <Self as GatherIfInput<
                SoVA1<IndexSource>,
                SoVA1<StencilSource>,
                Initial,
                Pred,
            >>::Output;

            fn gather_if_input(
                self,
                indices: IndexSource,
                stencil: StencilSource,
                initial: Initial,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <Self as GatherIfInput<
                    SoVA1<IndexSource>,
                    SoVA1<StencilSource>,
                    Initial,
                    Pred,
                >>::gather_if_input(
                    self,
                    SoVA1 { source: indices },
                    SoVA1 { source: stencil },
                    initial,
                    pred,
                )
            }
        }
    };
}

impl_gather_if_input_sources!(SoVA2<A, B>);
impl_gather_if_input_sources!(SoA2<A, B>);
impl_gather_if_input_sources!(SoVA3<A, B, C>);
impl_gather_if_input_sources!(SoA3<A, B, C>);
impl_gather_if_input_sources!(SoVA4<A, B, C, D>);
impl_gather_if_input_sources!(SoA4<A, B, C, D>);
impl_gather_if_input_sources!(SoVA5<A, B, C, D, E>);
impl_gather_if_input_sources!(SoA5<A, B, C, D, E>);
impl_gather_if_input_sources!(SoVA6<A, B, C, D, E, F>);
impl_gather_if_input_sources!(SoA6<A, B, C, D, E, F>);
impl_gather_if_input_sources!(SoVA7<A, B, C, D, E, F, G>);
impl_gather_if_input_sources!(SoA7<A, B, C, D, E, F, G>);
impl_gather_if_input_sources!(SoVA8<A, B, C, D, E, F, G, H>);
impl_gather_if_input_sources!(SoA8<A, B, C, D, E, F, G, H>);
impl_gather_if_input_sources!(SoVA9<A, B, C, D, E, F, G, H, I>);
impl_gather_if_input_sources!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_gather_if_input_sources!(SoVA10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_if_input_sources!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_if_input_sources!(SoVA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_if_input_sources!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_if_input_sources!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_gather_if_input_sources!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Input accepted by [`scatter`].
#[doc(hidden)]
pub trait ScatterInput<Indices, Initial> {
    /// Output produced by scatter.
    type Output;

    /// Scatters `self[i]` into a copy of `initial[indices[i]]`.
    fn scatter_input(self, indices: Indices, initial: Initial) -> Result<Self::Output, Error>;
}

impl<ValueSource, IndexSource, InitialSource> ScatterInput<SoVA1<IndexSource>, SoVA1<InitialSource>>
    for SoVA1<ValueSource>
where
    Self: SoVA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    SoVA1<IndexSource>: SoVA<Item = u32, Scalar = u32>,
    SoVA1<InitialSource>: SoVA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InitialSource: StorageKernelColumn<Runtime = ValueSource::Runtime, Item = ValueSource::Item>
        + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
    InitialSource::Expr: DeviceGpuExpr<ValueSource::Item>,
{
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_input(
        self,
        indices: SoVA1<IndexSource>,
        initial: SoVA1<InitialSource>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        SoVA::validate(&indices)?;
        SoVA::validate(&initial)?;
        Ok(SoA1 {
            source: super::device_expr_scatter::<ValueSource, IndexSource, InitialSource>(
                &self.source,
                &indices.source,
                &initial.source,
            )?,
        })
    }
}

impl<ValueSource, IndexSource, InitialSource> ScatterInput<IndexSource, InitialSource>
    for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InitialSource:
        KernelColumn<Runtime = ValueSource::Runtime, Item = ValueSource::Item> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
    InitialSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    SoVA1<ValueSource>: ScatterInput<
            SoVA1<IndexSource>,
            SoVA1<InitialSource>,
            Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>,
        >,
{
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_input(
        self,
        indices: IndexSource,
        initial: InitialSource,
    ) -> Result<Self::Output, Error> {
        <SoVA1<ValueSource> as ScatterInput<SoVA1<IndexSource>, SoVA1<InitialSource>>>::scatter_input(
            SoVA1 { source: self },
            SoVA1 { source: indices },
            SoVA1 { source: initial },
        )
    }
}

impl<ValueSource, IndexSource, R, T> ScatterInput<SoVA1<IndexSource>, DeviceVec<R, T>>
    for SoVA1<ValueSource>
where
    SoVA1<ValueSource>: ScatterInput<SoVA1<IndexSource>, SoA1<DeviceVec<R, T>>>,
    R: Runtime,
{
    type Output =
        <SoVA1<ValueSource> as ScatterInput<SoVA1<IndexSource>, SoA1<DeviceVec<R, T>>>>::Output;

    fn scatter_input(
        self,
        indices: SoVA1<IndexSource>,
        initial: DeviceVec<R, T>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<ValueSource> as ScatterInput<SoVA1<IndexSource>, SoA1<DeviceVec<R, T>>>>::scatter_input(
            self,
            indices,
            SoA1 { source: initial },
        )
    }
}

macro_rules! impl_scatter_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        #[allow(non_camel_case_types)]
        impl<$first, $( $rest ),+, IndexSource, InitialFirst, $( $field ),+>
            ScatterInput<SoVA1<IndexSource>, $output<InitialFirst, $( $field ),+>>
            for $input<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            IndexSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = u32> + KernelColumnAt<S0>,
            InitialFirst: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$first as KernelColumn>::Item> + KernelColumnAt<S0>,
            $(
                $field: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$rest as KernelColumn>::Item>
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
            IndexSource::Expr: GpuExpr<u32>,
            InitialFirst::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$field as KernelColumn>::Expr: DeviceGpuExpr<<$field as KernelColumn>::Item>,
            )+
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn scatter_input(
                self,
                indices: SoVA1<IndexSource>,
                initial: $output<InitialFirst, $( $field ),+>,
            ) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                SoVA::validate(&indices)?;
                let $first_field = super::device_expr_scatter::<$first, IndexSource, InitialFirst>(
                    &self.$first_field,
                    &indices.source,
                    &initial.$first_field,
                )?;
                $(
                    let $field = super::device_expr_scatter::<$rest, IndexSource, $field>(
                        &self.$field,
                        &indices.source,
                        &initial.$field,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_scatter_input!(SoVA2 -> SoA2<A, B> { left, right });
impl_scatter_input!(SoA2 -> SoA2<A, B> { left, right });
impl_scatter_input!(SoVA3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_scatter_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_scatter_input_index_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource, Initial> ScatterInput<IndexSource, Initial>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: ScatterInput<SoVA1<IndexSource>, Initial>,
        {
            type Output = <Self as ScatterInput<SoVA1<IndexSource>, Initial>>::Output;

            fn scatter_input(
                self,
                indices: IndexSource,
                initial: Initial,
            ) -> Result<Self::Output, Error> {
                <Self as ScatterInput<SoVA1<IndexSource>, Initial>>::scatter_input(
                    self,
                    SoVA1 { source: indices },
                    initial,
                )
            }
        }
    };
}

impl_scatter_input_index_source!(SoVA2<A, B>);
impl_scatter_input_index_source!(SoA2<A, B>);
impl_scatter_input_index_source!(SoVA3<A, B, C>);
impl_scatter_input_index_source!(SoA3<A, B, C>);
impl_scatter_input_index_source!(SoVA4<A, B, C, D>);
impl_scatter_input_index_source!(SoA4<A, B, C, D>);
impl_scatter_input_index_source!(SoVA5<A, B, C, D, E>);
impl_scatter_input_index_source!(SoA5<A, B, C, D, E>);
impl_scatter_input_index_source!(SoVA6<A, B, C, D, E, F>);
impl_scatter_input_index_source!(SoA6<A, B, C, D, E, F>);
impl_scatter_input_index_source!(SoVA7<A, B, C, D, E, F, G>);
impl_scatter_input_index_source!(SoA7<A, B, C, D, E, F, G>);
impl_scatter_input_index_source!(SoVA8<A, B, C, D, E, F, G, H>);
impl_scatter_input_index_source!(SoA8<A, B, C, D, E, F, G, H>);
impl_scatter_input_index_source!(SoVA9<A, B, C, D, E, F, G, H, I>);
impl_scatter_input_index_source!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_scatter_input_index_source!(SoVA10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_input_index_source!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_input_index_source!(SoVA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_input_index_source!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_input_index_source!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_scatter_input_index_source!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Input accepted by [`scatter_if`].
#[doc(hidden)]
pub trait ScatterIfInput<Indices, Stencil, Initial, Pred> {
    /// Output produced by scatter-if.
    type Output;

    /// Scatters selected values into a copy of `initial[indices[i]]`.
    fn scatter_if_input(
        self,
        indices: Indices,
        stencil: Stencil,
        initial: Initial,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<ValueSource, IndexSource, StencilSource, InitialSource, Pred>
    ScatterIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, SoVA1<InitialSource>, Pred>
    for SoVA1<ValueSource>
where
    Self: SoVA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    SoVA1<IndexSource>: SoVA<Item = u32, Scalar = u32>,
    SoVA1<StencilSource>: SoVA<Item = StencilSource::Item, Scalar = StencilSource::Item>,
    SoVA1<InitialSource>: SoVA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    StencilSource: KernelColumn<Runtime = ValueSource::Runtime> + KernelColumnAt<S0>,
    InitialSource: StorageKernelColumn<Runtime = ValueSource::Runtime, Item = ValueSource::Item>
        + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    StencilSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
    StencilSource::Expr: GpuExpr<StencilSource::Item>,
    InitialSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Pred: PredicateOp<StencilSource::Item>,
{
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_if_input(
        self,
        indices: SoVA1<IndexSource>,
        stencil: SoVA1<StencilSource>,
        initial: SoVA1<InitialSource>,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        SoVA::validate(&indices)?;
        SoVA::validate(&stencil)?;
        SoVA::validate(&initial)?;
        Ok(SoA1 {
            source: super::device_expr_scatter_if::<
                ValueSource,
                IndexSource,
                StencilSource,
                InitialSource,
                Pred,
            >(
                &self.source,
                &indices.source,
                &stencil.source,
                &initial.source,
            )?,
        })
    }
}

impl<ValueSource, IndexSource, StencilSource, InitialSource, Pred>
    ScatterIfInput<IndexSource, StencilSource, InitialSource, Pred> for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    StencilSource: KernelColumn<Runtime = ValueSource::Runtime> + KernelColumnAt<S0>,
    InitialSource:
        KernelColumn<Runtime = ValueSource::Runtime, Item = ValueSource::Item> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    StencilSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
    StencilSource::Expr: GpuExpr<StencilSource::Item>,
    InitialSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Pred: PredicateOp<StencilSource::Item>,
    SoVA1<ValueSource>: ScatterIfInput<
            SoVA1<IndexSource>,
            SoVA1<StencilSource>,
            SoVA1<InitialSource>,
            Pred,
            Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>,
        >,
{
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_if_input(
        self,
        indices: IndexSource,
        stencil: StencilSource,
        initial: InitialSource,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<ValueSource> as ScatterIfInput<
            SoVA1<IndexSource>,
            SoVA1<StencilSource>,
            SoVA1<InitialSource>,
            Pred,
        >>::scatter_if_input(
            SoVA1 { source: self },
            SoVA1 { source: indices },
            SoVA1 { source: stencil },
            SoVA1 { source: initial },
            pred,
        )
    }
}

impl<ValueSource, IndexSource, StencilSource, R, T, Pred>
    ScatterIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, DeviceVec<R, T>, Pred>
    for SoVA1<ValueSource>
where
    SoVA1<ValueSource>:
        ScatterIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, SoA1<DeviceVec<R, T>>, Pred>,
    R: Runtime,
{
    type Output = <SoVA1<ValueSource> as ScatterIfInput<
        SoVA1<IndexSource>,
        SoVA1<StencilSource>,
        SoA1<DeviceVec<R, T>>,
        Pred,
    >>::Output;

    fn scatter_if_input(
        self,
        indices: SoVA1<IndexSource>,
        stencil: SoVA1<StencilSource>,
        initial: DeviceVec<R, T>,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<ValueSource> as ScatterIfInput<
            SoVA1<IndexSource>,
            SoVA1<StencilSource>,
            SoA1<DeviceVec<R, T>>,
            Pred,
        >>::scatter_if_input(self, indices, stencil, SoA1 { source: initial }, pred)
    }
}

macro_rules! impl_scatter_if_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        #[allow(non_camel_case_types)]
        impl<$first, $( $rest ),+, IndexSource, StencilSource, InitialFirst, $( $field ),+, Pred>
            ScatterIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, $output<InitialFirst, $( $field ),+>, Pred>
            for $input<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            IndexSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = u32> + KernelColumnAt<S0>,
            StencilSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            InitialFirst: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$first as KernelColumn>::Item> + KernelColumnAt<S0>,
            $(
                $field: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$rest as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            StencilSource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: GpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: GpuExpr<<$rest as KernelColumn>::Item>,
            )+
            IndexSource::Expr: GpuExpr<u32>,
            StencilSource::Expr: GpuExpr<StencilSource::Item>,
            InitialFirst::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$field as KernelColumn>::Expr: DeviceGpuExpr<<$field as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<StencilSource::Item>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn scatter_if_input(
                self,
                indices: SoVA1<IndexSource>,
                stencil: SoVA1<StencilSource>,
                initial: $output<InitialFirst, $( $field ),+>,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                SoVA::validate(&indices)?;
                SoVA::validate(&stencil)?;
                let $first_field = super::device_expr_scatter_if::<
                    $first,
                    IndexSource,
                    StencilSource,
                    InitialFirst,
                    Pred,
                >(
                    &self.$first_field,
                    &indices.source,
                    &stencil.source,
                    &initial.$first_field,
                )?;
                $(
                    let $field = super::device_expr_scatter_if::<
                        $rest,
                        IndexSource,
                        StencilSource,
                        $field,
                        Pred,
                    >(
                        &self.$field,
                        &indices.source,
                        &stencil.source,
                        &initial.$field,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_scatter_if_input!(SoVA2 -> SoA2<A, B> { left, right });
impl_scatter_if_input!(SoA2 -> SoA2<A, B> { left, right });
impl_scatter_if_input!(SoVA3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_if_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_if_input!(SoVA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_if_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_if_input!(SoVA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_if_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_if_input!(SoVA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_if_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_if_input!(SoVA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_if_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_if_input!(SoVA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_if_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_if_input!(SoVA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_if_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_if_input!(SoVA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_if_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_if_input!(SoVA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_if_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_if_input!(SoVA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_scatter_if_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_scatter_if_input_sources {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource, StencilSource, Initial, Pred>
            ScatterIfInput<IndexSource, StencilSource, Initial, Pred>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            StencilSource: KernelColumn + KernelColumnAt<S0>,
            Self: ScatterIfInput<SoVA1<IndexSource>, SoVA1<StencilSource>, Initial, Pred>,
        {
            type Output = <Self as ScatterIfInput<
                SoVA1<IndexSource>,
                SoVA1<StencilSource>,
                Initial,
                Pred,
            >>::Output;

            fn scatter_if_input(
                self,
                indices: IndexSource,
                stencil: StencilSource,
                initial: Initial,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <Self as ScatterIfInput<
                    SoVA1<IndexSource>,
                    SoVA1<StencilSource>,
                    Initial,
                    Pred,
                >>::scatter_if_input(
                    self,
                    SoVA1 { source: indices },
                    SoVA1 { source: stencil },
                    initial,
                    pred,
                )
            }
        }
    };
}

impl_scatter_if_input_sources!(SoVA2<A, B>);
impl_scatter_if_input_sources!(SoA2<A, B>);
impl_scatter_if_input_sources!(SoVA3<A, B, C>);
impl_scatter_if_input_sources!(SoA3<A, B, C>);
impl_scatter_if_input_sources!(SoVA4<A, B, C, D>);
impl_scatter_if_input_sources!(SoA4<A, B, C, D>);
impl_scatter_if_input_sources!(SoVA5<A, B, C, D, E>);
impl_scatter_if_input_sources!(SoA5<A, B, C, D, E>);
impl_scatter_if_input_sources!(SoVA6<A, B, C, D, E, F>);
impl_scatter_if_input_sources!(SoA6<A, B, C, D, E, F>);
impl_scatter_if_input_sources!(SoVA7<A, B, C, D, E, F, G>);
impl_scatter_if_input_sources!(SoA7<A, B, C, D, E, F, G>);
impl_scatter_if_input_sources!(SoVA8<A, B, C, D, E, F, G, H>);
impl_scatter_if_input_sources!(SoA8<A, B, C, D, E, F, G, H>);
impl_scatter_if_input_sources!(SoVA9<A, B, C, D, E, F, G, H, I>);
impl_scatter_if_input_sources!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_scatter_if_input_sources!(SoVA10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_if_input_sources!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_if_input_sources!(SoVA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_if_input_sources!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_if_input_sources!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_scatter_if_input_sources!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Gathers `input[indices[i]]` into new owned device storage.
///
/// This is a borrowing algorithm: `input` and `indices` are read-only. For
/// multiple value columns, pass a borrowed SoA built with [`zip`](crate::zip).
pub fn gather<Input, Indices>(
    input: Input,
    indices: Indices,
) -> Result<<<Input as GatherInput<Indices>>::Output as MaterializeOutput>::Output, Error>
where
    Input: GatherInput<Indices>,
    <Input as GatherInput<Indices>>::Output: MaterializeOutput,
{
    materialize(input.gather_input(indices)?)
}

/// Gathers selected elements into a copy of `initial`.
///
/// This is a mixed algorithm: `input`, `indices`, and `stencil` are read-only,
/// while `initial` is owned output storage that is consumed and copied before
/// selected positions are overwritten.
pub fn gather_if<Input, Indices, Stencil, Initial, Pred>(
    input: Input,
    indices: Indices,
    stencil: Stencil,
    initial: Initial,
    _pred: Pred,
) -> Result<
    <<Input as GatherIfInput<Indices, Stencil, Initial, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Input: GatherIfInput<Indices, Stencil, Initial, Pred>,
    <Input as GatherIfInput<Indices, Stencil, Initial, Pred>>::Output: MaterializeOutput,
{
    materialize(input.gather_if_input(indices, stencil, initial, GpuOp::<Pred>::new())?)
}

/// Scatters `values[i]` into a copy of `initial[indices[i]]`.
///
/// This is a mixed algorithm: `values` and `indices` are read-only, while
/// `initial` is owned output storage. For multiple value columns, pass
/// `zip(...)` for `values` and `initial`.
pub fn scatter<Values, Indices, Initial>(
    values: Values,
    indices: Indices,
    initial: Initial,
) -> Result<<<Values as ScatterInput<Indices, Initial>>::Output as MaterializeOutput>::Output, Error>
where
    Values: ScatterInput<Indices, Initial>,
    <Values as ScatterInput<Indices, Initial>>::Output: MaterializeOutput,
{
    materialize(values.scatter_input(indices, initial)?)
}

/// Scatters selected values into a copy of `initial[indices[i]]`.
///
/// This is a mixed algorithm: `values`, `indices`, and `stencil` are read-only,
/// while `initial` is owned output storage.
pub fn scatter_if<Values, Indices, Stencil, Initial, Pred>(
    values: Values,
    indices: Indices,
    stencil: Stencil,
    initial: Initial,
    _pred: Pred,
) -> Result<
    <<Values as ScatterIfInput<Indices, Stencil, Initial, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Values: ScatterIfInput<Indices, Stencil, Initial, Pred>,
    <Values as ScatterIfInput<Indices, Stencil, Initial, Pred>>::Output: MaterializeOutput,
{
    materialize(values.scatter_if_input(indices, stencil, initial, GpuOp::<Pred>::new())?)
}
