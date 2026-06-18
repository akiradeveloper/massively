use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA1, SoA2, SoA3, SoA4, SoA5,
        SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoAView1, SoAView2, SoAView3, SoAView4,
        SoAView5, SoAView6, SoAView7, SoAView8, SoAView9, SoAView10, SoAView11, SoAView12,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::{GpuOp, PredicateOp},
    primitives::range as primitive_range,
};
use cubecl::prelude::*;

const BLOCK_API_SIZE: u32 = 256;

fn gather_if_one<InputSource, IndexSource, Stencil, Pred>(
    input: &InputSource,
    indices: &IndexSource,
    stencil: &Stencil,
) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = InputSource::Runtime> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement + Default,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    input.validate()?;
    indices.validate()?;
    stencil.validate()?;

    let input = super::device_expr_collect(input)?;
    let indices = super::device_expr_collect(indices)?;
    super::ensure_same_len(indices.len, stencil.len())?;
    let flags = super::device_expr_selection_handles::<Stencil, Pred>(stencil, false)?;

    let output =
        primitive_range::filled(input.policy(), indices.len, InputSource::Item::default())?;
    let num_blocks = indices.len.div_ceil(BLOCK_API_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = input.policy.client();

    if indices.len != 0 {
        unsafe {
            gather_if_flags_kernel::launch_unchecked::<InputSource::Item, InputSource::Runtime>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_API_SIZE),
                ArrayArg::from_raw_parts::<InputSource::Item>(&input.handle, input.len, 1),
                ArrayArg::from_raw_parts::<u32>(&indices.handle, indices.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flags.flag, flags.len, 1),
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

impl<InputSource, IndexSource> GatherInput<SoAView1<IndexSource>> for SoAView1<InputSource>
where
    Self: ReadOnlySoA<Item = InputSource::Item, Scalar = InputSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = u32, Scalar = u32>,
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_input(self, indices: SoAView1<IndexSource>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&indices)?;
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
        <SoAView1<InputSource> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView1 { source: self },
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
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn gather_input(self, indices: SoAView1<IndexSource>) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
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

impl_gather_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_gather_input!(SoA2 -> SoA2<A, B> { left, right });
impl_gather_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_gather_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_gather_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_gather_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_gather_input_index_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource> GatherInput<IndexSource>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: GatherInput<SoAView1<IndexSource>>,
        {
            type Output = <Self as GatherInput<SoAView1<IndexSource>>>::Output;

            fn gather_input(self, indices: IndexSource) -> Result<Self::Output, Error> {
                <Self as GatherInput<SoAView1<IndexSource>>>::gather_input(
                    self,
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
impl_gather_input_index_source!(SoAView4<A, B, C, D>);
impl_gather_input_index_source!(SoA4<A, B, C, D>);
impl_gather_input_index_source!(SoAView5<A, B, C, D, E>);
impl_gather_input_index_source!(SoA5<A, B, C, D, E>);
impl_gather_input_index_source!(SoAView6<A, B, C, D, E, F>);
impl_gather_input_index_source!(SoA6<A, B, C, D, E, F>);
impl_gather_input_index_source!(SoAView7<A, B, C, D, E, F, G>);
impl_gather_input_index_source!(SoA7<A, B, C, D, E, F, G>);
impl_gather_input_index_source!(SoAView8<A, B, C, D, E, F, G, H>);
impl_gather_input_index_source!(SoA8<A, B, C, D, E, F, G, H>);
impl_gather_input_index_source!(SoAView9<A, B, C, D, E, F, G, H, I>);
impl_gather_input_index_source!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_gather_input_index_source!(SoAView10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_input_index_source!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_input_index_source!(SoAView11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_input_index_source!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_input_index_source!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_gather_input_index_source!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Input accepted by [`gather_if`].
#[doc(hidden)]
pub trait GatherIfInput<Indices, Stencil, Pred> {
    /// Output produced by gather-if.
    type Output;

    /// Gathers selected elements into default-initialized output.
    fn gather_if_input(
        self,
        indices: Indices,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<InputSource, IndexSource, Stencil, Pred> GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>
    for SoAView1<InputSource>
where
    Self: ReadOnlySoA<Item = InputSource::Item, Scalar = InputSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = u32, Scalar = u32>,
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = InputSource::Runtime> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement + Default,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_if_input(
        self,
        indices: SoAView1<IndexSource>,
        stencil: Stencil,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        Ok(SoA1 {
            source: gather_if_one::<InputSource, IndexSource, Stencil, Pred>(
                &self.source,
                &indices.source,
                &stencil,
            )?,
        })
    }
}

impl<InputSource, IndexSource, Stencil, Pred> GatherIfInput<IndexSource, Stencil, Pred>
    for InputSource
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = InputSource::Runtime> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement + Default,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_if_input(
        self,
        indices: IndexSource,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
            SoAView1 { source: self },
            SoAView1 { source: indices },
            stencil,
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
            Stencil: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement + Default,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement + Default,
            )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            IndexSource::Expr: DeviceGpuExpr<u32>,
            Stencil::Item: CubePrimitive + CubeElement,
            Stencil::Expr: GpuExpr<Stencil::Item>,
            Pred: PredicateOp<Stencil::Item>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn gather_if_input(
                self,
                indices: SoAView1<IndexSource>,
                stencil: Stencil,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let $first_field = gather_if_one::<$first, IndexSource, Stencil, Pred>(
                    &self.$first_field,
                    &indices.source,
                    &stencil,
                )?;
                $(
                    let $field = gather_if_one::<$rest, IndexSource, Stencil, Pred>(
                        &self.$field,
                        &indices.source,
                        &stencil,
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
impl_gather_if_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_if_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_gather_if_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_if_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_gather_if_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_if_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_gather_if_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_if_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_gather_if_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_if_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_gather_if_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_if_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_gather_if_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_if_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_gather_if_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_if_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_gather_if_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_gather_if_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_gather_if_input_sources {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource, Stencil, Pred>
            GatherIfInput<IndexSource, Stencil, Pred>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>,
        {
            type Output = <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;

            fn gather_if_input(
                self,
                indices: IndexSource,
                stencil: Stencil,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
                    self,
                    SoAView1 { source: indices },
                    stencil,
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
impl_gather_if_input_sources!(SoAView4<A, B, C, D>);
impl_gather_if_input_sources!(SoA4<A, B, C, D>);
impl_gather_if_input_sources!(SoAView5<A, B, C, D, E>);
impl_gather_if_input_sources!(SoA5<A, B, C, D, E>);
impl_gather_if_input_sources!(SoAView6<A, B, C, D, E, F>);
impl_gather_if_input_sources!(SoA6<A, B, C, D, E, F>);
impl_gather_if_input_sources!(SoAView7<A, B, C, D, E, F, G>);
impl_gather_if_input_sources!(SoA7<A, B, C, D, E, F, G>);
impl_gather_if_input_sources!(SoAView8<A, B, C, D, E, F, G, H>);
impl_gather_if_input_sources!(SoA8<A, B, C, D, E, F, G, H>);
impl_gather_if_input_sources!(SoAView9<A, B, C, D, E, F, G, H, I>);
impl_gather_if_input_sources!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_gather_if_input_sources!(SoAView10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_if_input_sources!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_gather_if_input_sources!(SoAView11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_if_input_sources!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_gather_if_input_sources!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_gather_if_input_sources!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

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

/// Gathers selected elements into default-initialized output.
///
/// This is a borrowing algorithm: `input` and `indices` are read-only.
pub fn gather_if<Input, Indices, Stencil, Pred>(
    input: Input,
    indices: Indices,
    stencil: Stencil,
    _pred: Pred,
) -> Result<
    <<Input as GatherIfInput<Indices, Stencil, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Input: GatherIfInput<Indices, Stencil, Pred>,
    <Input as GatherIfInput<Indices, Stencil, Pred>>::Output: MaterializeOutput,
{
    materialize(input.gather_if_input(indices, stencil, GpuOp::<Pred>::new())?)
}
