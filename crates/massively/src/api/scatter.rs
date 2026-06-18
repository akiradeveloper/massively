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
    primitives::range,
};
use cubecl::prelude::*;

fn scatter_one<ValueSource, IndexSource>(
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
    let initial = range::filled(values.policy(), len, default)?;
    super::device_expr_scatter::<
        ValueSource,
        IndexSource,
        DeviceVec<ValueSource::Runtime, ValueSource::Item>,
    >(values, indices, &initial)
}

fn scatter_if_one<ValueSource, IndexSource, Stencil, Pred>(
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
    Stencil: KernelColumn<Runtime = ValueSource::Runtime> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    values.validate()?;
    indices.validate()?;
    stencil.validate()?;
    super::ensure_same_len(values.len(), indices.len())?;
    super::ensure_same_len(values.len(), stencil.len())?;
    let values = super::device_expr_collect(values)?;
    let indices = super::device_expr_collect(indices)?;
    let flags = super::device_expr_selection_handles::<Stencil, Pred>(stencil, false)?;
    let initial = range::filled(values.policy(), len, default)?;
    let block_count = values.len.div_ceil(256);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    if values.len != 0 {
        unsafe {
            scatter_if_flags_kernel::launch_unchecked::<ValueSource::Item, ValueSource::Runtime>(
                values.policy().client(),
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts::<ValueSource::Item>(&values.handle, values.len, 1),
                ArrayArg::from_raw_parts::<u32>(&indices.handle, indices.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flags.flag, flags.len, 1),
                ArrayArg::from_raw_parts::<ValueSource::Item>(&initial.handle, initial.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }
    Ok(initial)
}

/// Input accepted by [`scatter`].
#[doc(hidden)]
pub trait ScatterInput<Indices> {
    /// Default value accepted by scatter.
    type Default;
    /// Output produced by scatter.
    type Output;

    /// Scatters `self[i]` into default-initialized output at `indices[i]`.
    fn scatter_input(
        self,
        indices: Indices,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error>;
}

impl<ValueSource, IndexSource> ScatterInput<SoAView1<IndexSource>> for SoAView1<ValueSource>
where
    Self: ReadOnlySoA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = u32, Scalar = u32>,
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_input(
        self,
        indices: SoAView1<IndexSource>,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&indices)?;
        Ok(SoA1 {
            source: scatter_one::<ValueSource, IndexSource>(
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
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_input(
        self,
        indices: IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        Ok(SoA1 {
            source: scatter_one::<ValueSource, IndexSource>(&self, &indices, len, default)?,
        })
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
                indices: SoAView1<IndexSource>,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let ($first_field, $( $field ),+) = default;
                let $first_field = scatter_one::<$first, IndexSource>(
                    &self.$first_field,
                    &indices.source,
                    len,
                    $first_field,
                )?;
                $(
                    let $field = scatter_one::<$rest, IndexSource>(
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
impl_scatter_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_scatter_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_scatter_input_index_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource> ScatterInput<IndexSource>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: ScatterInput<SoAView1<IndexSource>>,
        {
            type Default = <Self as ScatterInput<SoAView1<IndexSource>>>::Default;
            type Output = <Self as ScatterInput<SoAView1<IndexSource>>>::Output;

            fn scatter_input(
                self,
                indices: IndexSource,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                <Self as ScatterInput<SoAView1<IndexSource>>>::scatter_input(
                    self,
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
impl_scatter_input_index_source!(SoAView4<A, B, C, D>);
impl_scatter_input_index_source!(SoA4<A, B, C, D>);
impl_scatter_input_index_source!(SoAView5<A, B, C, D, E>);
impl_scatter_input_index_source!(SoA5<A, B, C, D, E>);
impl_scatter_input_index_source!(SoAView6<A, B, C, D, E, F>);
impl_scatter_input_index_source!(SoA6<A, B, C, D, E, F>);
impl_scatter_input_index_source!(SoAView7<A, B, C, D, E, F, G>);
impl_scatter_input_index_source!(SoA7<A, B, C, D, E, F, G>);
impl_scatter_input_index_source!(SoAView8<A, B, C, D, E, F, G, H>);
impl_scatter_input_index_source!(SoA8<A, B, C, D, E, F, G, H>);
impl_scatter_input_index_source!(SoAView9<A, B, C, D, E, F, G, H, I>);
impl_scatter_input_index_source!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_scatter_input_index_source!(SoAView10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_input_index_source!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_input_index_source!(SoAView11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_input_index_source!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_input_index_source!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_scatter_input_index_source!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Input accepted by [`scatter_if`].
#[doc(hidden)]
pub trait ScatterIfInput<Indices, Stencil, Pred> {
    /// Default value accepted by scatter-if.
    type Default;
    /// Output produced by scatter-if.
    type Output;

    /// Scatters selected values into default-initialized output at `indices[i]`.
    fn scatter_if_input(
        self,
        indices: Indices,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<ValueSource, IndexSource, Stencil, Pred> ScatterIfInput<SoAView1<IndexSource>, Stencil, Pred>
    for SoAView1<ValueSource>
where
    Self: ReadOnlySoA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    SoAView1<IndexSource>: ReadOnlySoA<Item = u32, Scalar = u32>,
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = ValueSource::Runtime> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_if_input(
        self,
        indices: SoAView1<IndexSource>,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&indices)?;
        Ok(SoA1 {
            source: scatter_if_one::<ValueSource, IndexSource, Stencil, Pred>(
                &self.source,
                &indices.source,
                &stencil,
                len,
                default,
            )?,
        })
    }
}

impl<ValueSource, IndexSource, Stencil, Pred> ScatterIfInput<IndexSource, Stencil, Pred>
    for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = ValueSource::Runtime> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Default = ValueSource::Item;
    type Output = SoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_if_input(
        self,
        indices: IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        let _ = pred;
        Ok(SoA1 {
            source: scatter_if_one::<ValueSource, IndexSource, Stencil, Pred>(
                &self, &indices, &stencil, len, default,
            )?,
        })
    }
}

macro_rules! impl_scatter_if_input {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, IndexSource, Stencil, Pred> ScatterIfInput<SoAView1<IndexSource>, Stencil, Pred>
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
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
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
            type Default = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn scatter_if_input(
                self,
                indices: SoAView1<IndexSource>,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let ($first_field, $( $field ),+) = default;
                let $first_field = scatter_if_one::<$first, IndexSource, Stencil, Pred>(
                    &self.$first_field,
                    &indices.source,
                    &stencil,
                    len,
                    $first_field,
                )?;
                $(
                    let $field = scatter_if_one::<$rest, IndexSource, Stencil, Pred>(
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

impl_scatter_if_input!(SoAView2 -> SoA2<A, B> { left, right });
impl_scatter_if_input!(SoA2 -> SoA2<A, B> { left, right });
impl_scatter_if_input!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_if_input!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_scatter_if_input!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_if_input!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_scatter_if_input!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_if_input!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_scatter_if_input!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_if_input!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_scatter_if_input!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_if_input!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_scatter_if_input!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_if_input!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_scatter_if_input!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_if_input!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_scatter_if_input!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_if_input!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_scatter_if_input!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_if_input!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_scatter_if_input!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_scatter_if_input!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_scatter_if_input_sources {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<$( $field_ty ),+, IndexSource, Stencil, Pred> ScatterIfInput<IndexSource, Stencil, Pred>
            for $name<$( $field_ty ),+>
        where
            IndexSource: KernelColumn + KernelColumnAt<S0>,
            Self: ScatterIfInput<SoAView1<IndexSource>, Stencil, Pred>,
        {
            type Default = <Self as ScatterIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;
            type Output = <Self as ScatterIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;

            fn scatter_if_input(
                self,
                indices: IndexSource,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <Self as ScatterIfInput<SoAView1<IndexSource>, Stencil, Pred>>::scatter_if_input(
                    self,
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

impl_scatter_if_input_sources!(SoAView2<A, B>);
impl_scatter_if_input_sources!(SoA2<A, B>);
impl_scatter_if_input_sources!(SoAView3<A, B, C>);
impl_scatter_if_input_sources!(SoA3<A, B, C>);
impl_scatter_if_input_sources!(SoAView4<A, B, C, D>);
impl_scatter_if_input_sources!(SoA4<A, B, C, D>);
impl_scatter_if_input_sources!(SoAView5<A, B, C, D, E>);
impl_scatter_if_input_sources!(SoA5<A, B, C, D, E>);
impl_scatter_if_input_sources!(SoAView6<A, B, C, D, E, F>);
impl_scatter_if_input_sources!(SoA6<A, B, C, D, E, F>);
impl_scatter_if_input_sources!(SoAView7<A, B, C, D, E, F, G>);
impl_scatter_if_input_sources!(SoA7<A, B, C, D, E, F, G>);
impl_scatter_if_input_sources!(SoAView8<A, B, C, D, E, F, G, H>);
impl_scatter_if_input_sources!(SoA8<A, B, C, D, E, F, G, H>);
impl_scatter_if_input_sources!(SoAView9<A, B, C, D, E, F, G, H, I>);
impl_scatter_if_input_sources!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_scatter_if_input_sources!(SoAView10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_if_input_sources!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_scatter_if_input_sources!(SoAView11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_if_input_sources!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_scatter_if_input_sources!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L>);
impl_scatter_if_input_sources!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Scatters `values[i]` into a new output at `indices[i]`.
///
/// The output is allocated with `len` elements, initialized with `default`, and
/// then updated by the scatter. For multiple value columns, pass `zip(...)` for
/// `values` and a tuple for `default`.
pub fn scatter<Values, Indices>(
    values: Values,
    indices: Indices,
    len: usize,
    default: <Values as ScatterInput<Indices>>::Default,
) -> Result<<<Values as ScatterInput<Indices>>::Output as MaterializeOutput>::Output, Error>
where
    Values: ScatterInput<Indices>,
    <Values as ScatterInput<Indices>>::Output: MaterializeOutput,
{
    materialize(values.scatter_input(indices, len, default)?)
}

/// Scatters selected values into a new output at `indices[i]`.
///
/// The output is allocated with `len` elements, initialized with `default`, and
/// then updated for values satisfying `Pred`.
pub fn scatter_if<Values, Indices, Stencil, Pred>(
    values: Values,
    indices: Indices,
    len: usize,
    default: <Values as ScatterIfInput<Indices, Stencil, Pred>>::Default,
    stencil: Stencil,
    _pred: Pred,
) -> Result<
    <<Values as ScatterIfInput<Indices, Stencil, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Values: ScatterIfInput<Indices, Stencil, Pred>,
    <Values as ScatterIfInput<Indices, Stencil, Pred>>::Output: MaterializeOutput,
{
    materialize(values.scatter_if_input(indices, stencil, len, default, GpuOp::<Pred>::new())?)
}
