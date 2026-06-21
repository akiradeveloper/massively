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
    primitives::range as primitive_range,
};
use cubecl::prelude::*;

const BLOCK_API_SIZE: u32 = 256;

fn gather_if_one<InputSource, IndexSource, Stencil, Pred>(
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

    let input = super::device_expr_collect(input)?;
    let indices = super::device_expr_collect(indices)?;
    super::ensure_same_len(indices.len, stencil.len())?;
    let flags = stencil.selection_handles(false)?;

    let output = primitive_range::filled(input.policy(), indices.len, default)?;
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
                unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len) },
                unsafe { BufferArg::from_raw_parts(indices.handle.clone(), indices.len) },
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
    /// Output produced by gather.
    type Output;

    /// Gathers `self[indices[i]]`.
    fn gather_input(self, indices: Indices) -> Result<Self::Output, Error>;
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

impl<InputSource, IndexSource> GatherInput<(IndexSource,)> for (InputSource,)
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_input(self, indices: (IndexSource,)) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView1 { source: self.0 },
            SoAView1 { source: indices.0 },
        )
    }
}

impl<Left, Right, IndexSource> GatherInput<(IndexSource,)> for (Left, Right)
where
    SoAView2<Left, Right>: GatherInput<SoAView1<IndexSource>>,
{
    type Output = <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(self, indices: (IndexSource,)) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            SoAView1 { source: indices.0 },
        )
    }
}

impl<Left, Right, IndexSource> GatherInput<IndexSource> for (Left, Right)
where
    SoAView2<Left, Right>: GatherInput<SoAView1<IndexSource>>,
{
    type Output = <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(self, indices: IndexSource) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            SoAView1 { source: indices },
        )
    }
}

impl<First, Second, Third, IndexSource> GatherInput<(IndexSource,)> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: GatherInput<SoAView1<IndexSource>>,
{
    type Output = <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(self, indices: (IndexSource,)) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            SoAView1 { source: indices.0 },
        )
    }
}

impl<First, Second, Third, IndexSource> GatherInput<IndexSource> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: GatherInput<SoAView1<IndexSource>>,
{
    type Output = <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::Output;

    fn gather_input(self, indices: IndexSource) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as GatherInput<SoAView1<IndexSource>>>::gather_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
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

/// Input accepted by [`gather_if`].
#[doc(hidden)]
pub trait GatherIfInput<Indices, Stencil, Pred> {
    /// Output produced by gather-if.
    type Output;
    /// Default value used for positions that are not selected.
    type Default;

    /// Gathers selected elements into default-initialized output.
    fn gather_if_input(
        self,
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
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;
    type Default = (InputSource::Item,);

    fn gather_if_input(
        self,
        indices: SoAView1<IndexSource>,
        stencil: Stencil,
        default: Self::Default,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        Ok(SoA1 {
            source: gather_if_one::<InputSource, IndexSource, Stencil, Pred>(
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
    type Output = SoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;
    type Default = InputSource::Item;

    fn gather_if_input(
        self,
        indices: IndexSource,
        stencil: Stencil,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
            SoAView1 { source: self },
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
    type Output =
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;
    type Default =
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;

    fn gather_if_input(
        self,
        indices: (IndexSource,),
        stencil: Stencil,
        default: Self::Default,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
            SoAView1 { source: self.0 },
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
    type Output =
        <SoAView2<Left, Right> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;
    type Default =
        <SoAView2<Left, Right> as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;

    fn gather_if_input(
        self,
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
                indices: SoAView1<IndexSource>,
                stencil: Stencil,
                default: Self::Default,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&indices)?;
                let ($first_field, $( $field ),+) = default;
                let $first_field = gather_if_one::<$first, IndexSource, Stencil, Pred>(
                    &self.$first_field,
                    &indices.source,
                    &stencil,
                    $first_field,
                )?;
                $(
                    let $field = gather_if_one::<$rest, IndexSource, Stencil, Pred>(
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
            type Output = <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Output;
            type Default = <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::Default;

            fn gather_if_input(
                self,
                indices: IndexSource,
                stencil: Stencil,
                default: Self::Default,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <Self as GatherIfInput<SoAView1<IndexSource>, Stencil, Pred>>::gather_if_input(
                    self,
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
/// multiple value columns, pass borrowed columns as a tuple, such as
/// `(values.slice(..), tags.slice(..))`. Indices may also be passed as a
/// one-column tuple, such as `(indices.slice(..),)`.
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

/// Gathers elements whose staged stencil flag satisfies `Pred`.
///
/// This is a borrowing algorithm: `input` and `indices` are read-only.
pub fn gather_if<Input, Indices, Stencil, Pred>(
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
    <Input as GatherIfInput<Indices, Stencil, Pred>>::Output: MaterializeOutput,
{
    materialize(input.gather_if_input(indices, stencil, default, GpuOp::<Pred>::new())?)
}
