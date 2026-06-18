use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoA4,
        SoA5, SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoAView1, SoAView2, SoAView3, SoAView4,
        SoAView5, SoAView6, SoAView7, SoAView8, SoAView9, SoAView10, SoAView11, SoAView12,
        StorageKernelColumn,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::{GpuOp, PredicateOp},
    primitives::{scan, search, select},
};
use cubecl::prelude::*;

const BLOCK_SELECTION_SIZE: u32 = 256;

fn selection_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SELECTION_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

fn materialize_one<Source>(
    input: SoAView1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoAView1<Source>: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    ReadOnlySoA::validate(&input)?;
    super::device_expr_collect(&input.source)
}

struct TupleSelectionHandles {
    flag: cubecl::server::Handle,
    len: usize,
    len_u32: u32,
}

macro_rules! tuple_selection_handles {
    (
        $self:expr,
        $invert:expr,
        $kernel_name:ident,
        ($first_item_ty:ty, $( $item_ty:ty ),+),
        $runtime_ty:ty,
        $pred:ty,
        $first_field:ident,
        $( $field:ident ),+
    ) => {{
        $self.$first_field.validate()?;
        $(
            $self.$field.validate()?;
            super::ensure_same_len($self.$field.len(), $self.$first_field.len())?;
        )+
        let $first_field = super::device_expr_collect(&$self.$first_field)?;
        $(
            let $field = super::device_expr_collect(&$self.$field)?;
        )+
        let len = $first_field.len();
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let client = $first_field.policy().client();
        let flag = client.empty(len * std::mem::size_of::<u32>());
        if len != 0 {
            let block_count_u32 = selection_block_count(len)?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let invert_values = [if $invert { 1_u32 } else { 0_u32 }];
            let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
            unsafe {
                $kernel_name::launch_unchecked::<
                    $first_item_ty,
                    $( $item_ty, )+
                    $pred,
                    $runtime_ty,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SELECTION_SIZE),
                    ArrayArg::from_raw_parts::<$first_item_ty>(&$first_field.handle, len, 1),
                    $(
                        ArrayArg::from_raw_parts::<$item_ty>(&$field.handle, len, 1),
                    )+
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&invert_handle, 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&flag, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok::<_, Error>((TupleSelectionHandles { flag, len, len_u32 }, $first_field, $( $field ),+))
    }};
}

#[doc(hidden)]
pub trait SelectInput<Pred> {
    type Output;

    fn select_input(self, invert: bool, pred: GpuOp<Pred>) -> Result<Self::Output, Error>;
}

#[doc(hidden)]
pub trait OwnedSelectionInput {}

#[doc(hidden)]
pub trait CopyIfInput<Stencil, Pred> {
    type Output;

    fn copy_if_input(self, stencil: Stencil, pred: GpuOp<Pred>) -> Result<Self::Output, Error>;
}

impl<Source, Stencil, Pred> CopyIfInput<Stencil, Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = Source::Runtime> + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn copy_if_input(self, stencil: Stencil, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        stencil.validate()?;
        super::ensure_same_len(self.source.len(), stencil.len())?;
        let values = super::device_expr_collect(&self.source)?;
        let handles = super::device_expr_selection_handles::<Stencil, Pred>(&stencil, false)?;
        let count = select::selected_count(values.policy(), &handles)?;
        Ok(SoA1 {
            source: select::compact_with_count(
                values.policy(),
                select::handles_for_value(&handles, values.handle.clone()),
                count,
            )?,
        })
    }
}

impl<Source, Stencil, Pred> CopyIfInput<Stencil, Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = Source::Runtime> + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn copy_if_input(self, stencil: Stencil, pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        <SoAView1<Source> as CopyIfInput<Stencil, Pred>>::copy_if_input(
            SoAView1 { source: self },
            stencil,
            pred,
        )
    }
}

impl<Source, Pred> SelectInput<Pred> for SoAView1<Source>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn select_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        self.source.validate()?;
        Ok(SoA1 {
            source: super::device_expr_copy_if::<Source, Pred>(&self.source, invert)?,
        })
    }
}

impl<Source, Pred> SelectInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn select_input(self, invert: bool, pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        <SoAView1<Source> as SelectInput<Pred>>::select_input(
            SoAView1 { source: self },
            invert,
            pred,
        )
    }
}

impl<Source> OwnedSelectionInput for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
{
}

impl<Source> OwnedSelectionInput for Source
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
{
}

macro_rules! impl_tuple_selection {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> SelectInput<Pred> for $name<$first, $( $rest ),+>
        where
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<(
                impl_tuple_selection!(@item_ty $first),
                $( impl_tuple_selection!(@item_ty $rest) ),+
            )>,
        {
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn select_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        invert,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                let count = select::selected_count($first_field.policy(), &first_handles)?;
                let control_handles = first_handles.clone();
                let $first_field = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), first_handles, count)?;
                $(
                    let $field = select::compact_with_count::<
                        <$rest as KernelColumn>::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        $field.policy(),
                        select::handles_for_value(&control_handles, $field.handle.clone()),
                        count,
                    )?;
                )+
                Ok($name { $first_field, $( $field ),+ })
            }
        }

        impl<$first, $( $rest ),+> OwnedSelectionInput for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: StorageKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
        }

        impl<$first, $( $rest ),+, Pred> PredicateQueryInput<Pred> for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<(
                impl_tuple_selection!(@item_ty $first),
                $( impl_tuple_selection!(@item_ty $rest) ),+
            )>,
        {
            fn count_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<usize, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        invert,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                if handles.len == 0 {
                    return Ok(0);
                }
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    $first_field.policy().client(),
                    first_handles.count,
                ) as usize)
            }

            fn find_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<Option<usize>, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        invert,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                search::first_flag($first_field.policy(), handles.flag, handles.len, handles.len)
            }
        }

        impl<$first, $( $rest ),+, Pred> PartitionInput<Pred> for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<(
                impl_tuple_selection!(@item_ty $first),
                $( impl_tuple_selection!(@item_ty $rest) ),+
            )>,
        {
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;
            type SplitOutput = (Self::Output, Self::Output);

            fn partition_input(self, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        false,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                let control_handles = first_handles.clone();
                let $first_field = select::partition_from_handles::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), first_handles)?;
                $(
                    let $field = select::partition_from_handles::<
                        <$rest as KernelColumn>::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        $field.policy(),
                        select::handles_for_value(&control_handles, $field.handle.clone()),
                    )?;
                )+
                Ok($name { $first_field, $( $field ),+ })
            }

            fn is_partitioned_input(self, _pred: GpuOp<Pred>) -> Result<bool, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        false,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count = select::selected_count($first_field.policy(), &first_handles)?;
                let first_rejected = search::first_unset_flag(
                    $first_field.policy(),
                    handles.flag,
                    handles.len,
                    handles.len,
                )?
                .unwrap_or(handles.len);
                Ok(selected_count == first_rejected)
            }

            fn partition_point_input(self, _pred: GpuOp<Pred>) -> Result<usize, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        true,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                Ok(search::first_flag($first_field.policy(), handles.flag, handles.len, handles.len)?
                    .unwrap_or(handles.len))
            }

            fn partition_copy_input(self, _pred: GpuOp<Pred>) -> Result<Self::SplitOutput, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        false,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                let selected_first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count =
                    select::selected_count($first_field.policy(), &selected_first_handles)?;
                let rejected_count = handles.len - selected_count;
                let selected_control = selected_first_handles.clone();
                let selected_first = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), selected_first_handles, selected_count)?;
                let rejected_first = select::compact_rejected_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(
                    $first_field.policy(),
                    select::handles_for_value(&selected_control, $first_field.handle.clone()),
                    rejected_count,
                )?;
                Ok((
                    $name {
                        $first_field: selected_first,
                        $(
                            $field: select::compact_with_count::<
                                <$rest as KernelColumn>::Runtime,
                                <$rest as KernelColumn>::Item,
                            >(
                                $field.policy(),
                                select::handles_for_value(&selected_control, $field.handle.clone()),
                                selected_count,
                            )?,
                        )+
                    },
                    $name {
                        $first_field: rejected_first,
                        $(
                            $field: select::compact_rejected_with_count::<
                                <$rest as KernelColumn>::Runtime,
                                <$rest as KernelColumn>::Item,
                            >(
                                $field.policy(),
                                select::handles_for_value(&selected_control, $field.handle.clone()),
                                rejected_count,
                            )?,
                        )+
                    },
                ))
            }
        }
    };
}

macro_rules! impl_tuple_copy_if {
    ($input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+, Stencil, Pred> CopyIfInput<Stencil, Pred>
            for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            Stencil: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Stencil::Item: CubePrimitive + CubeElement,
            Stencil::Expr: GpuExpr<Stencil::Item>,
            Pred: PredicateOp<Stencil::Item>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn copy_if_input(
                self,
                stencil: Stencil,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                stencil.validate()?;
                super::ensure_same_len(stencil.len(), ReadOnlySoA::len(&self))?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let handles = super::device_expr_selection_handles::<Stencil, Pred>(&stencil, false)?;
                let count = select::selected_count($first_field.policy(), &handles)?;
                let control = handles.clone();
                Ok($output {
                    $first_field: select::compact_with_count(
                        $first_field.policy(),
                        select::handles_for_value(&handles, $first_field.handle.clone()),
                        count,
                    )?,
                    $(
                        $field: select::compact_with_count(
                            $field.policy(),
                            select::handles_for_value(&control, $field.handle.clone()),
                            count,
                        )?,
                    )+
                })
            }
        }
    };
}

impl_tuple_copy_if!(SoAView2 -> SoA2<A, B> { left, right });
impl_tuple_copy_if!(SoA2 -> SoA2<A, B> { left, right });
impl_tuple_copy_if!(SoAView3 -> SoA3<A, B, C> { first, second, third });
impl_tuple_copy_if!(SoA3 -> SoA3<A, B, C> { first, second, third });
impl_tuple_copy_if!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_tuple_copy_if!(SoA4 -> SoA4<A, B, C, D> { a, b, c, d });
impl_tuple_copy_if!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_tuple_copy_if!(SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_tuple_copy_if!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_tuple_copy_if!(SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_tuple_copy_if!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_tuple_copy_if!(SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_tuple_copy_if!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_tuple_copy_if!(SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_tuple_copy_if!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_tuple_copy_if!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_tuple_copy_if!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_tuple_copy_if!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_tuple_copy_if!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_tuple_copy_if!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_tuple_copy_if!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });
impl_tuple_copy_if!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

impl_tuple_selection!(SoA2<A, B> { left, right }, tuple2_predicate_flags_kernel);
impl_tuple_selection!(SoA3<A, B, C> { first, second, third }, tuple3_predicate_flags_kernel);
impl_tuple_selection!(SoA4<A, B, C, D> { a, b, c, d }, tuple4_predicate_flags_kernel);
impl_tuple_selection!(SoA5<A, B, C, D, E> { a, b, c, d, e }, tuple5_predicate_flags_kernel);
impl_tuple_selection!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, tuple6_predicate_flags_kernel);
impl_tuple_selection!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, tuple7_predicate_flags_kernel);
impl_tuple_selection!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, tuple8_predicate_flags_kernel);
impl_tuple_selection!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, tuple9_predicate_flags_kernel);
impl_tuple_selection!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, tuple10_predicate_flags_kernel);
impl_tuple_selection!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, tuple11_predicate_flags_kernel);
impl_tuple_selection!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, tuple12_predicate_flags_kernel);

macro_rules! impl_readonly_tuple_selection {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> SelectInput<Pred> for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<(
                impl_readonly_tuple_selection!(@item_ty $first),
                $( impl_readonly_tuple_selection!(@item_ty $rest) ),+
            )>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn select_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        invert,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                let count = select::selected_count($first_field.policy(), &first_handles)?;
                let control_handles = first_handles.clone();
                let $first_field = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), first_handles, count)?;
                $(
                    let $field = select::compact_with_count::<
                        <$rest as KernelColumn>::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        $field.policy(),
                        select::handles_for_value(&control_handles, $field.handle.clone()),
                        count,
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }
        }

        impl<$first, $( $rest ),+, Pred> PredicateQueryInput<Pred> for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<(
                impl_readonly_tuple_selection!(@item_ty $first),
                $( impl_readonly_tuple_selection!(@item_ty $rest) ),+
            )>,
        {
            fn count_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<usize, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        invert,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                if handles.len == 0 {
                    return Ok(0);
                }
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    $first_field.policy().client(),
                    first_handles.count,
                ) as usize)
            }

            fn find_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<Option<usize>, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        invert,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                search::first_flag($first_field.policy(), handles.flag, handles.len, handles.len)
            }
        }
    };
}

impl_readonly_tuple_selection!(SoAView2 -> SoA2<A, B> { left, right }, tuple2_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView3 -> SoA3<A, B, C> { first, second, third }, tuple3_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d }, tuple4_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e }, tuple5_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, tuple6_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, tuple7_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, tuple8_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, tuple9_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, tuple10_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, tuple11_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, tuple12_predicate_flags_kernel);

macro_rules! impl_readonly_tuple_partition {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> PartitionInput<Pred> for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<(
                impl_readonly_tuple_partition!(@item_ty $first),
                $( impl_readonly_tuple_partition!(@item_ty $rest) ),+
            )>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;
            type SplitOutput = (Self::Output, Self::Output);

            fn partition_input(self, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        false,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                let control_handles = first_handles.clone();
                let $first_field = select::partition_from_handles::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), first_handles)?;
                $(
                    let $field = select::partition_from_handles::<
                        <$rest as KernelColumn>::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        $field.policy(),
                        select::handles_for_value(&control_handles, $field.handle.clone()),
                    )?;
                )+
                Ok($output { $first_field, $( $field ),+ })
            }

            fn is_partitioned_input(self, _pred: GpuOp<Pred>) -> Result<bool, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        false,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                let first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count = select::selected_count($first_field.policy(), &first_handles)?;
                let first_rejected = search::first_unset_flag(
                    $first_field.policy(),
                    handles.flag,
                    handles.len,
                    handles.len,
                )?
                .unwrap_or(handles.len);
                Ok(selected_count == first_rejected)
            }

            fn partition_point_input(self, _pred: GpuOp<Pred>) -> Result<usize, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        true,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                $( let _ = &$field; )+
                Ok(search::first_flag($first_field.policy(), handles.flag, handles.len, handles.len)?
                    .unwrap_or(handles.len))
            }

            fn partition_copy_input(self, _pred: GpuOp<Pred>) -> Result<Self::SplitOutput, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        false,
                        $kernel_name,
                        (
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item ),+
                        ),
                        <$first as KernelColumn>::Runtime,
                        Pred,
                        $first_field,
                        $( $field ),+
                    )?;
                let selected_first_handles = select::handles_from_flags(
                    $first_field.policy(),
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count =
                    select::selected_count($first_field.policy(), &selected_first_handles)?;
                let rejected_count = handles.len - selected_count;
                let selected_control = selected_first_handles.clone();
                let selected_first = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), selected_first_handles, selected_count)?;
                let rejected_first = select::compact_rejected_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(
                    $first_field.policy(),
                    select::handles_for_value(&selected_control, $first_field.handle.clone()),
                    rejected_count,
                )?;
                Ok((
                    $output {
                        $first_field: selected_first,
                        $(
                            $field: select::compact_with_count::<
                                <$rest as KernelColumn>::Runtime,
                                <$rest as KernelColumn>::Item,
                            >(
                                $field.policy(),
                                select::handles_for_value(&selected_control, $field.handle.clone()),
                                selected_count,
                            )?,
                        )+
                    },
                    $output {
                        $first_field: rejected_first,
                        $(
                            $field: select::compact_rejected_with_count::<
                                <$rest as KernelColumn>::Runtime,
                                <$rest as KernelColumn>::Item,
                            >(
                                $field.policy(),
                                select::handles_for_value(&selected_control, $field.handle.clone()),
                                rejected_count,
                            )?,
                        )+
                    },
                ))
            }
        }
    };
}

impl_readonly_tuple_partition!(SoAView2 -> SoA2<A, B> { left, right }, tuple2_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView3 -> SoA3<A, B, C> { first, second, third }, tuple3_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView4 -> SoA4<A, B, C, D> { a, b, c, d }, tuple4_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView5 -> SoA5<A, B, C, D, E> { a, b, c, d, e }, tuple5_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, tuple6_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, tuple7_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, tuple8_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, tuple9_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, tuple10_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, tuple11_predicate_flags_kernel);
impl_readonly_tuple_partition!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, tuple12_predicate_flags_kernel);

/// Keeps values satisfying `Pred`.
///
/// This is a borrowing algorithm. It reads the input and returns newly owned SoA
/// storage containing the selected values.
pub fn copy_if<Source, Stencil, Pred>(
    source: Source,
    stencil: Stencil,
    _pred: Pred,
) -> Result<<<Source as CopyIfInput<Stencil, Pred>>::Output as MaterializeOutput>::Output, Error>
where
    Source: CopyIfInput<Stencil, Pred>,
    <Source as CopyIfInput<Stencil, Pred>>::Output: MaterializeOutput,
{
    materialize(source.copy_if_input(stencil, GpuOp::<Pred>::new())?)
}

/// Removes values satisfying `Pred`.
///
/// This is a borrowing algorithm. It reads the input and returns newly owned SoA
/// storage for the remaining values.
pub fn remove_if<Source, Pred>(
    source: Source,
    _pred: Pred,
) -> Result<<<Source as SelectInput<Pred>>::Output as MaterializeOutput>::Output, Error>
where
    Source: SelectInput<Pred> + OwnedSelectionInput,
    <Source as SelectInput<Pred>>::Output: MaterializeOutput,
{
    materialize(source.select_input(true, GpuOp::<Pred>::new())?)
}

#[doc(hidden)]
pub trait PredicateQueryInput<Pred> {
    fn count_input(self, invert: bool, pred: GpuOp<Pred>) -> Result<usize, Error>;
    fn find_input(self, invert: bool, pred: GpuOp<Pred>) -> Result<Option<usize>, Error>;
}

#[doc(hidden)]
pub trait PartitionInput<Pred> {
    type Output;
    type SplitOutput;

    fn partition_input(self, pred: GpuOp<Pred>) -> Result<Self::Output, Error>;
    fn is_partitioned_input(self, pred: GpuOp<Pred>) -> Result<bool, Error>;
    fn partition_point_input(self, pred: GpuOp<Pred>) -> Result<usize, Error>;
    fn partition_copy_input(self, pred: GpuOp<Pred>) -> Result<Self::SplitOutput, Error>;
}

#[doc(hidden)]
pub trait TuplePair {
    type Left;
    type Right;

    fn into_pair(self) -> (Self::Left, Self::Right);
}

impl<Left, Right> TuplePair for (Left, Right) {
    type Left = Left;
    type Right = Right;

    fn into_pair(self) -> (Self::Left, Self::Right) {
        self
    }
}

impl<Source, Pred> PartitionInput<Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;
    type SplitOutput = (
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
    );

    fn partition_input(self, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        let input = materialize_one(self)?;
        Ok(SoA1 {
            source: select::partition(&input, GpuOp::<Pred>::new())?,
        })
    }

    fn is_partitioned_input(self, _pred: GpuOp<Pred>) -> Result<bool, Error> {
        let input = materialize_one(self)?;
        search::is_partitioned(&input, GpuOp::<Pred>::new())
    }

    fn partition_point_input(self, _pred: GpuOp<Pred>) -> Result<usize, Error> {
        let input = materialize_one(self)?;
        search::partition_point(&input, GpuOp::<Pred>::new())
    }

    fn partition_copy_input(self, _pred: GpuOp<Pred>) -> Result<Self::SplitOutput, Error> {
        let input = materialize_one(self)?;
        let (matching, failing) = select::partition_copy(&input, GpuOp::<Pred>::new())?;
        Ok((SoA1 { source: matching }, SoA1 { source: failing }))
    }
}

impl<Source, Pred> PartitionInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;
    type SplitOutput = (
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
    );

    fn partition_input(self, pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        <SoAView1<Source> as PartitionInput<Pred>>::partition_input(SoAView1 { source: self }, pred)
    }

    fn is_partitioned_input(self, pred: GpuOp<Pred>) -> Result<bool, Error> {
        <SoAView1<Source> as PartitionInput<Pred>>::is_partitioned_input(
            SoAView1 { source: self },
            pred,
        )
    }

    fn partition_point_input(self, pred: GpuOp<Pred>) -> Result<usize, Error> {
        <SoAView1<Source> as PartitionInput<Pred>>::partition_point_input(
            SoAView1 { source: self },
            pred,
        )
    }

    fn partition_copy_input(self, pred: GpuOp<Pred>) -> Result<Self::SplitOutput, Error> {
        <SoAView1<Source> as PartitionInput<Pred>>::partition_copy_input(
            SoAView1 { source: self },
            pred,
        )
    }
}

impl<Source, Pred> PredicateQueryInput<Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    fn count_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<usize, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_count_if::<Source, Pred>(&self.source, invert)
    }

    fn find_input(self, invert: bool, _pred: GpuOp<Pred>) -> Result<Option<usize>, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_find_if::<Source, Pred>(&self.source, invert)
    }
}

impl<Source, Pred> PredicateQueryInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    fn count_input(self, invert: bool, pred: GpuOp<Pred>) -> Result<usize, Error> {
        <SoAView1<Source> as PredicateQueryInput<Pred>>::count_input(
            SoAView1 { source: self },
            invert,
            pred,
        )
    }

    fn find_input(self, invert: bool, pred: GpuOp<Pred>) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as PredicateQueryInput<Pred>>::find_input(
            SoAView1 { source: self },
            invert,
            pred,
        )
    }
}

/// Counts values satisfying `Pred`.
pub fn count_if<Source, Pred>(source: Source, _pred: Pred) -> Result<usize, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    source.count_input(false, GpuOp::<Pred>::new())
}

/// Returns whether all values satisfy `Pred`.
pub fn all_of<Source, Pred>(source: Source, pred: Pred) -> Result<bool, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    Ok(find_if_not(source, pred)?.is_none())
}

/// Returns whether any value satisfies `Pred`.
pub fn any_of<Source, Pred>(source: Source, pred: Pred) -> Result<bool, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    Ok(find_if(source, pred)?.is_some())
}

/// Returns whether no values satisfy `Pred`.
pub fn none_of<Source, Pred>(source: Source, pred: Pred) -> Result<bool, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    Ok(find_if(source, pred)?.is_none())
}

/// Finds the first value satisfying `Pred`.
pub fn find_if<Source, Pred>(source: Source, _pred: Pred) -> Result<Option<usize>, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    source.find_input(false, GpuOp::<Pred>::new())
}

fn find_if_not<Source, Pred>(source: Source, _pred: Pred) -> Result<Option<usize>, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    source.find_input(true, GpuOp::<Pred>::new())
}

/// Partitions elements by `Pred`, preserving relative order within each side.
pub fn partition<Input, Pred>(
    input: Input,
    _pred: Pred,
) -> Result<
    (
        <<<Input as PartitionInput<Pred>>::SplitOutput as TuplePair>::Left as MaterializeOutput>::Output,
        <<<Input as PartitionInput<Pred>>::SplitOutput as TuplePair>::Right as MaterializeOutput>::Output,
    ),
    Error,
>
where
    Input: PartitionInput<Pred>,
    <Input as PartitionInput<Pred>>::SplitOutput: TuplePair,
    <<Input as PartitionInput<Pred>>::SplitOutput as TuplePair>::Left: MaterializeOutput,
    <<Input as PartitionInput<Pred>>::SplitOutput as TuplePair>::Right: MaterializeOutput,
{
    let (matching, failing) = input
        .partition_copy_input(GpuOp::<Pred>::new())?
        .into_pair();
    Ok((materialize(matching)?, materialize(failing)?))
}

/// Returns whether all elements satisfying `Pred` appear before all non-matching elements.
pub fn is_partitioned<Input, Pred>(input: Input, _pred: Pred) -> Result<bool, Error>
where
    Input: PartitionInput<Pred>,
{
    input.is_partitioned_input(GpuOp::<Pred>::new())
}
