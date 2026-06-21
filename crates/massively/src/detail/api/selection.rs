use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoAView1,
        SoAView2, SoAView3, StorageKernelColumn,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::{GpuOp, PredicateOp},
    policy::CubePolicy,
    primitives::{scan, search, select},
};
use cubecl::prelude::*;

const BLOCK_SELECTION_SIZE: u32 = 256;

fn selection_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SELECTION_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

fn materialize_one<Source>(
    policy: &CubePolicy<Source::Runtime>,
    input: SoAView1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoAView1<Source>: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    ReadOnlySoA::validate(&input)?;
    super::device_expr_collect_with_policy(policy, &input.source)
}

struct TupleSelectionHandles {
    flag: cubecl::server::Handle,
    len: usize,
    len_u32: u32,
}

macro_rules! tuple_selection_handles {
    (
        $self:expr,
        $policy:expr,
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
        let $first_field = super::device_expr_collect_with_policy($policy, &$self.$first_field)?;
        $(
            let $field = super::device_expr_collect_with_policy($policy, &$self.$field)?;
        )+
        let len = $first_field.len();
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let client = $policy.client();
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
                    unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                    $(
                        unsafe { BufferArg::from_raw_parts($field.handle.clone(), len) },
                    )+
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                );
            }
        }
        Ok::<_, Error>((TupleSelectionHandles { flag, len, len_u32 }, $first_field, $( $field ),+))
    }};
}

#[doc(hidden)]
pub trait SelectInput<Pred> {
    type Runtime: Runtime;
    type Output;

    fn select_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

#[doc(hidden)]
pub trait OwnedSelectionInput {}

#[doc(hidden)]
pub trait CopyIfInput<Stencil, Pred> {
    type Runtime: Runtime;
    type Output;

    fn copy_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Stencil, Pred> CopyIfInput<Stencil, Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn copy_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        super::ensure_same_len(self.source.len(), stencil.len())?;
        let values = super::device_expr_collect_with_policy(policy, &self.source)?;
        let handles = stencil.selection_handles_with_policy(policy, false)?;
        let count = select::selected_count(policy, &handles)?;
        Ok(SoA1 {
            source: select::compact_with_count(
                policy,
                select::handles_for_value(&handles, values.handle.clone()),
                count,
            )?,
        })
    }
}

impl<Source, Stencil, Pred> CopyIfInput<Stencil, Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn copy_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as CopyIfInput<Stencil, Pred>>::copy_if_input(
            SoAView1 { source: self },
            policy,
            stencil,
            pred,
        )
    }
}

impl<Source, Stencil, Pred> CopyIfInput<Stencil, Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn copy_if_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <Source as CopyIfInput<Stencil, Pred>>::copy_if_input(self.0, policy, stencil, pred)
    }
}

impl<Source, Pred> SelectInput<Pred> for SoAView1<Source>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn select_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        self.source.validate()?;
        Ok(SoA1 {
            source: super::device_expr_copy_if_with_policy::<Source, Pred>(
                policy,
                &self.source,
                invert,
            )?,
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
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn select_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as SelectInput<Pred>>::select_input(
            SoAView1 { source: self },
            policy,
            invert,
            pred,
        )
    }
}

impl<Source, Pred> SelectInput<Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn select_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <Source as SelectInput<super::Tuple1PredicateOp<Pred>>>::select_input(
            self.0,
            policy,
            invert,
            GpuOp::<super::Tuple1PredicateOp<Pred>>::new(),
        )
    }
}

impl<Source> OwnedSelectionInput for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
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

impl<Source> OwnedSelectionInput for (Source,)
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
{
}

macro_rules! impl_selection_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Pred> SelectInput<Pred> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: SelectInput<Pred>,
        {
            type Runtime = <$view<$( $ty ),+> as SelectInput<Pred>>::Runtime;
            type Output = <$view<$( $ty ),+> as SelectInput<Pred>>::Output;

            fn select_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as SelectInput<Pred>>::select_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    invert,
                    pred,
                )
            }
        }

        impl<$( $ty ),+> OwnedSelectionInput for ($( $ty ),+)
        where
            $view<$( $ty ),+>: OwnedSelectionInput,
        {
        }

        impl<$( $ty ),+, Stencil, Pred> CopyIfInput<Stencil, Pred> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: CopyIfInput<Stencil, Pred>,
        {
            type Runtime = <$view<$( $ty ),+> as CopyIfInput<Stencil, Pred>>::Runtime;
            type Output = <$view<$( $ty ),+> as CopyIfInput<Stencil, Pred>>::Output;

            fn copy_if_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                stencil: Stencil,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as CopyIfInput<Stencil, Pred>>::copy_if_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    stencil,
                    pred,
                )
            }
        }
    };
}

impl_selection_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_selection_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn select_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                let count = select::selected_count(policy, &first_handles)?;
                let control_handles = first_handles.clone();
                let $first_field = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(policy, first_handles, count)?;
                $(
                    let $field = select::compact_with_count::<
                        <$rest as KernelColumn>::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        policy,
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
            type Runtime = <$first as KernelColumn>::Runtime;

            fn count_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                _pred: GpuOp<Pred>,
            ) -> Result<usize, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    policy.client(),
                    first_handles.count,
                )? as usize)
            }

            fn find_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                _pred: GpuOp<Pred>,
            ) -> Result<Option<usize>, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                let _ = &$first_field;
                $( let _ = &$field; )+
                search::first_flag(policy, handles.flag, handles.len, handles.len)
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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;
            type SplitOutput = (Self::Output, Self::Output);

            fn is_partitioned_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<bool, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count = select::selected_count(policy, &first_handles)?;
                let first_rejected = search::first_unset_flag(
                    policy,
                    handles.flag,
                    handles.len,
                    handles.len,
                )?
                .unwrap_or(handles.len);
                Ok(selected_count == first_rejected)
            }

            fn partition_copy_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::SplitOutput, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count =
                    select::selected_count(policy, &selected_first_handles)?;
                let rejected_count = handles.len - selected_count;
                let selected_control = selected_first_handles.clone();
                let selected_first = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(policy, selected_first_handles, selected_count)?;
                let rejected_first = select::compact_rejected_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(
                    policy,
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
                                policy,
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
                                policy,
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
            Stencil: super::SelectionStencil<Pred, Runtime = <$first as KernelColumn>::Runtime>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn copy_if_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                stencil: Stencil,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                super::ensure_same_len(stencil.len(), ReadOnlySoA::len(&self))?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let handles = stencil.selection_handles_with_policy(policy, false)?;
                let count = select::selected_count(policy, &handles)?;
                let control = handles.clone();
                Ok($output {
                    $first_field: select::compact_with_count(
                        policy,
                        select::handles_for_value(&handles, $first_field.handle.clone()),
                        count,
                    )?,
                    $(
                        $field: select::compact_with_count(
                            policy,
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

impl_tuple_selection!(SoA2<A, B> { left, right }, tuple2_predicate_flags_kernel);
impl_tuple_selection!(SoA3<A, B, C> { first, second, third }, tuple3_predicate_flags_kernel);

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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn select_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                let count = select::selected_count(policy, &first_handles)?;
                let control_handles = first_handles.clone();
                let $first_field = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(policy, first_handles, count)?;
                $(
                    let $field = select::compact_with_count::<
                        <$rest as KernelColumn>::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        policy,
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
            type Runtime = <$first as KernelColumn>::Runtime;

            fn count_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                _pred: GpuOp<Pred>,
            ) -> Result<usize, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag,
                    $first_field.handle.clone(),
                )?;
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    policy.client(),
                    first_handles.count,
                )? as usize)
            }

            fn find_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                _pred: GpuOp<Pred>,
            ) -> Result<Option<usize>, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                let _ = &$first_field;
                $( let _ = &$field; )+
                search::first_flag(policy, handles.flag, handles.len, handles.len)
            }
        }
    };
}

impl_readonly_tuple_selection!(SoAView2 -> SoA2<A, B> { left, right }, tuple2_predicate_flags_kernel);
impl_readonly_tuple_selection!(SoAView3 -> SoA3<A, B, C> { first, second, third }, tuple3_predicate_flags_kernel);

impl<Left, Right> OwnedSelectionInput for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
{
}

impl<First, Second, Third> OwnedSelectionInput for SoAView3<First, Second, Third>
where
    Self: ReadOnlySoA<Scalar = First::Item>,
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
{
}

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
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;
            type SplitOutput = (Self::Output, Self::Output);

            fn is_partitioned_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<bool, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count = select::selected_count(policy, &first_handles)?;
                let first_rejected = search::first_unset_flag(
                    policy,
                    handles.flag,
                    handles.len,
                    handles.len,
                )?
                .unwrap_or(handles.len);
                Ok(selected_count == first_rejected)
            }

            fn partition_copy_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::SplitOutput, Error> {
                let (handles, $first_field, $( $field ),+) =
                    tuple_selection_handles!(
                        self,
                        policy,
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
                    policy,
                    handles.len,
                    handles.len_u32,
                    handles.flag.clone(),
                    $first_field.handle.clone(),
                )?;
                let selected_count =
                    select::selected_count(policy, &selected_first_handles)?;
                let rejected_count = handles.len - selected_count;
                let selected_control = selected_first_handles.clone();
                let selected_first = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(policy, selected_first_handles, selected_count)?;
                let rejected_first = select::compact_rejected_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >(
                    policy,
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
                                policy,
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
                                policy,
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

macro_rules! impl_partition_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Pred> PartitionInput<Pred> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: PartitionInput<Pred>,
        {
            type Runtime = <$view<$( $ty ),+> as PartitionInput<Pred>>::Runtime;
            type Output = <$view<$( $ty ),+> as PartitionInput<Pred>>::Output;
            type SplitOutput = <$view<$( $ty ),+> as PartitionInput<Pred>>::SplitOutput;

            fn is_partitioned_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                pred: GpuOp<Pred>,
            ) -> Result<bool, Error> {
                <$view<$( $ty ),+> as PartitionInput<Pred>>::is_partitioned_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    pred,
                )
            }

            fn partition_copy_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                pred: GpuOp<Pred>,
            ) -> Result<Self::SplitOutput, Error> {
                <$view<$( $ty ),+> as PartitionInput<Pred>>::partition_copy_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    pred,
                )
            }
        }
    };
}

impl_partition_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_partition_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

/// Keeps values whose staged stencil flag satisfies `Pred`.
///
/// This is a borrowing algorithm. It reads the input and returns newly owned SoA
/// storage containing the selected values.
pub fn copy_if<Source, Stencil, Pred>(
    policy: &CubePolicy<<Source as CopyIfInput<Stencil, Pred>>::Runtime>,
    source: Source,
    stencil: Stencil,
    _pred: Pred,
) -> Result<<<Source as CopyIfInput<Stencil, Pred>>::Output as MaterializeOutput>::Output, Error>
where
    Source: CopyIfInput<Stencil, Pred>,
    <Source as CopyIfInput<Stencil, Pred>>::Output:
        MaterializeOutput<Runtime = <Source as CopyIfInput<Stencil, Pred>>::Runtime>,
{
    materialize(
        policy,
        source.copy_if_input(policy, stencil, GpuOp::<Pred>::new())?,
    )
}

/// Removes values satisfying `Pred`.
///
/// This is a borrowing algorithm. It reads the input and returns newly owned SoA
/// storage for the remaining values.
pub fn remove_if<Source, Pred>(
    policy: &CubePolicy<<Source as SelectInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<<<Source as SelectInput<Pred>>::Output as MaterializeOutput>::Output, Error>
where
    Source: SelectInput<Pred> + OwnedSelectionInput,
    <Source as SelectInput<Pred>>::Output:
        MaterializeOutput<Runtime = <Source as SelectInput<Pred>>::Runtime>,
{
    materialize(
        policy,
        source.select_input(policy, true, GpuOp::<Pred>::new())?,
    )
}

#[doc(hidden)]
pub trait PredicateQueryInput<Pred> {
    type Runtime: Runtime;

    fn count_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        pred: GpuOp<Pred>,
    ) -> Result<usize, Error>;
    fn find_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error>;
}

#[doc(hidden)]
pub trait PartitionInput<Pred> {
    type Runtime: Runtime;
    type Output;
    type SplitOutput;

    fn is_partitioned_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<bool, Error>;
    fn partition_copy_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Self::SplitOutput, Error>;
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
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;
    type SplitOutput = (
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
    );

    fn is_partitioned_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<bool, Error> {
        let input = materialize_one(policy, self)?;
        search::is_partitioned(policy, &input, GpuOp::<Pred>::new())
    }

    fn partition_copy_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::SplitOutput, Error> {
        let input = materialize_one(policy, self)?;
        let (matching, failing) = select::partition_copy(policy, &input, GpuOp::<Pred>::new())?;
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
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;
    type SplitOutput = (
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
    );

    fn is_partitioned_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<bool, Error> {
        <SoAView1<Source> as PartitionInput<Pred>>::is_partitioned_input(
            SoAView1 { source: self },
            policy,
            pred,
        )
    }

    fn partition_copy_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Self::SplitOutput, Error> {
        <SoAView1<Source> as PartitionInput<Pred>>::partition_copy_input(
            SoAView1 { source: self },
            policy,
            pred,
        )
    }
}

impl<Source, Pred> PartitionInput<Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;
    type SplitOutput = (
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
        SoA1<DeviceVec<Source::Runtime, Source::Item>>,
    );

    fn is_partitioned_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<bool, Error> {
        <Source as PartitionInput<super::Tuple1PredicateOp<Pred>>>::is_partitioned_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1PredicateOp<Pred>>::new(),
        )
    }

    fn partition_copy_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::SplitOutput, Error> {
        <Source as PartitionInput<super::Tuple1PredicateOp<Pred>>>::partition_copy_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1PredicateOp<Pred>>::new(),
        )
    }
}

impl<Source, Pred> PredicateQueryInput<Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn count_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        _pred: GpuOp<Pred>,
    ) -> Result<usize, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_count_if_with_policy::<Source, Pred>(policy, &self.source, invert)
    }

    fn find_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        _pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_find_if_with_policy::<Source, Pred>(policy, &self.source, invert)
    }
}

impl<Source, Pred> PredicateQueryInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn count_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        pred: GpuOp<Pred>,
    ) -> Result<usize, Error> {
        <SoAView1<Source> as PredicateQueryInput<Pred>>::count_input(
            SoAView1 { source: self },
            policy,
            invert,
            pred,
        )
    }

    fn find_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as PredicateQueryInput<Pred>>::find_input(
            SoAView1 { source: self },
            policy,
            invert,
            pred,
        )
    }
}

impl<Source, Pred> PredicateQueryInput<Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;

    fn count_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        _pred: GpuOp<Pred>,
    ) -> Result<usize, Error> {
        <Source as PredicateQueryInput<super::Tuple1PredicateOp<Pred>>>::count_input(
            self.0,
            policy,
            invert,
            GpuOp::<super::Tuple1PredicateOp<Pred>>::new(),
        )
    }

    fn find_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        _pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error> {
        <Source as PredicateQueryInput<super::Tuple1PredicateOp<Pred>>>::find_input(
            self.0,
            policy,
            invert,
            GpuOp::<super::Tuple1PredicateOp<Pred>>::new(),
        )
    }
}

macro_rules! impl_predicate_query_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Pred> PredicateQueryInput<Pred> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: PredicateQueryInput<Pred>,
        {
            type Runtime = <$view<$( $ty ),+> as PredicateQueryInput<Pred>>::Runtime;

            fn count_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                pred: GpuOp<Pred>,
            ) -> Result<usize, Error> {
                <$view<$( $ty ),+> as PredicateQueryInput<Pred>>::count_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    invert,
                    pred,
                )
            }

            fn find_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                pred: GpuOp<Pred>,
            ) -> Result<Option<usize>, Error> {
                <$view<$( $ty ),+> as PredicateQueryInput<Pred>>::find_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    invert,
                    pred,
                )
            }
        }
    };
}

impl_predicate_query_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_predicate_query_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

/// Counts values satisfying `Pred`.
pub fn count_if<Source, Pred>(
    policy: &CubePolicy<<Source as PredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<usize, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    source.count_input(policy, false, GpuOp::<Pred>::new())
}

/// Returns whether all values satisfy `Pred`.
pub fn all_of<Source, Pred>(
    policy: &CubePolicy<<Source as PredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    pred: Pred,
) -> Result<bool, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    Ok(find_if_not(policy, source, pred)?.is_none())
}

/// Returns whether any value satisfies `Pred`.
pub fn any_of<Source, Pred>(
    policy: &CubePolicy<<Source as PredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    pred: Pred,
) -> Result<bool, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    Ok(find_if(policy, source, pred)?.is_some())
}

/// Returns whether no values satisfy `Pred`.
pub fn none_of<Source, Pred>(
    policy: &CubePolicy<<Source as PredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    pred: Pred,
) -> Result<bool, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    Ok(find_if(policy, source, pred)?.is_none())
}

/// Finds the first value satisfying `Pred`.
pub fn find_if<Source, Pred>(
    policy: &CubePolicy<<Source as PredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<Option<usize>, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    source.find_input(policy, false, GpuOp::<Pred>::new())
}

fn find_if_not<Source, Pred>(
    policy: &CubePolicy<<Source as PredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<Option<usize>, Error>
where
    Source: PredicateQueryInput<Pred>,
{
    source.find_input(policy, true, GpuOp::<Pred>::new())
}

/// Partitions elements by `Pred`, preserving relative order within each side.
pub fn partition<Input, Pred>(
    policy: &CubePolicy<<Input as PartitionInput<Pred>>::Runtime>,
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
    <<Input as PartitionInput<Pred>>::SplitOutput as TuplePair>::Left:
        MaterializeOutput<Runtime = <Input as PartitionInput<Pred>>::Runtime>,
    <<Input as PartitionInput<Pred>>::SplitOutput as TuplePair>::Right:
        MaterializeOutput<Runtime = <Input as PartitionInput<Pred>>::Runtime>,
{
    let (matching, failing) = input
        .partition_copy_input(policy, GpuOp::<Pred>::new())?
        .into_pair();
    Ok((
        materialize(policy, matching)?,
        materialize(policy, failing)?,
    ))
}

/// Returns whether all elements satisfying `Pred` appear before all non-matching elements.
pub fn is_partitioned<Input, Pred>(
    policy: &CubePolicy<<Input as PartitionInput<Pred>>::Runtime>,
    input: Input,
    _pred: Pred,
) -> Result<bool, Error>
where
    Input: PartitionInput<Pred>,
{
    input.is_partitioned_input(policy, GpuOp::<Pred>::new())
}
