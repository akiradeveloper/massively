use super::memory::{MaterializeOutput, materialize};
use crate::{
    detail::op::kernel::BinaryPredicateOp,
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyKernelColumn, ReadOnlySoA, S0, SoA, SoA1,
        SoA2, SoA3, SoAView2, SoAView3, StorageKernelColumn,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::select,
};
use cubecl::prelude::*;

const BLOCK_SEQUENCE_SIZE: u32 = 256;

fn sequence_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SEQUENCE_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

struct StagedSequenceColumn {
    slot0: (cubecl::server::Handle, usize),
    slot1: (cubecl::server::Handle, usize),
    slot2: (cubecl::server::Handle, usize),
    slot3: (cubecl::server::Handle, usize),
    slot_offsets: cubecl::server::Handle,
}

fn stage_sequence_column<Source>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
) -> Result<StagedSequenceColumn, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
{
    let bindings = source.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(policy.client())?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    Ok(StagedSequenceColumn {
        slot0: (slot0.0.clone(), slot0.1),
        slot1: (slot1.0.clone(), slot1.1),
        slot2: (slot2.0.clone(), slot2.1),
        slot3: (slot3.0.clone(), slot3.1),
        slot_offsets,
    })
}

fn key_run_flags<KeySource, Eq>(
    policy: &CubePolicy<KeySource::Runtime>,
    keys: &KeySource,
) -> Result<cubecl::server::Handle, Error>
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Eq: BinaryPredicateOp<KeySource::Item>,
{
    keys.validate()?;
    let len = keys.len();
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = keys.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let block_count_u32 = sequence_block_count(len)?;

    unsafe {
        unique_by_key_device_expr_flags_kernel::launch_unchecked::<
            KeySource::Item,
            KeySource::Expr,
            Eq,
            KeySource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    Ok(flag_handle)
}

#[doc(hidden)]
pub trait ReplaceWhereInput<Stencil, Pred> {
    type Runtime: Runtime;
    type Item;
    type Output;

    fn replace_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Stencil, Pred> ReplaceWhereInput<Stencil, Pred> for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        super::ensure_same_len(self.source.len(), stencil.len())?;
        let flags = stencil.selection_handles_with_policy(policy, false)?;
        Ok(SoA1 {
            source: replace_expr_with_flags_with_policy(
                policy,
                &self.source,
                replacement,
                &flags.flag,
            )?,
        })
    }
}

impl<Source, Stencil, Pred> ReplaceWhereInput<Stencil, Pred> for Source
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as ReplaceWhereInput<Stencil, Pred>>::replace_where_input(
            SoA1 { source: self },
            policy,
            replacement,
            stencil,
            pred,
        )
    }
}

impl<Source, Stencil, Pred> ReplaceWhereInput<Stencil, Pred> for (Source,)
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Stencil: super::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = (Source::Item,);
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_where_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <Source as ReplaceWhereInput<Stencil, Pred>>::replace_where_input(
            self.0,
            policy,
            replacement.0,
            stencil,
            pred,
        )
    }
}

macro_rules! impl_replace_where_tuple {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident: $first_index:tt, $( $field:ident: $index:tt ),+ }
    ) => {
        impl<$first, $( $rest ),+, Stencil, Pred> ReplaceWhereInput<Stencil, Pred>
            for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: StorageKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            Stencil: super::SelectionStencil<Pred, Runtime = <$first as KernelColumn>::Runtime>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Item = (
                impl_replace_where_tuple!(@item_ty $first),
                $( impl_replace_where_tuple!(@item_ty $rest) ),+
            );
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn replace_where_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                replacement: Self::Item,
                stencil: Stencil,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                super::ensure_same_len(self.$first_field.len(), stencil.len())?;
                let flags = stencil.selection_handles_with_policy(policy, false)?;
                Ok($name {
                    $first_field: replace_expr_with_flags_with_policy(policy,
                        &self.$first_field,
                        replacement.$first_index,
                        &flags.flag,
                    )?,
                    $(
                        $field: replace_expr_with_flags_with_policy(policy,
                            &self.$field,
                            replacement.$index,
                            &flags.flag,
                        )?,
                    )+
                })
            }
        }
    };
}

impl_replace_where_tuple!(SoA2<A, B> { left: 0, right: 1 });
impl_replace_where_tuple!(SoA3<A, B, C> { first: 0, second: 1, third: 2 });

macro_rules! impl_readonly_replace_where_tuple {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident: $first_index:tt, $( $field:ident: $index:tt ),+ }
    ) => {
        impl<$first, $( $rest ),+, Stencil, Pred> ReplaceWhereInput<Stencil, Pred>
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
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Item = (
                impl_readonly_replace_where_tuple!(@item_ty $first),
                $( impl_readonly_replace_where_tuple!(@item_ty $rest) ),+
            );
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn replace_where_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                replacement: Self::Item,
                stencil: Stencil,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                super::ensure_same_len(ReadOnlySoA::len(&self), stencil.len())?;
                let flags = stencil.selection_handles_with_policy(policy, false)?;
                Ok($output {
                    $first_field: replace_expr_with_flags_with_policy(policy,
                        &self.$first_field,
                        replacement.$first_index,
                        &flags.flag,
                    )?,
                    $(
                        $field: replace_expr_with_flags_with_policy(policy,
                            &self.$field,
                            replacement.$index,
                            &flags.flag,
                        )?,
                    )+
                })
            }
        }
    };
}

impl_readonly_replace_where_tuple!(SoAView2 -> SoA2<A, B> { left: 0, right: 1 });
impl_readonly_replace_where_tuple!(SoAView3 -> SoA3<A, B, C> { first: 0, second: 1, third: 2 });

macro_rules! impl_replace_where_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Stencil, Pred> ReplaceWhereInput<Stencil, Pred> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: ReplaceWhereInput<Stencil, Pred>,
        {
            type Runtime = <$view<$( $ty ),+> as ReplaceWhereInput<Stencil, Pred>>::Runtime;
            type Item = <$view<$( $ty ),+> as ReplaceWhereInput<Stencil, Pred>>::Item;
            type Output = <$view<$( $ty ),+> as ReplaceWhereInput<Stencil, Pred>>::Output;

            fn replace_where_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                replacement: Self::Item,
                stencil: Stencil,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as ReplaceWhereInput<Stencil, Pred>>::replace_where_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    replacement,
                    stencil,
                    pred,
                )
            }
        }
    };
}

impl_replace_where_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_replace_where_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

#[doc(hidden)]
pub trait UniqueInput<Pred> {
    type Runtime: Runtime;
    type Output;

    fn unique_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

#[doc(hidden)]
pub trait UniqueByKeyInput<Values, Eq> {
    type Runtime: Runtime;
    type Output;

    fn unique_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error>;
}

impl<KeySource, ValueSource, Eq> UniqueByKeyInput<SoA1<ValueSource>, Eq> for SoA1<KeySource>
where
    Self: SoA<Item = (KeySource::Item,), Scalar = KeySource::Item>,
    SoA1<ValueSource>: SoA<Item = (ValueSource::Item,), Scalar = ValueSource::Item>,
    KeySource: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Eq: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn unique_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: SoA1<ValueSource>,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        SoA::validate(&values)?;
        super::ensure_same_len(values.source.len(), self.source.len())?;
        let flags = key_run_flags::<KeySource, Eq>(policy, &self.source)?;
        let keys =
            super::device_expr_compact_with_flags_with_policy(policy, &self.source, flags.clone())?;
        let values =
            super::device_expr_compact_with_flags_with_policy(policy, &values.source, flags)?;
        Ok((SoA1 { source: keys }, SoA1 { source: values }))
    }
}

impl<KeySource, ValueSource, Eq> UniqueByKeyInput<ValueSource, Eq> for KeySource
where
    KeySource: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    SoA1<KeySource>: UniqueByKeyInput<SoA1<ValueSource>, Eq>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
{
    type Runtime = <SoA1<KeySource> as UniqueByKeyInput<SoA1<ValueSource>, Eq>>::Runtime;
    type Output = <SoA1<KeySource> as UniqueByKeyInput<SoA1<ValueSource>, Eq>>::Output;

    fn unique_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: ValueSource,
        eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        <SoA1<KeySource> as UniqueByKeyInput<SoA1<ValueSource>, Eq>>::unique_by_key_input(
            SoA1 { source: self },
            policy,
            SoA1 { source: values },
            eq,
        )
    }
}

macro_rules! impl_unique_by_key_view_values {
    ($view:ident -> $out:ident < $( $value:ident: $field:ident ),+ >) => {
        impl<KeySource, $( $value ),+, Eq> UniqueByKeyInput<$view<$( $value ),+>, Eq>
            for KeySource
        where
            KeySource: ReadOnlyKernelColumn + KernelColumnAt<S0>,
            $( $value: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>, )+
            $view<$( $value ),+>: ReadOnlySoA,
            KeySource::Item: CubePrimitive + CubeElement,
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            Eq: BinaryPredicateOp<KeySource::Item>,
        {
            type Runtime = KeySource::Runtime;
            type Output = (
                SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
                $out<$( DeviceVec<KeySource::Runtime, <$value as KernelColumn>::Item> ),+>,
            );

            fn unique_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $view<$( $value ),+>,
                _eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                self.validate()?;
                ReadOnlySoA::validate(&values)?;
                let flags = key_run_flags::<KeySource, Eq>(policy, &self)?;
                $(
                    super::ensure_same_len(values.$field.len(), self.len())?;
                )+
                if self.len() == 0 {
                    return Ok((
                        SoA1 {
                            source: policy.empty_device_vec(),
                        },
                        $out {
                            $( $field: policy.empty_device_vec(), )+
                        },
                    ));
                }

                let out_keys = super::device_expr_compact_with_flags_with_policy(policy, &self, flags.clone())?;
                $(
                    let $field = super::device_expr_compact_with_flags_with_policy(policy, &values.$field, flags.clone())?;
                )+

                Ok((SoA1 { source: out_keys }, $out { $( $field ),+ }))
            }
        }
    };
}

impl_unique_by_key_view_values!(SoAView2 -> SoA2<A: left, B: right>);
impl_unique_by_key_view_values!(SoAView3 -> SoA3<A: first, B: second, C: third>);

impl<KeySource, ValueSource, Eq> UniqueByKeyInput<(ValueSource,), Eq> for (KeySource,)
where
    KeySource: UniqueByKeyInput<ValueSource, super::Tuple1Less<Eq>>,
{
    type Runtime = <KeySource as UniqueByKeyInput<ValueSource, super::Tuple1Less<Eq>>>::Runtime;
    type Output = <KeySource as UniqueByKeyInput<ValueSource, super::Tuple1Less<Eq>>>::Output;

    fn unique_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueSource,),
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        <KeySource as UniqueByKeyInput<ValueSource, super::Tuple1Less<Eq>>>::unique_by_key_input(
            self.0,
            policy,
            values.0,
            GpuOp::<super::Tuple1Less<Eq>>::new(),
        )
    }
}

macro_rules! impl_unique_by_key_key_forward {
    ($name:ident < $first:ident, $( $rest:ident ),+ >) => {
        impl<KeySource, $first, $( $rest ),+, Eq> UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>
            for KeySource
        where
            KeySource: StorageKernelColumn + KernelColumnAt<S0>,
            SoA1<KeySource>: UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>,
            KeySource::Item: CubePrimitive + CubeElement,
        {
            type Runtime = <SoA1<KeySource> as UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>>::Runtime;
            type Output = <SoA1<KeySource> as UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>>::Output;

            fn unique_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $name<$first, $( $rest ),+>,
                eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                <SoA1<KeySource> as UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>>::unique_by_key_input(
                    SoA1 { source: self },
            policy,
            values,
            eq,
        )
    }
        }
    };
}

impl_unique_by_key_key_forward!(SoA2<A, B>);
impl_unique_by_key_key_forward!(SoA3<A, B, C>);

macro_rules! impl_unique_by_key_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<KeySource, $first, $( $rest ),+, Eq> UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>
            for SoA1<KeySource>
        where
            Self: SoA<Item = (KeySource::Item,), Scalar = KeySource::Item>,
            $name<$first, $( $rest ),+>: SoA,
            KeySource: StorageKernelColumn + KernelColumnAt<S0>,
            $first: StorageKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
            )+
            KeySource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Eq: BinaryPredicateOp<KeySource::Item>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = (
                SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
                $name<
                    DeviceVec<KeySource::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<KeySource::Runtime, <$rest as KernelColumn>::Item> ),+
                >,
            );

            fn unique_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $name<$first, $( $rest ),+>,
                _eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                SoA::validate(&values)?;
                super::ensure_same_len(self.source.len(), values.$first_field.len())?;
                $(
                    super::ensure_same_len(self.source.len(), values.$field.len())?;
                )+

                let len = self.source.len();
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let flags = key_run_flags::<KeySource, Eq>(policy, &self.source)?;
                let control = select::handles_from_flags(
                    policy,
                    len,
                    len_u32,
                    flags,
                    policy.empty_handle(),
                )?;
                let count = select::selected_count(policy, &control)?;

                let out_keys = super::device_expr_compact_with_selection_with_policy(
                    policy,
                    &self.source,
                    &control,
                    count,
                )?;
                let $first_field = super::device_expr_compact_with_selection_with_policy(
                    policy,
                    &values.$first_field,
                    &control,
                    count,
                )?;
                $(
                    let $field = super::device_expr_compact_with_selection_with_policy(
                        policy,
                        &values.$field,
                        &control,
                        count,
                    )?;
                )+
                Ok((SoA1 { source: out_keys }, $name { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_unique_by_key_input!(SoA2<A, B> { left, right });
impl_unique_by_key_input!(SoA3<A, B, C> { first, second, third });

impl<Source, Pred> UniqueInput<Pred> for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn unique_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let flags = key_run_flags::<Source, Pred>(policy, &self.source)?;
        Ok(SoA1 {
            source: super::device_expr_compact_with_flags_with_policy(policy, &self.source, flags)?,
        })
    }
}

impl<Source, Pred> UniqueInput<Pred> for Source
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn unique_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as UniqueInput<Pred>>::unique_input(SoA1 { source: self }, policy, pred)
    }
}

impl<Source, Pred> UniqueInput<Pred> for (Source,)
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn unique_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <Source as UniqueInput<super::Tuple1Less<Pred>>>::unique_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Pred>>::new(),
        )
    }
}

macro_rules! impl_unique_tuple {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> UniqueInput<Pred> for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: StorageKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: BinaryPredicateOp<(
                impl_unique_tuple!(@item_ty $first),
                $( impl_unique_tuple!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn unique_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let len = self.$first_field.len();
                let $first_field = stage_sequence_column(policy, &self.$first_field)?;
                $(
                    let $field = stage_sequence_column(policy, &self.$field)?;
                )+

                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = policy.client();
                let flag = client.empty(len * std::mem::size_of::<u32>());

                if len != 0 {
                    let block_count_u32 = sequence_block_count(len)?;
                    unsafe {
                        $kernel_name::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            <$first as KernelColumn>::Expr,
                            $( <$rest as KernelColumn>::Expr, )+
                            Pred,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                            unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                            $(
                                unsafe { BufferArg::from_raw_parts($field.slot0.0.clone(), $field.slot0.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot1.0.clone(), $field.slot1.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot2.0.clone(), $field.slot2.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot3.0.clone(), $field.slot3.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot_offsets.clone(), 4) },
                            )+
                            unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                        );
                    }
                }

                let handles = select::handles_from_flags(
                    policy,
                    len,
                    len_u32,
                    flag,
                    policy.empty_handle(),
                )?;
                let count = select::selected_count(policy, &handles)?;

                Ok($name {
                    $first_field: super::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$first_field,
                        &handles,
                        count,
                    )?,
                    $(
                        $field: super::device_expr_compact_with_selection_with_policy(
                            policy,
                            &self.$field,
                            &handles,
                            count,
                        )?,
                    )+
                })
            }
        }
    };
}

impl_unique_tuple!(SoA2<A, B> { left, right }, tuple2_unique_device_expr_flags_kernel);
impl_unique_tuple!(SoA3<A, B, C> { first, second, third }, tuple3_unique_device_expr_flags_kernel);

macro_rules! impl_readonly_unique_tuple {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> UniqueInput<Pred> for $input<$first, $( $rest ),+>
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
            Pred: BinaryPredicateOp<(
                impl_readonly_unique_tuple!(@item_ty $first),
                $( impl_readonly_unique_tuple!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn unique_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let len = self.$first_field.len();
                let $first_field = stage_sequence_column(policy, &self.$first_field)?;
                $(
                    let $field = stage_sequence_column(policy, &self.$field)?;
                )+

                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = policy.client();
                let flag = client.empty(len * std::mem::size_of::<u32>());

                if len != 0 {
                    let block_count_u32 = sequence_block_count(len)?;
                    unsafe {
                        $kernel_name::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            <$first as KernelColumn>::Expr,
                            $( <$rest as KernelColumn>::Expr, )+
                            Pred,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                            unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                            $(
                                unsafe { BufferArg::from_raw_parts($field.slot0.0.clone(), $field.slot0.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot1.0.clone(), $field.slot1.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot2.0.clone(), $field.slot2.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot3.0.clone(), $field.slot3.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot_offsets.clone(), 4) },
                            )+
                            unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                        );
                    }
                }

                let handles = select::handles_from_flags(
                    policy,
                    len,
                    len_u32,
                    flag,
                    policy.empty_handle(),
                )?;
                let count = select::selected_count(policy, &handles)?;

                Ok($output {
                    $first_field: super::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$first_field,
                        &handles,
                        count,
                    )?,
                    $(
                        $field: super::device_expr_compact_with_selection_with_policy(
                            policy,
                            &self.$field,
                            &handles,
                            count,
                        )?,
                    )+
                })
            }
        }
    };
}

impl_readonly_unique_tuple!(SoAView2 -> SoA2<A, B> { left, right }, tuple2_unique_device_expr_flags_kernel);
impl_readonly_unique_tuple!(SoAView3 -> SoA3<A, B, C> { first, second, third }, tuple3_unique_device_expr_flags_kernel);

macro_rules! impl_unique_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Pred> UniqueInput<Pred> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: UniqueInput<Pred>,
        {
            type Runtime = <$view<$( $ty ),+> as UniqueInput<Pred>>::Runtime;
            type Output = <$view<$( $ty ),+> as UniqueInput<Pred>>::Output;

            fn unique_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as UniqueInput<Pred>>::unique_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    pred,
                )
            }
        }
    };
}

impl_unique_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_unique_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

fn replace_expr_with_flags_with_policy<Source>(
    policy: &CubePolicy<Source::Runtime>,
    input: &Source,
    replacement: Source::Item,
    flag: &cubecl::server::Handle,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    input.validate()?;
    let len = input.len();
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<Source::Item>());
    let block_count_u32 = sequence_block_count(len)?;
    let replacement_values = [replacement];
    let replacement_handle = client.create_from_slice(Source::Item::as_bytes(&replacement_values));
    let bindings = input.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);

    unsafe {
        replace_device_expr_with_flags_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(replacement_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

/// Replaces elements whose staged stencil flag satisfies `Pred`.
pub fn replace_where<R, Input, Stencil, Pred>(
    policy: &CubePolicy<R>,
    input: Input,
    replacement: <Input as ReplaceWhereInput<Stencil, Pred>>::Item,
    stencil: Stencil,
    _pred: Pred,
) -> Result<<<Input as ReplaceWhereInput<Stencil, Pred>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Input: ReplaceWhereInput<Stencil, Pred, Runtime = R>,
    <Input as ReplaceWhereInput<Stencil, Pred>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        input.replace_where_input(policy, replacement, stencil, GpuOp::<Pred>::new())?,
    )
}

/// Removes consecutive duplicates.
pub fn unique<R, Input, Pred>(
    policy: &CubePolicy<R>,
    input: Input,
    _pred: Pred,
) -> Result<<<Input as UniqueInput<Pred>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Input: UniqueInput<Pred, Runtime = R>,
    <Input as UniqueInput<Pred>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(policy, input.unique_input(policy, GpuOp::<Pred>::new())?)
}

/// Removes consecutive duplicate keys and carries the first value for each key.
pub fn unique_by_key<R, Keys, Values, Eq>(
    policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _eq: Eq,
) -> Result<<<Keys as UniqueByKeyInput<Values, Eq>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Keys: UniqueByKeyInput<Values, Eq, Runtime = R>,
    <Keys as UniqueByKeyInput<Values, Eq>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        keys.unique_by_key_input(policy, values, GpuOp::<Eq>::new())?,
    )
}
