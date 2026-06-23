use super::memory::{MaterializeOutput, materialize};
use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp},
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoAView1,
        SoAView2, SoAView3,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::{reduce as primitive_reduce, scan as primitive_scan, select},
};
use cubecl::prelude::*;

/// Input accepted by [`reduce`].
#[doc(hidden)]
pub trait ReduceInput<Op> {
    /// CubeCL runtime used by this input.
    type Runtime: Runtime;
    /// Initial value type.
    type Init;
    /// Reduction output type.
    type Output;

    /// Reduces this input.
    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Op> ReduceInput<Op> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Init = (Source::Item,);
    type Output = (Source::Item,);

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let bindings = self.source.stage(policy)?;
        primitive_reduce::reduce_tuple1_device_expr::<_, _, Source::Expr, Op>(
            policy,
            &bindings,
            self.source.len(),
            init,
        )
    }
}

impl<Source, Op> ReduceInput<Op> for (Source,)
where
    SoAView1<Source>: ReduceInput<Op>,
{
    type Runtime = <SoAView1<Source> as ReduceInput<Op>>::Runtime;
    type Init = <SoAView1<Source> as ReduceInput<Op>>::Init;
    type Output = <SoAView1<Source> as ReduceInput<Op>>::Output;

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as ReduceInput<Op>>::reduce_input(
            SoAView1 { source: self.0 },
            policy,
            init,
            op,
        )
    }
}

macro_rules! impl_reduce_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $reduce_fn:ident) => {
        impl<$first, $( $rest ),+, Op> ReduceInput<Op> for $name<$first, $( $rest ),+>
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
            type Output = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );

            fn reduce_input(
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
                primitive_reduce::$reduce_fn::<
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

impl_reduce_input!(SoAView2<A, B> { left, right } => reduce_tuple2_device_expr);
impl_reduce_input!(SoAView3<A, B, C> { first, second, third } => reduce_tuple3_device_expr);

impl<Left, Right, Op> ReduceInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: ReduceInput<Op>,
{
    type Runtime = <SoAView2<Left, Right> as ReduceInput<Op>>::Runtime;
    type Init = <SoAView2<Left, Right> as ReduceInput<Op>>::Init;
    type Output = <SoAView2<Left, Right> as ReduceInput<Op>>::Output;

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ReduceInput<Op>>::reduce_input(
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

impl<First, Second, Third, Op> ReduceInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ReduceInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Runtime;
    type Init = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Init;
    type Output = <SoAView3<First, Second, Third> as ReduceInput<Op>>::Output;

    fn reduce_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ReduceInput<Op>>::reduce_input(
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

macro_rules! impl_reduce_soa_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ } => $reduce_fn:ident) => {
        impl<$first, $( $rest ),+, Op> ReduceInput<Op> for $name<$first, $( $rest ),+>
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
            type Output = (
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            );

            fn reduce_input(
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
                primitive_reduce::$reduce_fn::<
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

impl_reduce_soa_input!(SoA2<A, B> { left, right } => reduce_tuple2_device_expr);
impl_reduce_soa_input!(SoA3<A, B, C> { first, second, third } => reduce_tuple3_device_expr);

/// Reduces read-only device input to a host tuple item.
///
/// This is a borrowing algorithm: pass `&DeviceVec` for one column or [`zip`]
/// for multiple read-only columns. No output device storage is allocated.
///
/// [`zip`]: crate::zip
pub fn reduce<Input, Op>(
    policy: &CubePolicy<<Input as ReduceInput<Op>>::Runtime>,
    input: Input,
    init: <Input as ReduceInput<Op>>::Init,
    _op: Op,
) -> Result<<Input as ReduceInput<Op>>::Output, Error>
where
    Input: ReduceInput<Op>,
{
    input.reduce_input(policy, init, GpuOp::<Op>::new())
}

#[doc(hidden)]
pub trait ReduceByKeyCall<Values, KeyEq, Op> {
    type Runtime: Runtime;
    type Init;
    type Output;

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

impl<KeySource, ValueSource, KeyEq, Op> ReduceByKeyCall<(ValueSource,), KeyEq, Op> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
    Op: BinaryOp<(ValueSource::Item,)>,
{
    type Runtime = KeySource::Runtime;
    type Init = (ValueSource::Item,);
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueSource,),
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        self.0.validate()?;
        values.0.validate()?;
        super::ensure_same_len(values.0.len(), self.0.len())?;
        let len = self.0.len();
        if len == 0 {
            return Ok((
                SoA1 {
                    source: policy.empty_device_vec(),
                },
                SoA1 {
                    source: policy.empty_device_vec(),
                },
            ));
        }

        let client = policy.client();
        let key_bindings = self.0.stage(policy)?;
        let value_bindings = values.0.stage(policy)?;
        let inclusive_handle = primitive_scan::inclusive_scan_by_key_device_expr_handle::<
            KeySource::Runtime,
            KeySource::Item,
            ValueSource::Item,
            KeySource::Expr,
            ValueSource::Expr,
            super::Tuple1Less<KeyEq>,
            super::Tuple1BinaryOp<Op>,
        >(policy, &key_bindings, &value_bindings, len)?;

        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let init_handle = client.create_from_slice(ValueSource::Item::as_bytes(&[init.0]));
        let flag_handle = client.empty(len * std::mem::size_of::<u32>());
        let reduced_value_handle = client.empty(len * std::mem::size_of::<ValueSource::Item>());
        let key_slot0 = key_bindings.slots.first().unwrap();
        let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
        let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
        let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
        let key_offsets = key_bindings.slot_offsets_handle(client)?;
        let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_device_expr_end_flags_kernel::launch_unchecked::<
                KeySource::Item,
                ValueSource::Item,
                KeySource::Expr,
                super::Tuple1Less<KeyEq>,
                super::Tuple1BinaryOp<Op>,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(inclusive_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_value_handle.clone(), len) },
            );
        }

        let out_keys = super::device_expr_compact_with_flags_with_policy(
            policy,
            &self.0,
            flag_handle.clone(),
        )?;
        let handles =
            select::handles_from_flags(policy, len, len_u32, flag_handle, reduced_value_handle)?;
        let out_values = select::compact::<KeySource::Runtime, ValueSource::Item>(policy, handles)?;
        Ok((SoA1 { source: out_keys }, SoA1 { source: out_values }))
    }
}

impl<KeySource, ValueA, ValueB, KeyEq, Op> ReduceByKeyCall<(ValueA, ValueB), KeyEq, Op>
    for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
    Op: BinaryOp<(ValueA::Item, ValueB::Item)>,
{
    type Runtime = KeySource::Runtime;
    type Init = (ValueA::Item, ValueB::Item);
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA2<
            DeviceVec<KeySource::Runtime, ValueA::Item>,
            DeviceVec<KeySource::Runtime, ValueB::Item>,
        >,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueA, ValueB),
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        self.0.validate()?;
        values.0.validate()?;
        values.1.validate()?;
        super::ensure_same_len(values.0.len(), self.0.len())?;
        super::ensure_same_len(values.1.len(), self.0.len())?;
        let len = self.0.len();
        if len == 0 {
            return Ok((
                SoA1 {
                    source: policy.empty_device_vec(),
                },
                SoA2 {
                    left: policy.empty_device_vec(),
                    right: policy.empty_device_vec(),
                },
            ));
        }

        let client = policy.client();
        let key_bindings = self.0.stage(policy)?;
        let a_bindings = values.0.stage(policy)?;
        let b_bindings = values.1.stage(policy)?;
        let (inclusive_a, inclusive_b) =
            primitive_scan::inclusive_scan_tuple2_by_key_values_device_expr_handle::<
                KeySource::Runtime,
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                KeySource::Expr,
                ValueA::Expr,
                ValueB::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
            >(policy, &key_bindings, &a_bindings, &b_bindings, len)?;

        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let flag_handle = client.empty(len * std::mem::size_of::<u32>());
        let reduced_a_handle = client.empty(len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(len * std::mem::size_of::<ValueB::Item>());
        let key_slot0 = key_bindings.slots.first().unwrap();
        let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
        let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
        let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
        let key_offsets = key_bindings.slot_offsets_handle(client)?;
        let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple2_device_expr_end_flags_kernel::launch_unchecked::<
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                KeySource::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(inclusive_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(inclusive_b.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_b_handle.clone(), len) },
            );
        }

        let out_keys = super::device_expr_compact_with_flags_with_policy(
            policy,
            &self.0,
            flag_handle.clone(),
        )?;
        let left_handles = select::handles_from_flags(
            policy,
            len,
            len_u32,
            flag_handle.clone(),
            reduced_a_handle,
        )?;
        let right_handles =
            select::handles_from_flags(policy, len, len_u32, flag_handle, reduced_b_handle)?;
        let left = select::compact::<KeySource::Runtime, ValueA::Item>(policy, left_handles)?;
        let right = select::compact::<KeySource::Runtime, ValueB::Item>(policy, right_handles)?;

        Ok((SoA1 { source: out_keys }, SoA2 { left, right }))
    }
}

impl<KeySource, ValueA, ValueB, ValueC, KeyEq, Op>
    ReduceByKeyCall<(ValueA, ValueB, ValueC), KeyEq, Op> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueA: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    ValueC: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
    Op: BinaryOp<(ValueA::Item, ValueB::Item, ValueC::Item)>,
{
    type Runtime = KeySource::Runtime;
    type Init = (ValueA::Item, ValueB::Item, ValueC::Item);
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA3<
            DeviceVec<KeySource::Runtime, ValueA::Item>,
            DeviceVec<KeySource::Runtime, ValueB::Item>,
            DeviceVec<KeySource::Runtime, ValueC::Item>,
        >,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueA, ValueB, ValueC),
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        self.0.validate()?;
        values.0.validate()?;
        values.1.validate()?;
        values.2.validate()?;
        super::ensure_same_len(values.0.len(), self.0.len())?;
        super::ensure_same_len(values.1.len(), self.0.len())?;
        super::ensure_same_len(values.2.len(), self.0.len())?;
        let len = self.0.len();
        if len == 0 {
            return Ok((
                SoA1 {
                    source: policy.empty_device_vec(),
                },
                SoA3 {
                    first: policy.empty_device_vec(),
                    second: policy.empty_device_vec(),
                    third: policy.empty_device_vec(),
                },
            ));
        }

        let client = policy.client();
        let key_bindings = self.0.stage(policy)?;
        let a_bindings = values.0.stage(policy)?;
        let b_bindings = values.1.stage(policy)?;
        let c_bindings = values.2.stage(policy)?;
        let (inclusive_a, inclusive_b, inclusive_c) =
            primitive_scan::inclusive_scan_tuple3_by_key_values_device_expr_handle::<
                KeySource::Runtime,
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                ValueC::Item,
                KeySource::Expr,
                ValueA::Expr,
                ValueB::Expr,
                ValueC::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
            >(
                policy,
                &key_bindings,
                &a_bindings,
                &b_bindings,
                &c_bindings,
                len,
            )?;

        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let init_c = client.create_from_slice(ValueC::Item::as_bytes(&[init.2]));
        let flag_handle = client.empty(len * std::mem::size_of::<u32>());
        let reduced_a_handle = client.empty(len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(len * std::mem::size_of::<ValueB::Item>());
        let reduced_c_handle = client.empty(len * std::mem::size_of::<ValueC::Item>());
        let key_slot0 = key_bindings.slots.first().unwrap();
        let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
        let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
        let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
        let key_offsets = key_bindings.slot_offsets_handle(client)?;
        let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple3_device_expr_end_flags_kernel::launch_unchecked::<
                KeySource::Item,
                ValueA::Item,
                ValueB::Item,
                ValueC::Item,
                KeySource::Expr,
                super::Tuple1Less<KeyEq>,
                Op,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(inclusive_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(inclusive_b.clone(), len) },
                unsafe { BufferArg::from_raw_parts(inclusive_c.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(init_c.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_c_handle.clone(), len) },
            );
        }

        let out_keys = super::device_expr_compact_with_flags_with_policy(
            policy,
            &self.0,
            flag_handle.clone(),
        )?;
        let first_handles = select::handles_from_flags(
            policy,
            len,
            len_u32,
            flag_handle.clone(),
            reduced_a_handle,
        )?;
        let second_handles = select::handles_from_flags(
            policy,
            len,
            len_u32,
            flag_handle.clone(),
            reduced_b_handle,
        )?;
        let third_handles =
            select::handles_from_flags(policy, len, len_u32, flag_handle, reduced_c_handle)?;
        let first = select::compact::<KeySource::Runtime, ValueA::Item>(policy, first_handles)?;
        let second = select::compact::<KeySource::Runtime, ValueB::Item>(policy, second_handles)?;
        let third = select::compact::<KeySource::Runtime, ValueC::Item>(policy, third_handles)?;

        Ok((
            SoA1 { source: out_keys },
            SoA3 {
                first,
                second,
                third,
            },
        ))
    }
}

/// Reduces contiguous equal-key runs using read-only keys and values.
///
/// This is a borrowing algorithm: values may be a borrowed column or a read-only
/// SoA from [`zip`](crate::zip). The returned keys and values are owned SoA
/// storage.
pub fn reduce_by_key<R, Keys, Values, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    init: <Keys as ReduceByKeyCall<Values, KeyEq, Op>>::Init,
    _op: Op,
) -> Result<
    <<Keys as ReduceByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Keys: ReduceByKeyCall<Values, KeyEq, Op, Runtime = R>,
    <Keys as ReduceByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        keys.reduce_by_key_call(
            policy,
            values,
            GpuOp::<KeyEq>::new(),
            init,
            GpuOp::<Op>::new(),
        )?,
    )
}
