use super::super::*;
use crate::detail::control::ReduceByKeyControl;

pub(crate) trait KernelReduceByKeyKeys<KeyEq>: Sized {
    type Runtime: Runtime;
    type OutputKeys;
    type Control;

    fn reduce_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, Self::Control), Error>;
}

pub(crate) trait KernelReduceByKeyValues<Control, KeyEq, Op>: Sized {
    type Runtime: Runtime;
    type Init;
    type OutputValues;

    fn reduce_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &Control,
        init: Self::Init,
    ) -> Result<Self::OutputValues, Error>;
}

#[allow(dead_code)]

pub(crate) trait KernelReduceByKeyCall<Values, KeyEq, Op>: Sized {
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

#[allow(dead_code)]
impl<KeySource, KeyEq> KernelReduceByKeyKeys<KeyEq> for KeySource
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;
    type Control = ReduceByKeyControl<KeySource::Runtime, KeySource::Item, KeySource::Expr, KeyEq>;

    fn reduce_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, Self::Control), Error> {
        <KeySource as KernelColumn>::validate(&self)?;
        let len = <KeySource as KernelColumn>::len(&self);
        if len == 0 {
            return Ok((
                DeviceSoA1 {
                    source: policy.empty_device_vec(),
                },
                ReduceByKeyControl {
                    key_bindings: KernelColumnBindings::empty(policy.client()),
                    head_flags: policy.empty_handle(),
                    end_flags: policy.empty_handle(),
                    len,
                    len_u32: 0,
                    _marker: std::marker::PhantomData,
                },
            ));
        }

        let client = policy.client();
        let head_flags =
            super::scan::scan_by_key_head_flags_read::<KeySource, KeyEq>(policy, &self)?;
        let key_bindings = <KeySource as KernelColumn>::stage(&self, policy)?;
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let end_flags = client.empty(len * std::mem::size_of::<u32>());
        let key_slot0 = key_bindings.slots.first().unwrap();
        let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
        let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
        let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
        let key_offsets = key_bindings.slot_offsets_handle(client)?;
        let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_device_expr_key_end_flags_kernel::launch_unchecked::<
                KeySource::Item,
                KeySource::Expr,
                KeyEq,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1),
                BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1),
                BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1),
                BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1),
                BufferArg::from_raw_parts(key_offsets.clone(), 4),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(end_flags.clone(), len),
            );
        }

        let out_keys = crate::detail::api::device_expr_compact_with_flags_with_policy(
            policy,
            &self,
            end_flags.clone(),
        )?;
        Ok((
            DeviceSoA1 { source: out_keys },
            ReduceByKeyControl {
                key_bindings,
                head_flags,
                end_flags,
                len,
                len_u32,
                _marker: std::marker::PhantomData,
            },
        ))
    }
}

impl<KeySource, KeyEq> KernelReduceByKeyKeys<KeyEq> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
    crate::detail::api::Tuple1Less<KeyEq>: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;
    type Control = ReduceByKeyControl<
        KeySource::Runtime,
        KeySource::Item,
        KeySource::Expr,
        crate::detail::api::Tuple1Less<KeyEq>,
    >;

    fn reduce_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, Self::Control), Error> {
        <KeySource as KernelReduceByKeyKeys<crate::detail::api::Tuple1Less<KeyEq>>>::reduce_by_key_control(
            self.0,
            policy,
        )
    }
}

impl<ValueSource, K, KeyExpr, KeyPred, KeyEq, Op>
    KernelReduceByKeyValues<
        ReduceByKeyControl<ValueSource::Runtime, K, KeyExpr, KeyPred>,
        KeyEq,
        Op,
    > for (ValueSource,)
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    K: Scalar + 'static,
    KeyExpr: DeviceGpuExpr<K>,
    KeyPred: BinaryPredicateOp<K>,
    Op: BinaryOp<(ValueSource::Item,)>,
{
    type Runtime = ValueSource::Runtime;
    type Init = (ValueSource::Item,);
    type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn reduce_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ReduceByKeyControl<ValueSource::Runtime, K, KeyExpr, KeyPred>,
        init: Self::Init,
    ) -> Result<Self::OutputValues, Error> {
        <ValueSource as KernelColumn>::validate(&self.0)?;
        ensure_same_len(<ValueSource as KernelColumn>::len(&self.0), control.len)?;
        if control.len == 0 {
            return Ok(DeviceSoA1 {
                source: policy.empty_device_vec(),
            });
        }

        let client = policy.client();
        let value_bindings = ValueSource::stage(&self.0, policy)?;
        let inclusive = primitive_scan::inclusive_scan_by_key_device_expr::<
            ValueSource::Runtime,
            K,
            ValueSource::Item,
            KeyExpr,
            ValueSource::Expr,
            KeyPred,
            crate::detail::api::Tuple1BinaryOp<Op>,
        >(policy, &control.key_bindings, &value_bindings, control.len)?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
        let init_handle = client.create_from_slice(ValueSource::Item::as_bytes(&[init.0]));
        let reduced_value_handle =
            client.empty(control.len * std::mem::size_of::<ValueSource::Item>());
        let num_blocks = control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_apply_init_kernel::launch_unchecked::<
                ValueSource::Item,
                Op,
                ValueSource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.handle.clone(), control.len),
                BufferArg::from_raw_parts(init_handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_value_handle.clone(), control.len),
            );
        }

        let handles = select::handles_from_flags(
            policy,
            control.len,
            control.len_u32,
            control.end_flags.clone(),
            reduced_value_handle,
        )?;
        Ok(DeviceSoA1 {
            source: select::compact::<ValueSource::Runtime, ValueSource::Item>(policy, handles)?,
        })
    }
}

impl<ValueA, ValueB, K, KeyExpr, KeyPred, KeyEq, Op>
    KernelReduceByKeyValues<ReduceByKeyControl<ValueA::Runtime, K, KeyExpr, KeyPred>, KeyEq, Op>
    for (ValueA, ValueB)
where
    ValueA: KernelColumn + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = ValueA::Runtime> + KernelColumnAt<S0>,
    ValueA::Item: Scalar + 'static,
    ValueB::Item: Scalar + 'static,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    K: Scalar + 'static,
    KeyExpr: DeviceGpuExpr<K>,
    KeyPred: BinaryPredicateOp<K>,
    Op: BinaryOp<(ValueA::Item, ValueB::Item)>,
{
    type Runtime = ValueA::Runtime;
    type Init = (ValueA::Item, ValueB::Item);
    type OutputValues = DeviceSoA2<
        DeviceVec<ValueA::Runtime, ValueA::Item>,
        DeviceVec<ValueA::Runtime, ValueB::Item>,
    >;

    fn reduce_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ReduceByKeyControl<ValueA::Runtime, K, KeyExpr, KeyPred>,
        init: Self::Init,
    ) -> Result<Self::OutputValues, Error> {
        validate_columns2(&self.0, &self.1)?;
        ensure_same_len(<ValueA as KernelColumn>::len(&self.0), control.len)?;
        if control.len == 0 {
            return Ok(DeviceSoA2 {
                left: policy.empty_device_vec(),
                right: policy.empty_device_vec(),
            });
        }

        let client = policy.client();
        let left_bindings = ValueA::stage(&self.0, policy)?;
        let right_bindings = ValueB::stage(&self.1, policy)?;
        let inclusive = primitive_scan::inclusive_scan_tuple2_by_key_values_device_expr::<
            ValueA::Runtime,
            K,
            ValueA::Item,
            ValueB::Item,
            KeyExpr,
            ValueA::Expr,
            ValueB::Expr,
            KeyPred,
            Op,
        >(
            policy,
            &control.key_bindings,
            &left_bindings,
            &right_bindings,
            control.len,
        )?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let reduced_a_handle = client.empty(control.len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(control.len * std::mem::size_of::<ValueB::Item>());
        let num_blocks = control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple2_apply_init_kernel::launch_unchecked::<
                ValueA::Item,
                ValueB::Item,
                Op,
                ValueA::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.left.handle.clone(), control.len),
                BufferArg::from_raw_parts(inclusive.right.handle.clone(), control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), control.len),
            );
        }

        let left_handles = select::handles_from_flags(
            policy,
            control.len,
            control.len_u32,
            control.end_flags.clone(),
            reduced_a_handle,
        )?;
        let right_handles = select::handles_from_flags(
            policy,
            control.len,
            control.len_u32,
            control.end_flags.clone(),
            reduced_b_handle,
        )?;
        Ok(DeviceSoA2 {
            left: select::compact::<ValueA::Runtime, ValueA::Item>(policy, left_handles)?,
            right: select::compact::<ValueA::Runtime, ValueB::Item>(policy, right_handles)?,
        })
    }
}

impl<ValueA, ValueB, ValueC, K, KeyExpr, KeyPred, KeyEq, Op>
    KernelReduceByKeyValues<ReduceByKeyControl<ValueA::Runtime, K, KeyExpr, KeyPred>, KeyEq, Op>
    for (ValueA, ValueB, ValueC)
where
    ValueA: KernelColumn + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = ValueA::Runtime> + KernelColumnAt<S0>,
    ValueC: KernelColumn<Runtime = ValueA::Runtime> + KernelColumnAt<S0>,
    ValueA::Item: Scalar + 'static,
    ValueB::Item: Scalar + 'static,
    ValueC::Item: Scalar + 'static,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    K: Scalar + 'static,
    KeyExpr: DeviceGpuExpr<K>,
    KeyPred: BinaryPredicateOp<K>,
    Op: BinaryOp<(ValueA::Item, ValueB::Item, ValueC::Item)>,
{
    type Runtime = ValueA::Runtime;
    type Init = (ValueA::Item, ValueB::Item, ValueC::Item);
    type OutputValues = DeviceSoA3<
        DeviceVec<ValueA::Runtime, ValueA::Item>,
        DeviceVec<ValueA::Runtime, ValueB::Item>,
        DeviceVec<ValueA::Runtime, ValueC::Item>,
    >;

    fn reduce_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ReduceByKeyControl<ValueA::Runtime, K, KeyExpr, KeyPred>,
        init: Self::Init,
    ) -> Result<Self::OutputValues, Error> {
        validate_columns3(&self.0, &self.1, &self.2)?;
        ensure_same_len(<ValueA as KernelColumn>::len(&self.0), control.len)?;
        if control.len == 0 {
            return Ok(DeviceSoA3 {
                first: policy.empty_device_vec(),
                second: policy.empty_device_vec(),
                third: policy.empty_device_vec(),
            });
        }

        let client = policy.client();
        let first_bindings = ValueA::stage(&self.0, policy)?;
        let second_bindings = ValueB::stage(&self.1, policy)?;
        let third_bindings = ValueC::stage(&self.2, policy)?;
        let inclusive = primitive_scan::inclusive_scan_tuple3_by_key_values_device_expr::<
            ValueA::Runtime,
            K,
            ValueA::Item,
            ValueB::Item,
            ValueC::Item,
            KeyExpr,
            ValueA::Expr,
            ValueB::Expr,
            ValueC::Expr,
            KeyPred,
            Op,
        >(
            policy,
            &control.key_bindings,
            &first_bindings,
            &second_bindings,
            &third_bindings,
            control.len,
        )?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let init_c = client.create_from_slice(ValueC::Item::as_bytes(&[init.2]));
        let reduced_a_handle = client.empty(control.len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(control.len * std::mem::size_of::<ValueB::Item>());
        let reduced_c_handle = client.empty(control.len * std::mem::size_of::<ValueC::Item>());
        let num_blocks = control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple3_apply_init_kernel::launch_unchecked::<
                ValueA::Item,
                ValueB::Item,
                ValueC::Item,
                Op,
                ValueA::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.first.handle.clone(), control.len),
                BufferArg::from_raw_parts(inclusive.second.handle.clone(), control.len),
                BufferArg::from_raw_parts(inclusive.third.handle.clone(), control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(init_c.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), control.len),
                BufferArg::from_raw_parts(reduced_c_handle.clone(), control.len),
            );
        }

        let first_handles = select::handles_from_flags(
            policy,
            control.len,
            control.len_u32,
            control.end_flags.clone(),
            reduced_a_handle,
        )?;
        let second_handles = select::handles_from_flags(
            policy,
            control.len,
            control.len_u32,
            control.end_flags.clone(),
            reduced_b_handle,
        )?;
        let third_handles = select::handles_from_flags(
            policy,
            control.len,
            control.len_u32,
            control.end_flags.clone(),
            reduced_c_handle,
        )?;
        Ok(DeviceSoA3 {
            first: select::compact::<ValueA::Runtime, ValueA::Item>(policy, first_handles)?,
            second: select::compact::<ValueA::Runtime, ValueB::Item>(policy, second_handles)?,
            third: select::compact::<ValueA::Runtime, ValueC::Item>(policy, third_handles)?,
        })
    }
}
