use super::super::*;
use crate::detail::{
    control::SegmentControl,
    control::{ReduceByKeyControl, ScanByKeyControl, SelectedRankControl},
    device::DeviceColumnView,
};

fn reduce_output_selection_from_end_flags<R: Runtime>(
    policy: &CubePolicy<R>,
    len: usize,
    len_u32: u32,
    end_flags: cubecl::server::Handle,
) -> Result<(SelectedRankControl, usize), Error> {
    let control = select::selected_rank_from_flags(policy, len, len_u32, end_flags)?;
    let count = select::selected_count(policy, &control)?;
    Ok((control, count))
}

fn empty_reduce_by_key_control<R: Runtime>(
    policy: &CubePolicy<R>,
    len: usize,
) -> ReduceByKeyControl<R> {
    let segment =
        SegmentControl::from_head_end_flags(policy.empty_handle(), policy.empty_handle(), len, 0);
    ReduceByKeyControl::from_segment(
        segment,
        crate::detail::control::SelectedRankControl::empty(policy.client()),
        0,
    )
}

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

pub(in crate::detail) struct SegmentedReduceApply<'a, R: Runtime> {
    control: &'a ReduceByKeyControl<R>,
}

impl<'a, R: Runtime> SegmentedReduceApply<'a, R> {
    pub(in crate::detail) fn new(control: &'a ReduceByKeyControl<R>) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn apply_expr<ValueSource, Op>(
        &self,
        policy: &CubePolicy<R>,
        source: &ValueSource,
        init: ValueSource::Item,
    ) -> Result<DeviceVec<R, ValueSource::Item>, Error>
    where
        ValueSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueSource::Item: Scalar + 'static,
        ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
        Op: BinaryOp<(ValueSource::Item,)>,
    {
        let client = policy.client();
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = super::scan::SegmentedScanApply::new(&scan_control);
        let inclusive = scan_apply.inclusive_expr::<ValueSource, Op>(policy, source)?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[self.control.len_u32]));
        let init_handle = client.create_from_slice(ValueSource::Item::as_bytes(&[init]));
        let reduced_value_handle =
            client.empty(self.control.len * std::mem::size_of::<ValueSource::Item>());
        let num_blocks = self
            .control
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
                BufferArg::from_raw_parts(inclusive.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(init_handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_value_handle.clone(), self.control.len),
            );
        }

        let payload_apply = crate::detail::api::SelectedPayloadApply::new(
            &self.control.output_selection,
            self.control.output_count,
        );
        payload_apply.apply_value::<R, ValueSource::Item>(policy, reduced_value_handle)
    }

    pub(in crate::detail) fn apply_expr2<ValueA, ValueB, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &ValueA,
        b: &ValueB,
        init: (ValueA::Item, ValueB::Item),
    ) -> Result<DeviceSoA2<DeviceVec<R, ValueA::Item>, DeviceVec<R, ValueB::Item>>, Error>
    where
        ValueA: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueB: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueA::Item: Scalar + 'static,
        ValueB::Item: Scalar + 'static,
        ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
        ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
        (ValueA::Item, ValueB::Item): MItem<R>,
        Op: BinaryOp<(ValueA::Item, ValueB::Item)>,
    {
        let client = policy.client();
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = super::scan::SegmentedScanApply::new(&scan_control);
        let inclusive = scan_apply.inclusive_expr2::<ValueA, ValueB, Op>(policy, a, b)?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[self.control.len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let reduced_a_handle = client.empty(self.control.len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(self.control.len * std::mem::size_of::<ValueB::Item>());
        let num_blocks = self
            .control
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
                BufferArg::from_raw_parts(inclusive.left.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(inclusive.right.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), self.control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), self.control.len),
            );
        }

        let payload_apply = crate::detail::api::SelectedPayloadApply::new(
            &self.control.output_selection,
            self.control.output_count,
        );
        Ok(DeviceSoA2 {
            left: payload_apply.apply_value::<R, ValueA::Item>(policy, reduced_a_handle)?,
            right: payload_apply.apply_value::<R, ValueB::Item>(policy, reduced_b_handle)?,
        })
    }

    pub(in crate::detail) fn apply_expr3<ValueA, ValueB, ValueC, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &ValueA,
        b: &ValueB,
        c: &ValueC,
        init: (ValueA::Item, ValueB::Item, ValueC::Item),
    ) -> Result<
        DeviceSoA3<
            DeviceVec<R, ValueA::Item>,
            DeviceVec<R, ValueB::Item>,
            DeviceVec<R, ValueC::Item>,
        >,
        Error,
    >
    where
        ValueA: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueB: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueC: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueA::Item: Scalar + 'static,
        ValueB::Item: Scalar + 'static,
        ValueC::Item: Scalar + 'static,
        ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
        ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
        ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
        (ValueA::Item, ValueB::Item, ValueC::Item): MItem<R>,
        Op: BinaryOp<(ValueA::Item, ValueB::Item, ValueC::Item)>,
    {
        let client = policy.client();
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = super::scan::SegmentedScanApply::new(&scan_control);
        let inclusive =
            scan_apply.inclusive_expr3::<ValueA, ValueB, ValueC, Op>(policy, a, b, c)?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[self.control.len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let init_c = client.create_from_slice(ValueC::Item::as_bytes(&[init.2]));
        let reduced_a_handle = client.empty(self.control.len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(self.control.len * std::mem::size_of::<ValueB::Item>());
        let reduced_c_handle = client.empty(self.control.len * std::mem::size_of::<ValueC::Item>());
        let num_blocks = self
            .control
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
                BufferArg::from_raw_parts(inclusive.first.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(inclusive.second.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(inclusive.third.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(init_c.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), self.control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), self.control.len),
                BufferArg::from_raw_parts(reduced_c_handle.clone(), self.control.len),
            );
        }

        let payload_apply = crate::detail::api::SelectedPayloadApply::new(
            &self.control.output_selection,
            self.control.output_count,
        );
        Ok(DeviceSoA3 {
            first: payload_apply.apply_value::<R, ValueA::Item>(policy, reduced_a_handle)?,
            second: payload_apply.apply_value::<R, ValueB::Item>(policy, reduced_b_handle)?,
            third: payload_apply.apply_value::<R, ValueC::Item>(policy, reduced_c_handle)?,
        })
    }
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
    type Control = ReduceByKeyControl<KeySource::Runtime>;

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
                empty_reduce_by_key_control(policy, len),
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

        let (output_selection, output_count) =
            reduce_output_selection_from_end_flags(policy, len, len_u32, end_flags.clone())?;
        let payload_apply =
            crate::detail::api::SelectedPayloadApply::new(&output_selection, output_count);
        let out_keys = payload_apply.apply_expr(policy, &self)?;
        Ok((
            DeviceSoA1 { source: out_keys },
            ReduceByKeyControl::from_segment(
                SegmentControl::from_head_end_flags(head_flags, end_flags, len, len_u32),
                output_selection,
                output_count,
            ),
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
    type Control = ReduceByKeyControl<KeySource::Runtime>;

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

impl<First, Second, KeyEq> KernelReduceByKeyKeys<KeyEq> for (First, Second)
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: Scalar + 'static,
    Second::Item: Scalar + 'static,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    KeyEq: BinaryPredicateOp<(First::Item, Second::Item)>,
{
    type Runtime = First::Runtime;
    type OutputKeys =
        DeviceSoA2<DeviceVec<First::Runtime, First::Item>, DeviceVec<First::Runtime, Second::Item>>;
    type Control = ReduceByKeyControl<First::Runtime>;

    fn reduce_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, Self::Control), Error> {
        <First as KernelColumn>::validate(&self.0)?;
        <Second as KernelColumn>::validate(&self.1)?;
        let len = <First as KernelColumn>::len(&self.0);
        let second_len = <Second as KernelColumn>::len(&self.1);
        if len != second_len {
            return Err(Error::LengthMismatch {
                input: len,
                output: second_len,
            });
        }
        if len == 0 {
            return Ok((
                DeviceSoA2 {
                    left: policy.empty_device_vec(),
                    right: policy.empty_device_vec(),
                },
                empty_reduce_by_key_control(policy, len),
            ));
        }

        let head_flags = super::super::selection::unique_tuple2_flags_read::<First, Second, KeyEq>(
            policy, &self.0, &self.1,
        )?;
        let end_flags =
            crate::detail::impls::end_flags_from_head_flags(policy, head_flags.clone(), len)?;
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let (output_selection, output_count) =
            reduce_output_selection_from_end_flags(policy, len, len_u32, end_flags.clone())?;
        let payload_apply =
            crate::detail::api::SelectedPayloadApply::new(&output_selection, output_count);
        let left = payload_apply.apply_expr(policy, &self.0)?;
        let right = payload_apply.apply_expr(policy, &self.1)?;
        Ok((
            DeviceSoA2 { left, right },
            ReduceByKeyControl::from_segment(
                SegmentControl::from_head_end_flags(head_flags, end_flags, len, len_u32),
                output_selection,
                output_count,
            ),
        ))
    }
}

impl<First, Second, Third, KeyEq> KernelReduceByKeyKeys<KeyEq> for (First, Second, Third)
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: Scalar + 'static,
    Second::Item: Scalar + 'static,
    Third::Item: Scalar + 'static,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    type Runtime = First::Runtime;
    type OutputKeys = DeviceSoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;
    type Control = ReduceByKeyControl<First::Runtime>;

    fn reduce_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, Self::Control), Error> {
        <First as KernelColumn>::validate(&self.0)?;
        <Second as KernelColumn>::validate(&self.1)?;
        <Third as KernelColumn>::validate(&self.2)?;
        let len = <First as KernelColumn>::len(&self.0);
        let second_len = <Second as KernelColumn>::len(&self.1);
        if len != second_len {
            return Err(Error::LengthMismatch {
                input: len,
                output: second_len,
            });
        }
        let third_len = <Third as KernelColumn>::len(&self.2);
        if len != third_len {
            return Err(Error::LengthMismatch {
                input: len,
                output: third_len,
            });
        }
        if len == 0 {
            return Ok((
                DeviceSoA3 {
                    first: policy.empty_device_vec(),
                    second: policy.empty_device_vec(),
                    third: policy.empty_device_vec(),
                },
                empty_reduce_by_key_control(policy, len),
            ));
        }

        let head_flags =
            super::super::selection::unique_tuple3_flags_read::<First, Second, Third, KeyEq>(
                policy, &self.0, &self.1, &self.2,
            )?;
        let end_flags =
            crate::detail::impls::end_flags_from_head_flags(policy, head_flags.clone(), len)?;
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let (output_selection, output_count) =
            reduce_output_selection_from_end_flags(policy, len, len_u32, end_flags.clone())?;
        let payload_apply =
            crate::detail::api::SelectedPayloadApply::new(&output_selection, output_count);
        let first = payload_apply.apply_expr(policy, &self.0)?;
        let second = payload_apply.apply_expr(policy, &self.1)?;
        let third = payload_apply.apply_expr(policy, &self.2)?;
        Ok((
            DeviceSoA3 {
                first,
                second,
                third,
            },
            ReduceByKeyControl::from_segment(
                SegmentControl::from_head_end_flags(head_flags, end_flags, len, len_u32),
                output_selection,
                output_count,
            ),
        ))
    }
}

impl<ValueSource, KeyEq, Op>
    KernelReduceByKeyValues<ReduceByKeyControl<ValueSource::Runtime>, KeyEq, Op> for (ValueSource,)
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Op: BinaryOp<(ValueSource::Item,)>,
{
    type Runtime = ValueSource::Runtime;
    type Init = (ValueSource::Item,);
    type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn reduce_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ReduceByKeyControl<ValueSource::Runtime>,
        init: Self::Init,
    ) -> Result<Self::OutputValues, Error> {
        <ValueSource as KernelColumn>::validate(&self.0)?;
        ensure_same_len(<ValueSource as KernelColumn>::len(&self.0), control.len)?;
        if control.len == 0 {
            return Ok(DeviceSoA1 {
                source: policy.empty_device_vec(),
            });
        }

        let apply = SegmentedReduceApply::new(control);
        Ok(DeviceSoA1 {
            source: apply.apply_expr::<ValueSource, Op>(policy, &self.0, init.0)?,
        })
    }
}

impl<ValueA, ValueB, KeyEq, Op>
    KernelReduceByKeyValues<ReduceByKeyControl<ValueA::Runtime>, KeyEq, Op> for (ValueA, ValueB)
where
    ValueA: KernelColumn + KernelColumnAt<S0>,
    ValueB: KernelColumn<Runtime = ValueA::Runtime> + KernelColumnAt<S0>,
    ValueA::Item: Scalar + 'static,
    ValueB::Item: Scalar + 'static,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
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
        control: &ReduceByKeyControl<ValueA::Runtime>,
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

        let apply = SegmentedReduceApply::new(control);
        apply.apply_expr2::<ValueA, ValueB, Op>(policy, &self.0, &self.1, init)
    }
}

impl<ValueA, ValueB, ValueC, KeyEq, Op>
    KernelReduceByKeyValues<ReduceByKeyControl<ValueA::Runtime>, KeyEq, Op>
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
        control: &ReduceByKeyControl<ValueA::Runtime>,
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

        let apply = SegmentedReduceApply::new(control);
        apply.apply_expr3::<ValueA, ValueB, ValueC, Op>(policy, &self.0, &self.1, &self.2, init)
    }
}

macro_rules! reduce_by_key_tuple7_scanned_values {
    ($policy:ident, $control:ident, $inclusive:ident, $init:expr; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; $op:ty) => {{
        let client = $policy.client();
        let len_handle = client.create_from_slice(u32::as_bytes(&[$control.len_u32]));
        let init_a = client.create_from_slice($ty0::as_bytes(&[$init.0]));
        let init_b = client.create_from_slice($ty1::as_bytes(&[$init.1]));
        let init_c = client.create_from_slice($ty2::as_bytes(&[$init.2]));
        let init_d = client.create_from_slice($ty3::as_bytes(&[$init.3]));
        let init_e = client.create_from_slice($ty4::as_bytes(&[$init.4]));
        let init_f = client.create_from_slice($ty5::as_bytes(&[$init.5]));
        let init_g = client.create_from_slice($ty6::as_bytes(&[$init.6]));
        let reduced_a_handle = client.empty($control.len * std::mem::size_of::<$ty0>());
        let reduced_b_handle = client.empty($control.len * std::mem::size_of::<$ty1>());
        let reduced_c_handle = client.empty($control.len * std::mem::size_of::<$ty2>());
        let reduced_d_handle = client.empty($control.len * std::mem::size_of::<$ty3>());
        let reduced_e_handle = client.empty($control.len * std::mem::size_of::<$ty4>());
        let reduced_f_handle = client.empty($control.len * std::mem::size_of::<$ty5>());
        let reduced_g_handle = client.empty($control.len * std::mem::size_of::<$ty6>());
        let num_blocks = $control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        unsafe {
            reduce_by_key_tuple7_apply_init_kernel::launch_unchecked::<
                $ty0,
                $ty1,
                $ty2,
                $ty3,
                $ty4,
                $ty5,
                $ty6,
                $op,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts($inclusive.0.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.1.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.2.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.3.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.4.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.5.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.6.handle.clone(), $control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(init_c.clone(), 1),
                BufferArg::from_raw_parts(init_d.clone(), 1),
                BufferArg::from_raw_parts(init_e.clone(), 1),
                BufferArg::from_raw_parts(init_f.clone(), 1),
                BufferArg::from_raw_parts(init_g.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_c_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_d_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_e_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_f_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_g_handle.clone(), $control.len),
            );
        }
        let payload_apply = crate::detail::api::SelectedPayloadApply::new(
            &$control.output_selection,
            $control.output_count,
        );
        Ok::<_, Error>((
            payload_apply.apply_value::<R, $ty0>($policy, reduced_a_handle)?,
            payload_apply.apply_value::<R, $ty1>($policy, reduced_b_handle)?,
            payload_apply.apply_value::<R, $ty2>($policy, reduced_c_handle)?,
            payload_apply.apply_value::<R, $ty3>($policy, reduced_d_handle)?,
            payload_apply.apply_value::<R, $ty4>($policy, reduced_e_handle)?,
            payload_apply.apply_value::<R, $ty5>($policy, reduced_f_handle)?,
            payload_apply.apply_value::<R, $ty6>($policy, reduced_g_handle)?,
        ))
    }};
}

impl<'a, R: Runtime> SegmentedReduceApply<'a, R> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn apply_views7<A, B, C, D, E, F, G, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
        init: (A, B, C, D, E, F, G),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
            DeviceVec<R, G>,
        ),
        Error,
    >
    where
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        E: Scalar + 'static,
        F: Scalar + 'static,
        G: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = super::scan::SegmentedScanApply::new(&scan_control);
        let inclusive =
            scan_apply.inclusive_views7::<A, B, C, D, E, F, G, Op>(policy, a, b, c, d, e, f, g)?;
        let control = self.control;
        reduce_by_key_tuple7_scanned_values!(
            policy, control, inclusive, init;
            A, B, C, D, E, F, G;
            Op
        )
    }
}

macro_rules! impl_kernel_reduce_by_key_tuple4_views {
    () => {
        impl<R, A, B, C, D, KeyEq, Op> KernelReduceByKeyValues<ReduceByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
            )
        where
            R: Runtime,
            A: Scalar + 'static,
            B: Scalar + 'static,
            C: Scalar + 'static,
            D: Scalar + 'static,
            Op: BinaryOp<(A, B, C, D)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D);
            type OutputValues = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
            );

            fn reduce_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ReduceByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::OutputValues, Error> {
                if control.len == 0 {
                    return Ok((
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                    ));
                }
                let dummy4 = primitive_range::indices_mindex(policy, self.0.len)?;
                let dummy5 = primitive_range::indices_mindex(policy, self.0.len)?;
                let dummy6 = primitive_range::indices_mindex(policy, self.0.len)?;
                let dummy4 = DeviceColumnView::from_column(&dummy4);
                let dummy5 = DeviceColumnView::from_column(&dummy5);
                let dummy6 = DeviceColumnView::from_column(&dummy6);
                let apply = SegmentedReduceApply::new(control);
                let (a, b, c, d, _, _, _) = apply.apply_views7::<
                                                    A, B, C, D, u32, u32, u32,
                                                    crate::detail::api::Tuple4AsTuple7BinaryOp<Op>,
                                                >(
                                                    policy,
                                                    &self.0,
                                                    &self.1,
                                                    &self.2,
                                                    &self.3,
                                                    &dummy4,
                                                    &dummy5,
                                                    &dummy6,
                                                    (init.0, init.1, init.2, init.3, 0, 0, 0),
                                                )?;
                Ok((a, b, c, d))
            }
        }
    };
}

macro_rules! impl_kernel_reduce_by_key_tuple5_views {
    () => {
        impl<R, A, B, C, D, E, KeyEq, Op> KernelReduceByKeyValues<ReduceByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
                DeviceColumnView<R, E>,
            )
        where
            R: Runtime,
            A: Scalar + 'static,
            B: Scalar + 'static,
            C: Scalar + 'static,
            D: Scalar + 'static,
            E: Scalar + 'static,
            Op: BinaryOp<(A, B, C, D, E)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D, E);
            type OutputValues = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
            );

            fn reduce_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ReduceByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::OutputValues, Error> {
                if control.len == 0 {
                    return Ok((
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                    ));
                }
                let dummy5 = primitive_range::indices_mindex(policy, self.0.len)?;
                let dummy6 = primitive_range::indices_mindex(policy, self.0.len)?;
                let dummy5 = DeviceColumnView::from_column(&dummy5);
                let dummy6 = DeviceColumnView::from_column(&dummy6);
                let apply = SegmentedReduceApply::new(control);
                let (a, b, c, d, e, _, _) = apply.apply_views7::<
                                                    A, B, C, D, E, u32, u32,
                                                    crate::detail::api::Tuple5AsTuple7BinaryOp<Op>,
                                                >(
                                                    policy,
                                                    &self.0,
                                                    &self.1,
                                                    &self.2,
                                                    &self.3,
                                                    &self.4,
                                                    &dummy5,
                                                    &dummy6,
                                                    (init.0, init.1, init.2, init.3, init.4, 0, 0),
                                                )?;
                Ok((a, b, c, d, e))
            }
        }
    };
}

macro_rules! impl_kernel_reduce_by_key_tuple6_views {
    () => {
        impl<R, A, B, C, D, E, F, KeyEq, Op>
            KernelReduceByKeyValues<ReduceByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
                DeviceColumnView<R, E>,
                DeviceColumnView<R, F>,
            )
        where
            R: Runtime,
            A: Scalar + 'static,
            B: Scalar + 'static,
            C: Scalar + 'static,
            D: Scalar + 'static,
            E: Scalar + 'static,
            F: Scalar + 'static,
            Op: BinaryOp<(A, B, C, D, E, F)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D, E, F);
            type OutputValues = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
                DeviceVec<R, F>,
            );

            fn reduce_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ReduceByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::OutputValues, Error> {
                if control.len == 0 {
                    return Ok((
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                    ));
                }
                let dummy6 = primitive_range::indices_mindex(policy, self.0.len)?;
                let dummy6 = DeviceColumnView::from_column(&dummy6);
                let apply = SegmentedReduceApply::new(control);
                let (a, b, c, d, e, f, _) = apply.apply_views7::<
                                            A, B, C, D, E, F, u32,
                                            crate::detail::api::Tuple6AsTuple7BinaryOp<Op>,
                                        >(
                                            policy,
                                            &self.0,
                                            &self.1,
                                            &self.2,
                                            &self.3,
                                            &self.4,
                                            &self.5,
                                            &dummy6,
                                            (init.0, init.1, init.2, init.3, init.4, init.5, 0),
                                        )?;
                Ok((a, b, c, d, e, f))
            }
        }
    };
}

macro_rules! impl_kernel_reduce_by_key_tuple7_views {
    () => {
        impl<R, A, B, C, D, E, F, G, KeyEq, Op>
            KernelReduceByKeyValues<ReduceByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
                DeviceColumnView<R, E>,
                DeviceColumnView<R, F>,
                DeviceColumnView<R, G>,
            )
        where
            R: Runtime,
            A: Scalar + 'static,
            B: Scalar + 'static,
            C: Scalar + 'static,
            D: Scalar + 'static,
            E: Scalar + 'static,
            F: Scalar + 'static,
            G: Scalar + 'static,
            Op: BinaryOp<(A, B, C, D, E, F, G)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D, E, F, G);
            type OutputValues = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
                DeviceVec<R, F>,
                DeviceVec<R, G>,
            );

            fn reduce_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ReduceByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::OutputValues, Error> {
                if control.len == 0 {
                    return Ok((
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                    ));
                }
                let apply = SegmentedReduceApply::new(control);
                apply.apply_views7::<A, B, C, D, E, F, G, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, &self.4, &self.5, &self.6, init,
                )
            }
        }
    };
}

impl_kernel_reduce_by_key_tuple4_views!();
impl_kernel_reduce_by_key_tuple5_views!();
impl_kernel_reduce_by_key_tuple6_views!();
impl_kernel_reduce_by_key_tuple7_views!();
