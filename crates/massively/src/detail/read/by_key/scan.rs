use super::super::*;

pub(crate) struct ScanByKeyControl<R: Runtime> {
    pub(crate) head_flags: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

pub(crate) trait KernelScanByKeyKeys<KeyEq>: Sized {
    type Runtime: Runtime;
    type Control;

    fn scan_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Control, Error>;
}

pub(crate) trait KernelInclusiveScanByKeyValues<Control, KeyEq, Op>: Sized
where
    Control: Sized,
{
    type Runtime: Runtime;
    type Output;

    fn inclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &Control,
    ) -> Result<Self::Output, Error>;
}

pub(crate) trait KernelExclusiveScanByKeyValues<Control, KeyEq, Op>: Sized
where
    Control: Sized,
{
    type Runtime: Runtime;
    type Init;
    type Output;

    fn exclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &Control,
        init: Self::Init,
    ) -> Result<Self::Output, Error>;
}

pub(crate) trait KernelInclusiveScanByKeyCall<Values, KeyEq, Op>: Sized {
    type Runtime: Runtime;
    type Output;

    fn inclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelExclusiveScanByKeyCall<Values, KeyEq, Op>: Sized {
    type Runtime: Runtime;
    type Init;
    type Output;

    fn exclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        init: Self::Init,
        key_eq: GpuOp<KeyEq>,
        op: GpuOp<Op>,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(super) fn scan_by_key_head_flags_read<KeySource, KeyEq>(
    policy: &CubePolicy<KeySource::Runtime>,
    keys: &KeySource,
) -> Result<cubecl::server::Handle, Error>
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
{
    <KeySource as KernelColumn>::validate(keys)?;
    let len = <KeySource as KernelColumn>::len(keys);
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let client = policy.client();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let flags = client.empty(len * std::mem::size_of::<u32>());
    let key_bindings = <KeySource as KernelColumn>::stage(keys, policy)?;
    let key_slot0 = key_bindings.slots.first().unwrap();
    let key_slot1 = key_bindings.slots.get(1).unwrap_or(key_slot0);
    let key_slot2 = key_bindings.slots.get(2).unwrap_or(key_slot0);
    let key_slot3 = key_bindings.slots.get(3).unwrap_or(key_slot0);
    let key_offsets = key_bindings.slot_offsets_handle(client)?;
    let num_blocks = len.div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        scan_by_key_head_flags_device_expr_kernel::launch_unchecked::<
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
            BufferArg::from_raw_parts(flags.clone(), len),
        );
    }

    Ok(flags)
}

pub(super) fn inclusive_scan_by_flags_one<Source, Op>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
    control: &ScanByKeyControl<Source::Runtime>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    <Source as KernelColumn>::validate(source)?;
    ensure_same_len(<Source as KernelColumn>::len(source), control.len)?;
    if control.len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let bindings = <Source as KernelColumn>::stage(source, policy)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let offsets = bindings.slot_offsets_handle(client)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let output = client.empty(control.len * std::mem::size_of::<Source::Item>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        inclusive_scan_by_flags_device_expr_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Op,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), control.len),
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output, control.len))
}

pub(super) fn exclusive_scan_by_flags_one<Source, Op>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
    control: &ScanByKeyControl<Source::Runtime>,
    init: Source::Item,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    <Source as KernelColumn>::validate(source)?;
    ensure_same_len(<Source as KernelColumn>::len(source), control.len)?;
    if control.len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let bindings = <Source as KernelColumn>::stage(source, policy)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let offsets = bindings.slot_offsets_handle(client)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let init_handle = client.create_from_slice(Source::Item::as_bytes(&[init]));
    let output = client.empty(control.len * std::mem::size_of::<Source::Item>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        exclusive_scan_by_flags_device_expr_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Op,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(init_handle.clone(), 1),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), control.len),
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output, control.len))
}

pub(super) fn inclusive_scan_by_flags_two<A, C, Op>(
    policy: &CubePolicy<A::Runtime>,
    left: &A,
    right: &C,
    control: &ScanByKeyControl<A::Runtime>,
) -> Result<DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>, Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: Scalar + 'static,
    C::Item: Scalar + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    Op: BinaryOp<(A::Item, C::Item)>,
{
    validate_columns2(left, right)?;
    ensure_same_len(<A as KernelColumn>::len(left), control.len)?;
    if control.len == 0 {
        return Ok(DeviceSoA2 {
            left: policy.empty_device_vec(),
            right: policy.empty_device_vec(),
        });
    }

    let client = policy.client();
    let a = <A as KernelColumn>::stage(left, policy)?;
    let b = <C as KernelColumn>::stage(right, policy)?;
    let a0 = a.slots.first().unwrap();
    let a1 = a.slots.get(1).unwrap_or(a0);
    let a2 = a.slots.get(2).unwrap_or(a0);
    let a3 = a.slots.get(3).unwrap_or(a0);
    let b0 = b.slots.first().unwrap();
    let b1 = b.slots.get(1).unwrap_or(b0);
    let b2 = b.slots.get(2).unwrap_or(b0);
    let b3 = b.slots.get(3).unwrap_or(b0);
    let a_offsets = a.slot_offsets_handle(client)?;
    let b_offsets = b.slot_offsets_handle(client)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let out_a = client.empty(control.len * std::mem::size_of::<A::Item>());
    let out_b = client.empty(control.len * std::mem::size_of::<C::Item>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        inclusive_scan_tuple2_by_flags_device_expr_kernel::launch_unchecked::<
            A::Item,
            C::Item,
            A::Expr,
            C::Expr,
            Op,
            A::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a0.0.clone(), a0.1),
            BufferArg::from_raw_parts(a1.0.clone(), a1.1),
            BufferArg::from_raw_parts(a2.0.clone(), a2.1),
            BufferArg::from_raw_parts(a3.0.clone(), a3.1),
            BufferArg::from_raw_parts(a_offsets.clone(), 4),
            BufferArg::from_raw_parts(b0.0.clone(), b0.1),
            BufferArg::from_raw_parts(b1.0.clone(), b1.1),
            BufferArg::from_raw_parts(b2.0.clone(), b2.1),
            BufferArg::from_raw_parts(b3.0.clone(), b3.1),
            BufferArg::from_raw_parts(b_offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
        );
    }

    Ok(DeviceSoA2 {
        left: DeviceVec::from_handle(policy.id(), out_a, control.len),
        right: DeviceVec::from_handle(policy.id(), out_b, control.len),
    })
}

pub(super) fn exclusive_scan_by_flags_two<A, C, Op>(
    policy: &CubePolicy<A::Runtime>,
    left: &A,
    right: &C,
    control: &ScanByKeyControl<A::Runtime>,
    init: (A::Item, C::Item),
) -> Result<DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>, Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: Scalar + 'static,
    C::Item: Scalar + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    Op: BinaryOp<(A::Item, C::Item)>,
{
    validate_columns2(left, right)?;
    ensure_same_len(<A as KernelColumn>::len(left), control.len)?;
    if control.len == 0 {
        return Ok(DeviceSoA2 {
            left: policy.empty_device_vec(),
            right: policy.empty_device_vec(),
        });
    }

    let client = policy.client();
    let a = <A as KernelColumn>::stage(left, policy)?;
    let b = <C as KernelColumn>::stage(right, policy)?;
    let a0 = a.slots.first().unwrap();
    let a1 = a.slots.get(1).unwrap_or(a0);
    let a2 = a.slots.get(2).unwrap_or(a0);
    let a3 = a.slots.get(3).unwrap_or(a0);
    let b0 = b.slots.first().unwrap();
    let b1 = b.slots.get(1).unwrap_or(b0);
    let b2 = b.slots.get(2).unwrap_or(b0);
    let b3 = b.slots.get(3).unwrap_or(b0);
    let a_offsets = a.slot_offsets_handle(client)?;
    let b_offsets = b.slot_offsets_handle(client)?;
    let init_a = client.create_from_slice(A::Item::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(C::Item::as_bytes(&[init.1]));
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let out_a = client.empty(control.len * std::mem::size_of::<A::Item>());
    let out_b = client.empty(control.len * std::mem::size_of::<C::Item>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        exclusive_scan_tuple2_by_flags_device_expr_kernel::launch_unchecked::<
            A::Item,
            C::Item,
            A::Expr,
            C::Expr,
            Op,
            A::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a0.0.clone(), a0.1),
            BufferArg::from_raw_parts(a1.0.clone(), a1.1),
            BufferArg::from_raw_parts(a2.0.clone(), a2.1),
            BufferArg::from_raw_parts(a3.0.clone(), a3.1),
            BufferArg::from_raw_parts(a_offsets.clone(), 4),
            BufferArg::from_raw_parts(b0.0.clone(), b0.1),
            BufferArg::from_raw_parts(b1.0.clone(), b1.1),
            BufferArg::from_raw_parts(b2.0.clone(), b2.1),
            BufferArg::from_raw_parts(b3.0.clone(), b3.1),
            BufferArg::from_raw_parts(b_offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(init_a.clone(), 1),
            BufferArg::from_raw_parts(init_b.clone(), 1),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
        );
    }

    Ok(DeviceSoA2 {
        left: DeviceVec::from_handle(policy.id(), out_a, control.len),
        right: DeviceVec::from_handle(policy.id(), out_b, control.len),
    })
}

pub(super) fn inclusive_scan_by_flags_three<A, C, D, Op>(
    policy: &CubePolicy<A::Runtime>,
    first: &A,
    second: &C,
    third: &D,
    control: &ScanByKeyControl<A::Runtime>,
) -> Result<
    DeviceSoA3<
        DeviceVec<A::Runtime, A::Item>,
        DeviceVec<A::Runtime, C::Item>,
        DeviceVec<A::Runtime, D::Item>,
    >,
    Error,
>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: Scalar + 'static,
    C::Item: Scalar + 'static,
    D::Item: Scalar + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: BinaryOp<(A::Item, C::Item, D::Item)>,
{
    validate_columns3(first, second, third)?;
    ensure_same_len(<A as KernelColumn>::len(first), control.len)?;
    if control.len == 0 {
        return Ok(DeviceSoA3 {
            first: policy.empty_device_vec(),
            second: policy.empty_device_vec(),
            third: policy.empty_device_vec(),
        });
    }

    let client = policy.client();
    let a = <A as KernelColumn>::stage(first, policy)?;
    let b = <C as KernelColumn>::stage(second, policy)?;
    let c = <D as KernelColumn>::stage(third, policy)?;
    let a0 = a.slots.first().unwrap();
    let a1 = a.slots.get(1).unwrap_or(a0);
    let a2 = a.slots.get(2).unwrap_or(a0);
    let a3 = a.slots.get(3).unwrap_or(a0);
    let b0 = b.slots.first().unwrap();
    let b1 = b.slots.get(1).unwrap_or(b0);
    let b2 = b.slots.get(2).unwrap_or(b0);
    let b3 = b.slots.get(3).unwrap_or(b0);
    let c0 = c.slots.first().unwrap();
    let c1 = c.slots.get(1).unwrap_or(c0);
    let c2 = c.slots.get(2).unwrap_or(c0);
    let c3 = c.slots.get(3).unwrap_or(c0);
    let a_offsets = a.slot_offsets_handle(client)?;
    let b_offsets = b.slot_offsets_handle(client)?;
    let c_offsets = c.slot_offsets_handle(client)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let out_a = client.empty(control.len * std::mem::size_of::<A::Item>());
    let out_b = client.empty(control.len * std::mem::size_of::<C::Item>());
    let out_c = client.empty(control.len * std::mem::size_of::<D::Item>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        inclusive_scan_tuple3_by_flags_device_expr_kernel::launch_unchecked::<
            A::Item,
            C::Item,
            D::Item,
            A::Expr,
            C::Expr,
            D::Expr,
            Op,
            A::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a0.0.clone(), a0.1),
            BufferArg::from_raw_parts(a1.0.clone(), a1.1),
            BufferArg::from_raw_parts(a2.0.clone(), a2.1),
            BufferArg::from_raw_parts(a3.0.clone(), a3.1),
            BufferArg::from_raw_parts(a_offsets.clone(), 4),
            BufferArg::from_raw_parts(b0.0.clone(), b0.1),
            BufferArg::from_raw_parts(b1.0.clone(), b1.1),
            BufferArg::from_raw_parts(b2.0.clone(), b2.1),
            BufferArg::from_raw_parts(b3.0.clone(), b3.1),
            BufferArg::from_raw_parts(b_offsets.clone(), 4),
            BufferArg::from_raw_parts(c0.0.clone(), c0.1),
            BufferArg::from_raw_parts(c1.0.clone(), c1.1),
            BufferArg::from_raw_parts(c2.0.clone(), c2.1),
            BufferArg::from_raw_parts(c3.0.clone(), c3.1),
            BufferArg::from_raw_parts(c_offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
            BufferArg::from_raw_parts(out_c.clone(), control.len),
        );
    }

    Ok(DeviceSoA3 {
        first: DeviceVec::from_handle(policy.id(), out_a, control.len),
        second: DeviceVec::from_handle(policy.id(), out_b, control.len),
        third: DeviceVec::from_handle(policy.id(), out_c, control.len),
    })
}

pub(super) fn exclusive_scan_by_flags_three<A, C, D, Op>(
    policy: &CubePolicy<A::Runtime>,
    first: &A,
    second: &C,
    third: &D,
    control: &ScanByKeyControl<A::Runtime>,
    init: (A::Item, C::Item, D::Item),
) -> Result<
    DeviceSoA3<
        DeviceVec<A::Runtime, A::Item>,
        DeviceVec<A::Runtime, C::Item>,
        DeviceVec<A::Runtime, D::Item>,
    >,
    Error,
>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: Scalar + 'static,
    C::Item: Scalar + 'static,
    D::Item: Scalar + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: BinaryOp<(A::Item, C::Item, D::Item)>,
{
    validate_columns3(first, second, third)?;
    ensure_same_len(<A as KernelColumn>::len(first), control.len)?;
    if control.len == 0 {
        return Ok(DeviceSoA3 {
            first: policy.empty_device_vec(),
            second: policy.empty_device_vec(),
            third: policy.empty_device_vec(),
        });
    }

    let client = policy.client();
    let a = <A as KernelColumn>::stage(first, policy)?;
    let b = <C as KernelColumn>::stage(second, policy)?;
    let c = <D as KernelColumn>::stage(third, policy)?;
    let a0 = a.slots.first().unwrap();
    let a1 = a.slots.get(1).unwrap_or(a0);
    let a2 = a.slots.get(2).unwrap_or(a0);
    let a3 = a.slots.get(3).unwrap_or(a0);
    let b0 = b.slots.first().unwrap();
    let b1 = b.slots.get(1).unwrap_or(b0);
    let b2 = b.slots.get(2).unwrap_or(b0);
    let b3 = b.slots.get(3).unwrap_or(b0);
    let c0 = c.slots.first().unwrap();
    let c1 = c.slots.get(1).unwrap_or(c0);
    let c2 = c.slots.get(2).unwrap_or(c0);
    let c3 = c.slots.get(3).unwrap_or(c0);
    let a_offsets = a.slot_offsets_handle(client)?;
    let b_offsets = b.slot_offsets_handle(client)?;
    let c_offsets = c.slot_offsets_handle(client)?;
    let init_a = client.create_from_slice(A::Item::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(C::Item::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(D::Item::as_bytes(&[init.2]));
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let out_a = client.empty(control.len * std::mem::size_of::<A::Item>());
    let out_b = client.empty(control.len * std::mem::size_of::<C::Item>());
    let out_c = client.empty(control.len * std::mem::size_of::<D::Item>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        exclusive_scan_tuple3_by_flags_device_expr_kernel::launch_unchecked::<
            A::Item,
            C::Item,
            D::Item,
            A::Expr,
            C::Expr,
            D::Expr,
            Op,
            A::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a0.0.clone(), a0.1),
            BufferArg::from_raw_parts(a1.0.clone(), a1.1),
            BufferArg::from_raw_parts(a2.0.clone(), a2.1),
            BufferArg::from_raw_parts(a3.0.clone(), a3.1),
            BufferArg::from_raw_parts(a_offsets.clone(), 4),
            BufferArg::from_raw_parts(b0.0.clone(), b0.1),
            BufferArg::from_raw_parts(b1.0.clone(), b1.1),
            BufferArg::from_raw_parts(b2.0.clone(), b2.1),
            BufferArg::from_raw_parts(b3.0.clone(), b3.1),
            BufferArg::from_raw_parts(b_offsets.clone(), 4),
            BufferArg::from_raw_parts(c0.0.clone(), c0.1),
            BufferArg::from_raw_parts(c1.0.clone(), c1.1),
            BufferArg::from_raw_parts(c2.0.clone(), c2.1),
            BufferArg::from_raw_parts(c3.0.clone(), c3.1),
            BufferArg::from_raw_parts(c_offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(init_a.clone(), 1),
            BufferArg::from_raw_parts(init_b.clone(), 1),
            BufferArg::from_raw_parts(init_c.clone(), 1),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
            BufferArg::from_raw_parts(out_c.clone(), control.len),
        );
    }

    Ok(DeviceSoA3 {
        first: DeviceVec::from_handle(policy.id(), out_a, control.len),
        second: DeviceVec::from_handle(policy.id(), out_b, control.len),
        third: DeviceVec::from_handle(policy.id(), out_c, control.len),
    })
}

impl<KeySource, KeyEq> KernelScanByKeyKeys<KeyEq> for KeySource
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type Control = ScanByKeyControl<KeySource::Runtime>;

    fn scan_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Control, Error> {
        let len = <KeySource as KernelColumn>::len(&self);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let head_flags = scan_by_key_head_flags_read::<KeySource, KeyEq>(policy, &self)?;
        Ok(ScanByKeyControl {
            head_flags,
            len,
            len_u32,
            _runtime: std::marker::PhantomData,
        })
    }
}

macro_rules! impl_kernel_scan_by_key_keys_tuple1 {
    ($target:ty, $field:tt) => {
        impl<KeySource, KeyEq> KernelScanByKeyKeys<KeyEq> for $target
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            KeySource::Item: Scalar + 'static,
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            KeyEq: BinaryPredicateOp<KeySource::Item>,
        {
            type Runtime = KeySource::Runtime;
            type Control = ScanByKeyControl<KeySource::Runtime>;

            fn scan_by_key_control(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Control, Error> {
                <KeySource as KernelScanByKeyKeys<KeyEq>>::scan_by_key_control(self.$field, policy)
            }
        }
    };
}

impl_kernel_scan_by_key_keys_tuple1!(SoAView1<KeySource>, source);
impl_kernel_scan_by_key_keys_tuple1!(DeviceSoA1<KeySource>, source);

impl<KeySource, KeyEq> KernelScanByKeyKeys<KeyEq> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    KeyEq: BinaryPredicateOp<(KeySource::Item,)>,
    crate::detail::api::Tuple1Less<KeyEq>: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type Control = ScanByKeyControl<KeySource::Runtime>;

    fn scan_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Control, Error> {
        <KeySource as KernelScanByKeyKeys<crate::detail::api::Tuple1Less<KeyEq>>>::scan_by_key_control(
            self.0,
            policy,
        )
    }
}

macro_rules! impl_kernel_scan_by_key_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S, KeyEq, Op> KernelInclusiveScanByKeyValues<ScanByKeyControl<S::Runtime>, KeyEq, Op>
            for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: Scalar + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<S::Runtime>,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceSoA1 {
                    source: inclusive_scan_by_flags_one::<S, Op>(policy, &self.$field, control)?,
                })
            }
        }

        impl<S, KeyEq, Op> KernelExclusiveScanByKeyValues<ScanByKeyControl<S::Runtime>, KeyEq, Op>
            for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: Scalar + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Init = S::Item;
            type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<S::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceSoA1 {
                    source: exclusive_scan_by_flags_one::<S, Op>(
                        policy,
                        &self.$field,
                        control,
                        init,
                    )?,
                })
            }
        }
    };
}

impl_kernel_scan_by_key_tuple1!((S,), 0);
impl_kernel_scan_by_key_tuple1!(SoAView1<S>, source);
impl_kernel_scan_by_key_tuple1!(DeviceSoA1<S>, source);

macro_rules! impl_kernel_scan_by_key_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<A, C, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            (A::Item, C::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item)>,
        {
            type Runtime = A::Runtime;
            type Output =
                DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<A::Runtime>,
            ) -> Result<Self::Output, Error> {
                inclusive_scan_by_flags_two::<A, C, Op>(policy, &self.$left, &self.$right, control)
            }
        }

        impl<A, C, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            (A::Item, C::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item)>,
        {
            type Runtime = A::Runtime;
            type Init = (A::Item, C::Item);
            type Output =
                DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<A::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                exclusive_scan_by_flags_two::<A, C, Op>(
                    policy,
                    &self.$left,
                    &self.$right,
                    control,
                    init,
                )
            }
        }
    };
}

impl_kernel_scan_by_key_tuple2!(SoAView2<A, C>, left, right);
impl_kernel_scan_by_key_tuple2!(DeviceSoA2<A, C>, left, right);

macro_rules! impl_kernel_scan_by_key_tuple3 {
    ($target:ty, $first:tt, $second:tt, $third:tt) => {
        impl<A, C, D, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            D::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            D::Expr: DeviceGpuExpr<D::Item>,
            (A::Item, C::Item, D::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item, D::Item)>,
        {
            type Runtime = A::Runtime;
            type Output = DeviceSoA3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<A::Runtime>,
            ) -> Result<Self::Output, Error> {
                inclusive_scan_by_flags_three::<A, C, D, Op>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    control,
                )
            }
        }

        impl<A, C, D, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            D::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            D::Expr: DeviceGpuExpr<D::Item>,
            (A::Item, C::Item, D::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item, D::Item)>,
        {
            type Runtime = A::Runtime;
            type Init = (A::Item, C::Item, D::Item);
            type Output = DeviceSoA3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<A::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                exclusive_scan_by_flags_three::<A, C, D, Op>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    control,
                    init,
                )
            }
        }
    };
}

impl_kernel_scan_by_key_tuple3!(SoAView3<A, C, D>, first, second, third);
impl_kernel_scan_by_key_tuple3!(DeviceSoA3<A, C, D>, first, second, third);

impl<Left, Right, R, KeyEq, Op> KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
    for (Left, Right)
where
    R: Runtime,
    SoAView2<Left, Right>: KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <SoAView2<Left, Right> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Output;

    fn inclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ScanByKeyControl<R>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelInclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::inclusive_scan_by_key_values(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            control,
        )
    }
}

impl<First, Second, Third, R, KeyEq, Op>
    KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op> for (First, Second, Third)
where
    R: Runtime,
    SoAView3<First, Second, Third>: KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Output;

    fn inclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ScanByKeyControl<R>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelInclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::inclusive_scan_by_key_values(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            control,
        )
    }
}

impl<Left, Right, R, KeyEq, Op> KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
    for (Left, Right)
where
    R: Runtime,
    SoAView2<Left, Right>: KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <SoAView2<Left, Right> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Init = <SoAView2<Left, Right> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Init;
    type Output = <SoAView2<Left, Right> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Output;

    fn exclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ScanByKeyControl<R>,
        init: Self::Init,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelExclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::exclusive_scan_by_key_values(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            control,
            init,
        )
    }
}

impl<First, Second, Third, R, KeyEq, Op>
    KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op> for (First, Second, Third)
where
    R: Runtime,
    SoAView3<First, Second, Third>: KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Init = <SoAView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Init;
    type Output = <SoAView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Output;

    fn exclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ScanByKeyControl<R>,
        init: Self::Init,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::exclusive_scan_by_key_values(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            control,
            init,
        )
    }
}
