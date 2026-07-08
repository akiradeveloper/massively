use super::super::*;
use crate::detail::{
    control::{ScanByKeyControl, SegmentControl},
    device::{DeviceColumnMutView, DeviceColumnView},
};

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
    KeySource::Item: MStorageElement + 'static,
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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

#[allow(dead_code)]
pub(crate) fn inclusive_scan_by_flags_one<Source, Op>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
    control: &ScanByKeyControl<Source::Runtime>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: MStorageElement + 'static,
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
    let output_offset = client.create_from_slice(u32::as_bytes(&[0]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offset.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), control.len),
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output, control.len))
}

#[allow(dead_code)]
pub(crate) fn inclusive_scan_by_flags_one_into<Source, Op>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
    control: &ScanByKeyControl<Source::Runtime>,
    output: &DeviceColumnMutView<Source::Runtime, Source::Item>,
) -> Result<(), Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: MStorageElement + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    <Source as KernelColumn>::validate(source)?;
    ensure_same_len(<Source as KernelColumn>::len(source), control.len)?;
    ensure_same_len(output.len, control.len)?;
    if control.len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let bindings = <Source as KernelColumn>::stage(source, policy)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let offsets = bindings.slot_offsets_handle(client)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let output_offset_u32 =
        u32::try_from(output.offset).map_err(|_| Error::LengthTooLarge { len: output.offset })?;
    let output_offset = client.create_from_slice(u32::as_bytes(&[output_offset_u32]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offset.clone(), 1),
            BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()),
        );
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn exclusive_scan_by_flags_one<Source, Op>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
    control: &ScanByKeyControl<Source::Runtime>,
    init: Source::Item,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: MStorageElement + 'static,
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
    let output_offset = client.create_from_slice(u32::as_bytes(&[0]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(init_handle.clone(), 1),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offset.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), control.len),
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output, control.len))
}

#[allow(dead_code)]
pub(crate) fn exclusive_scan_by_flags_one_into<Source, Op>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
    control: &ScanByKeyControl<Source::Runtime>,
    init: Source::Item,
    output: &DeviceColumnMutView<Source::Runtime, Source::Item>,
) -> Result<(), Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: MStorageElement + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: BinaryOp<(Source::Item,)>,
{
    <Source as KernelColumn>::validate(source)?;
    ensure_same_len(<Source as KernelColumn>::len(source), control.len)?;
    ensure_same_len(output.len, control.len)?;
    if control.len == 0 {
        return Ok(());
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
    let output_offset_u32 =
        u32::try_from(output.offset).map_err(|_| Error::LengthTooLarge { len: output.offset })?;
    let output_offset = client.create_from_slice(u32::as_bytes(&[output_offset_u32]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(offsets.clone(), 4),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(init_handle.clone(), 1),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offset.clone(), 1),
            BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()),
        );
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn inclusive_scan_by_flags_two<A, C, Op>(
    policy: &CubePolicy<A::Runtime>,
    left: &A,
    right: &C,
    control: &ScanByKeyControl<A::Runtime>,
) -> Result<DeviceZip2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>, Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    Op: BinaryOp<(A::Item, C::Item)>,
{
    validate_columns2(left, right)?;
    ensure_same_len(<A as KernelColumn>::len(left), control.len)?;
    if control.len == 0 {
        return Ok(DeviceZip2 {
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
    let output_offsets = client.create_from_slice(u32::as_bytes(&[0, 0]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 2),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
        );
    }

    Ok(DeviceZip2 {
        left: DeviceVec::from_handle(policy.id(), out_a, control.len),
        right: DeviceVec::from_handle(policy.id(), out_b, control.len),
    })
}

#[allow(dead_code)]
pub(crate) fn inclusive_scan_by_flags_two_into<A, C, Op>(
    policy: &CubePolicy<A::Runtime>,
    left: &A,
    right: &C,
    control: &ScanByKeyControl<A::Runtime>,
    out_left: &DeviceColumnMutView<A::Runtime, A::Item>,
    out_right: &DeviceColumnMutView<A::Runtime, C::Item>,
) -> Result<(), Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    Op: BinaryOp<(A::Item, C::Item)>,
{
    validate_columns2(left, right)?;
    ensure_same_len(<A as KernelColumn>::len(left), control.len)?;
    ensure_same_len(out_left.len, control.len)?;
    ensure_same_len(out_right.len, control.len)?;
    if control.len == 0 {
        return Ok(());
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
    let output_offsets = [
        u32::try_from(out_left.offset).map_err(|_| Error::LengthTooLarge {
            len: out_left.offset,
        })?,
        u32::try_from(out_right.offset).map_err(|_| Error::LengthTooLarge {
            len: out_right.offset,
        })?,
    ];
    let output_offsets = client.create_from_slice(u32::as_bytes(&output_offsets));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 2),
            BufferArg::from_raw_parts(out_left.source.handle.clone(), out_left.source.len()),
            BufferArg::from_raw_parts(out_right.source.handle.clone(), out_right.source.len()),
        );
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn exclusive_scan_by_flags_two<A, C, Op>(
    policy: &CubePolicy<A::Runtime>,
    left: &A,
    right: &C,
    control: &ScanByKeyControl<A::Runtime>,
    init: (A::Item, C::Item),
) -> Result<DeviceZip2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>, Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    Op: BinaryOp<(A::Item, C::Item)>,
{
    validate_columns2(left, right)?;
    ensure_same_len(<A as KernelColumn>::len(left), control.len)?;
    if control.len == 0 {
        return Ok(DeviceZip2 {
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
    let output_offsets = client.create_from_slice(u32::as_bytes(&[0, 0]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 2),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
        );
    }

    Ok(DeviceZip2 {
        left: DeviceVec::from_handle(policy.id(), out_a, control.len),
        right: DeviceVec::from_handle(policy.id(), out_b, control.len),
    })
}

#[allow(dead_code)]
pub(crate) fn exclusive_scan_by_flags_two_into<A, C, Op>(
    policy: &CubePolicy<A::Runtime>,
    left: &A,
    right: &C,
    control: &ScanByKeyControl<A::Runtime>,
    init: (A::Item, C::Item),
    out_left: &DeviceColumnMutView<A::Runtime, A::Item>,
    out_right: &DeviceColumnMutView<A::Runtime, C::Item>,
) -> Result<(), Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    Op: BinaryOp<(A::Item, C::Item)>,
{
    validate_columns2(left, right)?;
    ensure_same_len(<A as KernelColumn>::len(left), control.len)?;
    ensure_same_len(out_left.len, control.len)?;
    ensure_same_len(out_right.len, control.len)?;
    if control.len == 0 {
        return Ok(());
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
    let output_offsets = [
        u32::try_from(out_left.offset).map_err(|_| Error::LengthTooLarge {
            len: out_left.offset,
        })?,
        u32::try_from(out_right.offset).map_err(|_| Error::LengthTooLarge {
            len: out_right.offset,
        })?,
    ];
    let output_offsets = client.create_from_slice(u32::as_bytes(&output_offsets));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 2),
            BufferArg::from_raw_parts(out_left.source.handle.clone(), out_left.source.len()),
            BufferArg::from_raw_parts(out_right.source.handle.clone(), out_right.source.len()),
        );
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn inclusive_scan_by_flags_three<A, C, D, Op>(
    policy: &CubePolicy<A::Runtime>,
    first: &A,
    second: &C,
    third: &D,
    control: &ScanByKeyControl<A::Runtime>,
) -> Result<
    DeviceZip3<
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
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    D::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: BinaryOp<(A::Item, C::Item, D::Item)>,
{
    validate_columns3(first, second, third)?;
    ensure_same_len(<A as KernelColumn>::len(first), control.len)?;
    if control.len == 0 {
        return Ok(DeviceZip3 {
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
    let output_offsets = client.create_from_slice(u32::as_bytes(&[0, 0, 0]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 3),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
            BufferArg::from_raw_parts(out_c.clone(), control.len),
        );
    }

    Ok(DeviceZip3 {
        first: DeviceVec::from_handle(policy.id(), out_a, control.len),
        second: DeviceVec::from_handle(policy.id(), out_b, control.len),
        third: DeviceVec::from_handle(policy.id(), out_c, control.len),
    })
}

#[allow(dead_code)]
pub(crate) fn inclusive_scan_by_flags_three_into<A, C, D, Op>(
    policy: &CubePolicy<A::Runtime>,
    first: &A,
    second: &C,
    third: &D,
    control: &ScanByKeyControl<A::Runtime>,
    out_first: &DeviceColumnMutView<A::Runtime, A::Item>,
    out_second: &DeviceColumnMutView<A::Runtime, C::Item>,
    out_third: &DeviceColumnMutView<A::Runtime, D::Item>,
) -> Result<(), Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    D::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: BinaryOp<(A::Item, C::Item, D::Item)>,
{
    validate_columns3(first, second, third)?;
    ensure_same_len(<A as KernelColumn>::len(first), control.len)?;
    ensure_same_len(out_first.len, control.len)?;
    ensure_same_len(out_second.len, control.len)?;
    ensure_same_len(out_third.len, control.len)?;
    if control.len == 0 {
        return Ok(());
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
    let output_offsets = [
        u32::try_from(out_first.offset).map_err(|_| Error::LengthTooLarge {
            len: out_first.offset,
        })?,
        u32::try_from(out_second.offset).map_err(|_| Error::LengthTooLarge {
            len: out_second.offset,
        })?,
        u32::try_from(out_third.offset).map_err(|_| Error::LengthTooLarge {
            len: out_third.offset,
        })?,
    ];
    let output_offsets = client.create_from_slice(u32::as_bytes(&output_offsets));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 3),
            BufferArg::from_raw_parts(out_first.source.handle.clone(), out_first.source.len()),
            BufferArg::from_raw_parts(out_second.source.handle.clone(), out_second.source.len()),
            BufferArg::from_raw_parts(out_third.source.handle.clone(), out_third.source.len()),
        );
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn exclusive_scan_by_flags_three<A, C, D, Op>(
    policy: &CubePolicy<A::Runtime>,
    first: &A,
    second: &C,
    third: &D,
    control: &ScanByKeyControl<A::Runtime>,
    init: (A::Item, C::Item, D::Item),
) -> Result<
    DeviceZip3<
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
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    D::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: BinaryOp<(A::Item, C::Item, D::Item)>,
{
    validate_columns3(first, second, third)?;
    ensure_same_len(<A as KernelColumn>::len(first), control.len)?;
    if control.len == 0 {
        return Ok(DeviceZip3 {
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
    let output_offsets = client.create_from_slice(u32::as_bytes(&[0, 0, 0]));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 3),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
            BufferArg::from_raw_parts(out_c.clone(), control.len),
        );
    }

    Ok(DeviceZip3 {
        first: DeviceVec::from_handle(policy.id(), out_a, control.len),
        second: DeviceVec::from_handle(policy.id(), out_b, control.len),
        third: DeviceVec::from_handle(policy.id(), out_c, control.len),
    })
}

#[allow(dead_code)]
pub(crate) fn exclusive_scan_by_flags_three_into<A, C, D, Op>(
    policy: &CubePolicy<A::Runtime>,
    first: &A,
    second: &C,
    third: &D,
    control: &ScanByKeyControl<A::Runtime>,
    init: (A::Item, C::Item, D::Item),
    out_first: &DeviceColumnMutView<A::Runtime, A::Item>,
    out_second: &DeviceColumnMutView<A::Runtime, C::Item>,
    out_third: &DeviceColumnMutView<A::Runtime, D::Item>,
) -> Result<(), Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: MStorageElement + 'static,
    C::Item: MStorageElement + 'static,
    D::Item: MStorageElement + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: BinaryOp<(A::Item, C::Item, D::Item)>,
{
    validate_columns3(first, second, third)?;
    ensure_same_len(<A as KernelColumn>::len(first), control.len)?;
    ensure_same_len(out_first.len, control.len)?;
    ensure_same_len(out_second.len, control.len)?;
    ensure_same_len(out_third.len, control.len)?;
    if control.len == 0 {
        return Ok(());
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
    let output_offsets = [
        u32::try_from(out_first.offset).map_err(|_| Error::LengthTooLarge {
            len: out_first.offset,
        })?,
        u32::try_from(out_second.offset).map_err(|_| Error::LengthTooLarge {
            len: out_second.offset,
        })?,
        u32::try_from(out_third.offset).map_err(|_| Error::LengthTooLarge {
            len: out_third.offset,
        })?,
    ];
    let output_offsets = client.create_from_slice(u32::as_bytes(&output_offsets));
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
            crate::detail::launch::cube_count_1d(num_blocks_u32),
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
            BufferArg::from_raw_parts(output_offsets.clone(), 3),
            BufferArg::from_raw_parts(out_first.source.handle.clone(), out_first.source.len()),
            BufferArg::from_raw_parts(out_second.source.handle.clone(), out_second.source.len()),
            BufferArg::from_raw_parts(out_third.source.handle.clone(), out_third.source.len()),
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn inclusive_scan_by_flags_seven_views<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    a: &DeviceColumnView<R, A>,
    b: &DeviceColumnView<R, B>,
    c: &DeviceColumnView<R, C>,
    d: &DeviceColumnView<R, D>,
    e: &DeviceColumnView<R, E>,
    f: &DeviceColumnView<R, F>,
    g: &DeviceColumnView<R, G>,
    control: &ScanByKeyControl<R>,
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
    R: Runtime,
    A: MStorageElement + 'static,
    B: MStorageElement + 'static,
    C: MStorageElement + 'static,
    D: MStorageElement + 'static,
    E: MStorageElement + 'static,
    F: MStorageElement + 'static,
    G: MStorageElement + 'static,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    ensure_same_len(a.len, control.len)?;
    ensure_same_len(b.len, control.len)?;
    ensure_same_len(c.len, control.len)?;
    ensure_same_len(d.len, control.len)?;
    ensure_same_len(e.len, control.len)?;
    ensure_same_len(f.len, control.len)?;
    ensure_same_len(g.len, control.len)?;
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

    let client = policy.client();
    let offsets = [
        u32::try_from(a.offset).map_err(|_| Error::LengthTooLarge { len: a.offset })?,
        u32::try_from(b.offset).map_err(|_| Error::LengthTooLarge { len: b.offset })?,
        u32::try_from(c.offset).map_err(|_| Error::LengthTooLarge { len: c.offset })?,
        u32::try_from(d.offset).map_err(|_| Error::LengthTooLarge { len: d.offset })?,
        u32::try_from(e.offset).map_err(|_| Error::LengthTooLarge { len: e.offset })?,
        u32::try_from(f.offset).map_err(|_| Error::LengthTooLarge { len: f.offset })?,
        u32::try_from(g.offset).map_err(|_| Error::LengthTooLarge { len: g.offset })?,
    ];
    let offsets_handle = client.create_from_slice(u32::as_bytes(&offsets));
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let output_offsets = client.create_from_slice(u32::as_bytes(&[0, 0, 0, 0, 0, 0, 0]));
    let out_a = client.empty(control.len * std::mem::size_of::<A>());
    let out_b = client.empty(control.len * std::mem::size_of::<B>());
    let out_c = client.empty(control.len * std::mem::size_of::<C>());
    let out_d = client.empty(control.len * std::mem::size_of::<D>());
    let out_e = client.empty(control.len * std::mem::size_of::<E>());
    let out_f = client.empty(control.len * std::mem::size_of::<F>());
    let out_g = client.empty(control.len * std::mem::size_of::<G>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        inclusive_scan_tuple7_by_flags_view_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a.source.handle.clone(), a.source.len()),
            BufferArg::from_raw_parts(b.source.handle.clone(), b.source.len()),
            BufferArg::from_raw_parts(c.source.handle.clone(), c.source.len()),
            BufferArg::from_raw_parts(d.source.handle.clone(), d.source.len()),
            BufferArg::from_raw_parts(e.source.handle.clone(), e.source.len()),
            BufferArg::from_raw_parts(f.source.handle.clone(), f.source.len()),
            BufferArg::from_raw_parts(g.source.handle.clone(), g.source.len()),
            BufferArg::from_raw_parts(offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offsets.clone(), 7),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
            BufferArg::from_raw_parts(out_c.clone(), control.len),
            BufferArg::from_raw_parts(out_d.clone(), control.len),
            BufferArg::from_raw_parts(out_e.clone(), control.len),
            BufferArg::from_raw_parts(out_f.clone(), control.len),
            BufferArg::from_raw_parts(out_g.clone(), control.len),
        );
    }

    Ok((
        DeviceVec::from_handle(policy.id(), out_a, control.len),
        DeviceVec::from_handle(policy.id(), out_b, control.len),
        DeviceVec::from_handle(policy.id(), out_c, control.len),
        DeviceVec::from_handle(policy.id(), out_d, control.len),
        DeviceVec::from_handle(policy.id(), out_e, control.len),
        DeviceVec::from_handle(policy.id(), out_f, control.len),
        DeviceVec::from_handle(policy.id(), out_g, control.len),
    ))
}

#[allow(dead_code, clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn inclusive_scan_by_flags_seven_views_into<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    a: &DeviceColumnView<R, A>,
    b: &DeviceColumnView<R, B>,
    c: &DeviceColumnView<R, C>,
    d: &DeviceColumnView<R, D>,
    e: &DeviceColumnView<R, E>,
    f: &DeviceColumnView<R, F>,
    g: &DeviceColumnView<R, G>,
    control: &ScanByKeyControl<R>,
    out_a: &DeviceColumnMutView<R, A>,
    out_b: &DeviceColumnMutView<R, B>,
    out_c: &DeviceColumnMutView<R, C>,
    out_d: &DeviceColumnMutView<R, D>,
    out_e: &DeviceColumnMutView<R, E>,
    out_f: &DeviceColumnMutView<R, F>,
    out_g: &DeviceColumnMutView<R, G>,
) -> Result<(), Error>
where
    R: Runtime,
    A: MStorageElement + 'static,
    B: MStorageElement + 'static,
    C: MStorageElement + 'static,
    D: MStorageElement + 'static,
    E: MStorageElement + 'static,
    F: MStorageElement + 'static,
    G: MStorageElement + 'static,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    ensure_same_len(a.len, control.len)?;
    ensure_same_len(b.len, control.len)?;
    ensure_same_len(c.len, control.len)?;
    ensure_same_len(d.len, control.len)?;
    ensure_same_len(e.len, control.len)?;
    ensure_same_len(f.len, control.len)?;
    ensure_same_len(g.len, control.len)?;
    ensure_same_len(out_a.len, control.len)?;
    ensure_same_len(out_b.len, control.len)?;
    ensure_same_len(out_c.len, control.len)?;
    ensure_same_len(out_d.len, control.len)?;
    ensure_same_len(out_e.len, control.len)?;
    ensure_same_len(out_f.len, control.len)?;
    ensure_same_len(out_g.len, control.len)?;
    if control.len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let offsets = [
        u32::try_from(a.offset).map_err(|_| Error::LengthTooLarge { len: a.offset })?,
        u32::try_from(b.offset).map_err(|_| Error::LengthTooLarge { len: b.offset })?,
        u32::try_from(c.offset).map_err(|_| Error::LengthTooLarge { len: c.offset })?,
        u32::try_from(d.offset).map_err(|_| Error::LengthTooLarge { len: d.offset })?,
        u32::try_from(e.offset).map_err(|_| Error::LengthTooLarge { len: e.offset })?,
        u32::try_from(f.offset).map_err(|_| Error::LengthTooLarge { len: f.offset })?,
        u32::try_from(g.offset).map_err(|_| Error::LengthTooLarge { len: g.offset })?,
    ];
    let output_offsets = [
        u32::try_from(out_a.offset).map_err(|_| Error::LengthTooLarge { len: out_a.offset })?,
        u32::try_from(out_b.offset).map_err(|_| Error::LengthTooLarge { len: out_b.offset })?,
        u32::try_from(out_c.offset).map_err(|_| Error::LengthTooLarge { len: out_c.offset })?,
        u32::try_from(out_d.offset).map_err(|_| Error::LengthTooLarge { len: out_d.offset })?,
        u32::try_from(out_e.offset).map_err(|_| Error::LengthTooLarge { len: out_e.offset })?,
        u32::try_from(out_f.offset).map_err(|_| Error::LengthTooLarge { len: out_f.offset })?,
        u32::try_from(out_g.offset).map_err(|_| Error::LengthTooLarge { len: out_g.offset })?,
    ];
    let offsets_handle = client.create_from_slice(u32::as_bytes(&offsets));
    let output_offsets_handle = client.create_from_slice(u32::as_bytes(&output_offsets));
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        inclusive_scan_tuple7_by_flags_view_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a.source.handle.clone(), a.source.len()),
            BufferArg::from_raw_parts(b.source.handle.clone(), b.source.len()),
            BufferArg::from_raw_parts(c.source.handle.clone(), c.source.len()),
            BufferArg::from_raw_parts(d.source.handle.clone(), d.source.len()),
            BufferArg::from_raw_parts(e.source.handle.clone(), e.source.len()),
            BufferArg::from_raw_parts(f.source.handle.clone(), f.source.len()),
            BufferArg::from_raw_parts(g.source.handle.clone(), g.source.len()),
            BufferArg::from_raw_parts(offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(out_a.source.handle.clone(), out_a.source.len()),
            BufferArg::from_raw_parts(out_b.source.handle.clone(), out_b.source.len()),
            BufferArg::from_raw_parts(out_c.source.handle.clone(), out_c.source.len()),
            BufferArg::from_raw_parts(out_d.source.handle.clone(), out_d.source.len()),
            BufferArg::from_raw_parts(out_e.source.handle.clone(), out_e.source.len()),
            BufferArg::from_raw_parts(out_f.source.handle.clone(), out_f.source.len()),
            BufferArg::from_raw_parts(out_g.source.handle.clone(), out_g.source.len()),
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn exclusive_scan_by_flags_seven_views<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    a: &DeviceColumnView<R, A>,
    b: &DeviceColumnView<R, B>,
    c: &DeviceColumnView<R, C>,
    d: &DeviceColumnView<R, D>,
    e: &DeviceColumnView<R, E>,
    f: &DeviceColumnView<R, F>,
    g: &DeviceColumnView<R, G>,
    control: &ScanByKeyControl<R>,
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
    R: Runtime,
    A: MStorageElement + 'static,
    B: MStorageElement + 'static,
    C: MStorageElement + 'static,
    D: MStorageElement + 'static,
    E: MStorageElement + 'static,
    F: MStorageElement + 'static,
    G: MStorageElement + 'static,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    ensure_same_len(a.len, control.len)?;
    ensure_same_len(b.len, control.len)?;
    ensure_same_len(c.len, control.len)?;
    ensure_same_len(d.len, control.len)?;
    ensure_same_len(e.len, control.len)?;
    ensure_same_len(f.len, control.len)?;
    ensure_same_len(g.len, control.len)?;
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

    let client = policy.client();
    let offsets = [
        u32::try_from(a.offset).map_err(|_| Error::LengthTooLarge { len: a.offset })?,
        u32::try_from(b.offset).map_err(|_| Error::LengthTooLarge { len: b.offset })?,
        u32::try_from(c.offset).map_err(|_| Error::LengthTooLarge { len: c.offset })?,
        u32::try_from(d.offset).map_err(|_| Error::LengthTooLarge { len: d.offset })?,
        u32::try_from(e.offset).map_err(|_| Error::LengthTooLarge { len: e.offset })?,
        u32::try_from(f.offset).map_err(|_| Error::LengthTooLarge { len: f.offset })?,
        u32::try_from(g.offset).map_err(|_| Error::LengthTooLarge { len: g.offset })?,
    ];
    let offsets_handle = client.create_from_slice(u32::as_bytes(&offsets));
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let output_offsets = client.create_from_slice(u32::as_bytes(&[0, 0, 0, 0, 0, 0, 0]));
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(C::as_bytes(&[init.2]));
    let init_d = client.create_from_slice(D::as_bytes(&[init.3]));
    let init_e = client.create_from_slice(E::as_bytes(&[init.4]));
    let init_f = client.create_from_slice(F::as_bytes(&[init.5]));
    let init_g = client.create_from_slice(G::as_bytes(&[init.6]));
    let out_a = client.empty(control.len * std::mem::size_of::<A>());
    let out_b = client.empty(control.len * std::mem::size_of::<B>());
    let out_c = client.empty(control.len * std::mem::size_of::<C>());
    let out_d = client.empty(control.len * std::mem::size_of::<D>());
    let out_e = client.empty(control.len * std::mem::size_of::<E>());
    let out_f = client.empty(control.len * std::mem::size_of::<F>());
    let out_g = client.empty(control.len * std::mem::size_of::<G>());
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        exclusive_scan_tuple7_by_flags_view_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a.source.handle.clone(), a.source.len()),
            BufferArg::from_raw_parts(b.source.handle.clone(), b.source.len()),
            BufferArg::from_raw_parts(c.source.handle.clone(), c.source.len()),
            BufferArg::from_raw_parts(d.source.handle.clone(), d.source.len()),
            BufferArg::from_raw_parts(e.source.handle.clone(), e.source.len()),
            BufferArg::from_raw_parts(f.source.handle.clone(), f.source.len()),
            BufferArg::from_raw_parts(g.source.handle.clone(), g.source.len()),
            BufferArg::from_raw_parts(offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(init_a.clone(), 1),
            BufferArg::from_raw_parts(init_b.clone(), 1),
            BufferArg::from_raw_parts(init_c.clone(), 1),
            BufferArg::from_raw_parts(init_d.clone(), 1),
            BufferArg::from_raw_parts(init_e.clone(), 1),
            BufferArg::from_raw_parts(init_f.clone(), 1),
            BufferArg::from_raw_parts(init_g.clone(), 1),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offsets.clone(), 7),
            BufferArg::from_raw_parts(out_a.clone(), control.len),
            BufferArg::from_raw_parts(out_b.clone(), control.len),
            BufferArg::from_raw_parts(out_c.clone(), control.len),
            BufferArg::from_raw_parts(out_d.clone(), control.len),
            BufferArg::from_raw_parts(out_e.clone(), control.len),
            BufferArg::from_raw_parts(out_f.clone(), control.len),
            BufferArg::from_raw_parts(out_g.clone(), control.len),
        );
    }

    Ok((
        DeviceVec::from_handle(policy.id(), out_a, control.len),
        DeviceVec::from_handle(policy.id(), out_b, control.len),
        DeviceVec::from_handle(policy.id(), out_c, control.len),
        DeviceVec::from_handle(policy.id(), out_d, control.len),
        DeviceVec::from_handle(policy.id(), out_e, control.len),
        DeviceVec::from_handle(policy.id(), out_f, control.len),
        DeviceVec::from_handle(policy.id(), out_g, control.len),
    ))
}

#[allow(dead_code, clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn exclusive_scan_by_flags_seven_views_into<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    a: &DeviceColumnView<R, A>,
    b: &DeviceColumnView<R, B>,
    c: &DeviceColumnView<R, C>,
    d: &DeviceColumnView<R, D>,
    e: &DeviceColumnView<R, E>,
    f: &DeviceColumnView<R, F>,
    g: &DeviceColumnView<R, G>,
    control: &ScanByKeyControl<R>,
    init: (A, B, C, D, E, F, G),
    out_a: &DeviceColumnMutView<R, A>,
    out_b: &DeviceColumnMutView<R, B>,
    out_c: &DeviceColumnMutView<R, C>,
    out_d: &DeviceColumnMutView<R, D>,
    out_e: &DeviceColumnMutView<R, E>,
    out_f: &DeviceColumnMutView<R, F>,
    out_g: &DeviceColumnMutView<R, G>,
) -> Result<(), Error>
where
    R: Runtime,
    A: MStorageElement + 'static,
    B: MStorageElement + 'static,
    C: MStorageElement + 'static,
    D: MStorageElement + 'static,
    E: MStorageElement + 'static,
    F: MStorageElement + 'static,
    G: MStorageElement + 'static,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    ensure_same_len(a.len, control.len)?;
    ensure_same_len(b.len, control.len)?;
    ensure_same_len(c.len, control.len)?;
    ensure_same_len(d.len, control.len)?;
    ensure_same_len(e.len, control.len)?;
    ensure_same_len(f.len, control.len)?;
    ensure_same_len(g.len, control.len)?;
    ensure_same_len(out_a.len, control.len)?;
    ensure_same_len(out_b.len, control.len)?;
    ensure_same_len(out_c.len, control.len)?;
    ensure_same_len(out_d.len, control.len)?;
    ensure_same_len(out_e.len, control.len)?;
    ensure_same_len(out_f.len, control.len)?;
    ensure_same_len(out_g.len, control.len)?;
    if control.len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let offsets = [
        u32::try_from(a.offset).map_err(|_| Error::LengthTooLarge { len: a.offset })?,
        u32::try_from(b.offset).map_err(|_| Error::LengthTooLarge { len: b.offset })?,
        u32::try_from(c.offset).map_err(|_| Error::LengthTooLarge { len: c.offset })?,
        u32::try_from(d.offset).map_err(|_| Error::LengthTooLarge { len: d.offset })?,
        u32::try_from(e.offset).map_err(|_| Error::LengthTooLarge { len: e.offset })?,
        u32::try_from(f.offset).map_err(|_| Error::LengthTooLarge { len: f.offset })?,
        u32::try_from(g.offset).map_err(|_| Error::LengthTooLarge { len: g.offset })?,
    ];
    let output_offsets = [
        u32::try_from(out_a.offset).map_err(|_| Error::LengthTooLarge { len: out_a.offset })?,
        u32::try_from(out_b.offset).map_err(|_| Error::LengthTooLarge { len: out_b.offset })?,
        u32::try_from(out_c.offset).map_err(|_| Error::LengthTooLarge { len: out_c.offset })?,
        u32::try_from(out_d.offset).map_err(|_| Error::LengthTooLarge { len: out_d.offset })?,
        u32::try_from(out_e.offset).map_err(|_| Error::LengthTooLarge { len: out_e.offset })?,
        u32::try_from(out_f.offset).map_err(|_| Error::LengthTooLarge { len: out_f.offset })?,
        u32::try_from(out_g.offset).map_err(|_| Error::LengthTooLarge { len: out_g.offset })?,
    ];
    let offsets_handle = client.create_from_slice(u32::as_bytes(&offsets));
    let output_offsets_handle = client.create_from_slice(u32::as_bytes(&output_offsets));
    let len_handle = client.create_from_slice(u32::as_bytes(&[control.len_u32]));
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(C::as_bytes(&[init.2]));
    let init_d = client.create_from_slice(D::as_bytes(&[init.3]));
    let init_e = client.create_from_slice(E::as_bytes(&[init.4]));
    let init_f = client.create_from_slice(F::as_bytes(&[init.5]));
    let init_g = client.create_from_slice(G::as_bytes(&[init.6]));
    let num_blocks = control
        .len
        .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        exclusive_scan_tuple7_by_flags_view_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            crate::detail::launch::cube_count_1d(num_blocks_u32),
            CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a.source.handle.clone(), a.source.len()),
            BufferArg::from_raw_parts(b.source.handle.clone(), b.source.len()),
            BufferArg::from_raw_parts(c.source.handle.clone(), c.source.len()),
            BufferArg::from_raw_parts(d.source.handle.clone(), d.source.len()),
            BufferArg::from_raw_parts(e.source.handle.clone(), e.source.len()),
            BufferArg::from_raw_parts(f.source.handle.clone(), f.source.len()),
            BufferArg::from_raw_parts(g.source.handle.clone(), g.source.len()),
            BufferArg::from_raw_parts(offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(control.head_flags.clone(), control.len),
            BufferArg::from_raw_parts(init_a.clone(), 1),
            BufferArg::from_raw_parts(init_b.clone(), 1),
            BufferArg::from_raw_parts(init_c.clone(), 1),
            BufferArg::from_raw_parts(init_d.clone(), 1),
            BufferArg::from_raw_parts(init_e.clone(), 1),
            BufferArg::from_raw_parts(init_f.clone(), 1),
            BufferArg::from_raw_parts(init_g.clone(), 1),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(out_a.source.handle.clone(), out_a.source.len()),
            BufferArg::from_raw_parts(out_b.source.handle.clone(), out_b.source.len()),
            BufferArg::from_raw_parts(out_c.source.handle.clone(), out_c.source.len()),
            BufferArg::from_raw_parts(out_d.source.handle.clone(), out_d.source.len()),
            BufferArg::from_raw_parts(out_e.source.handle.clone(), out_e.source.len()),
            BufferArg::from_raw_parts(out_f.source.handle.clone(), out_f.source.len()),
            BufferArg::from_raw_parts(out_g.source.handle.clone(), out_g.source.len()),
        );
    }

    Ok(())
}

impl<KeySource, KeyEq> KernelScanByKeyKeys<KeyEq> for KeySource
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: MStorageElement + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type Control = ScanByKeyControl<KeySource::Runtime>;

    fn scan_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Control, Error> {
        <KeySource as KernelColumn>::validate(&self)?;
        let len = <KeySource as KernelColumn>::len(&self);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let head_flags = scan_by_key_head_flags_read::<KeySource, KeyEq>(policy, &self)?;
        let segment = SegmentControl::from_head_flags(head_flags, len, len_u32);
        Ok(ScanByKeyControl::from_segment(&segment))
    }
}

macro_rules! impl_kernel_scan_by_key_keys_tuple1 {
    ($target:ty, $field:tt) => {
        impl<KeySource, KeyEq> KernelScanByKeyKeys<KeyEq> for $target
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            KeySource::Item: MStorageElement + 'static,
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

impl_kernel_scan_by_key_keys_tuple1!(ZipView1<KeySource>, source);
impl_kernel_scan_by_key_keys_tuple1!(DeviceZip1<KeySource>, source);

impl<KeySource, KeyEq> KernelScanByKeyKeys<KeyEq> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: MStorageElement + 'static,
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

impl<First, Second, KeyEq> KernelScanByKeyKeys<KeyEq> for (First, Second)
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: MStorageElement + 'static,
    Second::Item: MStorageElement + 'static,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    KeyEq: BinaryPredicateOp<(First::Item, Second::Item)>,
{
    type Runtime = First::Runtime;
    type Control = ScanByKeyControl<First::Runtime>;

    fn scan_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Control, Error> {
        <First as KernelColumn>::validate(&self.0)?;
        <Second as KernelColumn>::validate(&self.1)?;
        let len = <First as KernelColumn>::len(&self.0);
        if len != <Second as KernelColumn>::len(&self.1) {
            return Err(Error::LengthMismatch {
                input: len,
                output: <Second as KernelColumn>::len(&self.1),
            });
        }
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let head_flags = super::super::selection::unique_tuple2_flags_read::<First, Second, KeyEq>(
            policy, &self.0, &self.1,
        )?;
        let segment = SegmentControl::from_head_flags(head_flags, len, len_u32);
        Ok(ScanByKeyControl::from_segment(&segment))
    }
}

impl<First, Second, Third, KeyEq> KernelScanByKeyKeys<KeyEq> for (First, Second, Third)
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: MStorageElement + 'static,
    Second::Item: MStorageElement + 'static,
    Third::Item: MStorageElement + 'static,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    KeyEq: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    type Runtime = First::Runtime;
    type Control = ScanByKeyControl<First::Runtime>;

    fn scan_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Control, Error> {
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
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let head_flags =
            super::super::selection::unique_tuple3_flags_read::<First, Second, Third, KeyEq>(
                policy, &self.0, &self.1, &self.2,
            )?;
        let segment = SegmentControl::from_head_flags(head_flags, len, len_u32);
        Ok(ScanByKeyControl::from_segment(&segment))
    }
}

macro_rules! impl_kernel_scan_by_key_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S, KeyEq, Op> KernelInclusiveScanByKeyValues<ScanByKeyControl<S::Runtime>, KeyEq, Op>
            for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: MStorageElement + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Output = DeviceZip1<DeviceVec<S::Runtime, S::Item>>;

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<S::Runtime>,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                Ok(DeviceZip1 {
                    source: apply.inclusive_expr::<S, Op>(policy, &self.$field)?,
                })
            }
        }

        impl<S, KeyEq, Op> KernelExclusiveScanByKeyValues<ScanByKeyControl<S::Runtime>, KeyEq, Op>
            for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: MStorageElement + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Init = S::Item;
            type Output = DeviceZip1<DeviceVec<S::Runtime, S::Item>>;

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<S::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                Ok(DeviceZip1 {
                    source: apply.exclusive_expr::<S, Op>(policy, &self.$field, init)?,
                })
            }
        }
    };
}

impl_kernel_scan_by_key_tuple1!((S,), 0);
impl_kernel_scan_by_key_tuple1!(ZipView1<S>, source);
impl_kernel_scan_by_key_tuple1!(DeviceZip1<S>, source);

macro_rules! impl_kernel_scan_by_key_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<A, C, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: MStorageElement + 'static,
            C::Item: MStorageElement + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            (A::Item, C::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item)>,
        {
            type Runtime = A::Runtime;
            type Output =
                DeviceZip2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<A::Runtime>,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.inclusive_expr2::<A, C, Op>(policy, &self.$left, &self.$right)
            }
        }

        impl<A, C, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: MStorageElement + 'static,
            C::Item: MStorageElement + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            (A::Item, C::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item)>,
        {
            type Runtime = A::Runtime;
            type Init = (A::Item, C::Item);
            type Output =
                DeviceZip2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<A::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.exclusive_expr2::<A, C, Op>(policy, &self.$left, &self.$right, init)
            }
        }
    };
}

impl_kernel_scan_by_key_tuple2!(ZipView2<A, C>, left, right);
impl_kernel_scan_by_key_tuple2!(DeviceZip2<A, C>, left, right);

macro_rules! impl_kernel_scan_by_key_tuple3 {
    ($target:ty, $first:tt, $second:tt, $third:tt) => {
        impl<A, C, D, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: MStorageElement + 'static,
            C::Item: MStorageElement + 'static,
            D::Item: MStorageElement + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            D::Expr: DeviceGpuExpr<D::Item>,
            (A::Item, C::Item, D::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item, D::Item)>,
        {
            type Runtime = A::Runtime;
            type Output = DeviceZip3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &ScanByKeyControl<A::Runtime>,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.inclusive_expr3::<A, C, D, Op>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                )
            }
        }

        impl<A, C, D, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<A::Runtime>, KeyEq, Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: MStorageElement + 'static,
            C::Item: MStorageElement + 'static,
            D::Item: MStorageElement + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            D::Expr: DeviceGpuExpr<D::Item>,
            (A::Item, C::Item, D::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item, D::Item)>,
        {
            type Runtime = A::Runtime;
            type Init = (A::Item, C::Item, D::Item);
            type Output = DeviceZip3<
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
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.exclusive_expr3::<A, C, D, Op>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    init,
                )
            }
        }
    };
}

impl_kernel_scan_by_key_tuple3!(ZipView3<A, C, D>, first, second, third);
impl_kernel_scan_by_key_tuple3!(DeviceZip3<A, C, D>, first, second, third);

macro_rules! impl_kernel_scan_by_key_tuple4_views {
    () => {
        impl<R, A, B, C, D, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
            )
        where
            R: Runtime,
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D)>,
        {
            type Runtime = R;
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
            );

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.inclusive_views4::<A, B, C, D, Op>(policy, &self.0, &self.1, &self.2, &self.3)
            }
        }

        impl<R, A, B, C, D, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
            )
        where
            R: Runtime,
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D);
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
            );

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.exclusive_views4::<A, B, C, D, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, init,
                )
            }
        }
    };
}

macro_rules! impl_kernel_scan_by_key_tuple5_views {
    () => {
        impl<R, A, B, C, D, E, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
                DeviceColumnView<R, E>,
            )
        where
            R: Runtime,
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            E: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D, E)>,
        {
            type Runtime = R;
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
            );

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.inclusive_views5::<A, B, C, D, E, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, &self.4,
                )
            }
        }

        impl<R, A, B, C, D, E, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
            for (
                DeviceColumnView<R, A>,
                DeviceColumnView<R, B>,
                DeviceColumnView<R, C>,
                DeviceColumnView<R, D>,
                DeviceColumnView<R, E>,
            )
        where
            R: Runtime,
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            E: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D, E)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D, E);
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
            );

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.exclusive_views5::<A, B, C, D, E, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, &self.4, init,
                )
            }
        }
    };
}

macro_rules! impl_kernel_scan_by_key_tuple6_views {
    () => {
        impl<R, A, B, C, D, E, F, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
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
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            E: MStorageElement + 'static,
            F: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D, E, F)>,
        {
            type Runtime = R;
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
                DeviceVec<R, F>,
            );

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.inclusive_views6::<A, B, C, D, E, F, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, &self.4, &self.5,
                )
            }
        }

        impl<R, A, B, C, D, E, F, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
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
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            E: MStorageElement + 'static,
            F: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D, E, F)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D, E, F);
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
                DeviceVec<R, F>,
            );

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.exclusive_views6::<A, B, C, D, E, F, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, &self.4, &self.5, init,
                )
            }
        }
    };
}

macro_rules! impl_kernel_scan_by_key_tuple7_views {
    () => {
        impl<R, A, B, C, D, E, F, G, KeyEq, Op>
            KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
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
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            E: MStorageElement + 'static,
            F: MStorageElement + 'static,
            G: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D, E, F, G)>,
        {
            type Runtime = R;
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
                DeviceVec<R, F>,
                DeviceVec<R, G>,
            );

            fn inclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.inclusive_views7::<A, B, C, D, E, F, G, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, &self.4, &self.5, &self.6,
                )
            }
        }

        impl<R, A, B, C, D, E, F, G, KeyEq, Op>
            KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
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
            A: MStorageElement + 'static,
            B: MStorageElement + 'static,
            C: MStorageElement + 'static,
            D: MStorageElement + 'static,
            E: MStorageElement + 'static,
            F: MStorageElement + 'static,
            G: MStorageElement + 'static,
            Op: BinaryOp<(A, B, C, D, E, F, G)>,
        {
            type Runtime = R;
            type Init = (A, B, C, D, E, F, G);
            type Output = (
                DeviceVec<R, A>,
                DeviceVec<R, B>,
                DeviceVec<R, C>,
                DeviceVec<R, D>,
                DeviceVec<R, E>,
                DeviceVec<R, F>,
                DeviceVec<R, G>,
            );

            fn exclusive_scan_by_key_values(
                self,
                policy: &CubePolicy<R>,
                control: &ScanByKeyControl<R>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                apply.exclusive_views7::<A, B, C, D, E, F, G, Op>(
                    policy, &self.0, &self.1, &self.2, &self.3, &self.4, &self.5, &self.6, init,
                )
            }
        }
    };
}

impl_kernel_scan_by_key_tuple4_views!();
impl_kernel_scan_by_key_tuple5_views!();
impl_kernel_scan_by_key_tuple6_views!();
impl_kernel_scan_by_key_tuple7_views!();

impl<Left, Right, R, KeyEq, Op> KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>
    for (Left, Right)
where
    R: Runtime,
    ZipView2<Left, Right>: KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <ZipView2<Left, Right> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Output = <ZipView2<Left, Right> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Output;

    fn inclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ScanByKeyControl<R>,
    ) -> Result<Self::Output, Error> {
        <ZipView2<Left, Right> as KernelInclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::inclusive_scan_by_key_values(
            ZipView2 {
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
    ZipView3<First, Second, Third>: KernelInclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <ZipView3<First, Second, Third> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Output = <ZipView3<First, Second, Third> as KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Output;

    fn inclusive_scan_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &ScanByKeyControl<R>,
    ) -> Result<Self::Output, Error> {
        <ZipView3<First, Second, Third> as KernelInclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::inclusive_scan_by_key_values(
            ZipView3 {
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
    ZipView2<Left, Right>: KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <ZipView2<Left, Right> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Init = <ZipView2<Left, Right> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Init;
    type Output = <ZipView2<Left, Right> as KernelExclusiveScanByKeyValues<
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
        <ZipView2<Left, Right> as KernelExclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::exclusive_scan_by_key_values(
            ZipView2 {
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
    ZipView3<First, Second, Third>: KernelExclusiveScanByKeyValues<ScanByKeyControl<R>, KeyEq, Op>,
{
    type Runtime = <ZipView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Runtime;
    type Init = <ZipView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KeyEq,
        Op,
    >>::Init;
    type Output = <ZipView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
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
        <ZipView3<First, Second, Third> as KernelExclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KeyEq,
            Op,
        >>::exclusive_scan_by_key_values(
            ZipView3 {
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
