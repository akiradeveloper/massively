use crate::{
    device::{DeviceVec, KernelColumnBindings},
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr, Input},
    kernels::*,
    op::{BinaryOp, BinaryPredicateOp},
    policy::CubePolicy,
    primitives::{scan, segmented, workspace::Workspace},
};
use cubecl::prelude::*;

pub(crate) const BLOCK_REDUCE_SIZE: u32 = 256;

pub(crate) type ReduceByKeyControl = segmented::SegmentControl;

pub(crate) fn apply_tuple2_init<R, A, B, Op>(
    policy: &CubePolicy<R>,
    left: &DeviceVec<R, A>,
    right: &DeviceVec<R, B>,
    init: (A, B),
) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B)>,
{
    super::ensure_same_len(right.len(), left.len())?;
    let len = left.len();
    if len == 0 {
        return Ok((policy.empty_device_vec(), policy.empty_device_vec()));
    }
    let client = policy.client();
    let out_left = client.empty(len * std::mem::size_of::<A>());
    let out_right = client.empty(len * std::mem::size_of::<B>());
    let init_left = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_right = client.create_from_slice(B::as_bytes(&[init.1]));
    let blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let blocks_u32 = u32::try_from(blocks).map_err(|_| Error::LengthTooLarge { len: blocks })?;
    unsafe {
        tuple2_apply_init_kernel::launch_unchecked::<A, B, Op, R>(
            client,
            CubeCount::Static(blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            unsafe { BufferArg::from_raw_parts(left.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(right.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_left.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_right.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(out_left.clone(), len) },
            unsafe { BufferArg::from_raw_parts(out_right.clone(), len) },
        );
    }
    Ok((
        DeviceVec::from_handle(policy.id(), out_left, len),
        DeviceVec::from_handle(policy.id(), out_right, len),
    ))
}

pub(crate) fn apply_tuple3_init<R, A, B, C, Op>(
    policy: &CubePolicy<R>,
    first: &DeviceVec<R, A>,
    second: &DeviceVec<R, B>,
    third: &DeviceVec<R, C>,
    init: (A, B, C),
) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C)>,
{
    super::ensure_same_len(second.len(), first.len())?;
    super::ensure_same_len(third.len(), first.len())?;
    let len = first.len();
    if len == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
        ));
    }
    let client = policy.client();
    let out_first = client.empty(len * std::mem::size_of::<A>());
    let out_second = client.empty(len * std::mem::size_of::<B>());
    let out_third = client.empty(len * std::mem::size_of::<C>());
    let init_first = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_second = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_third = client.create_from_slice(C::as_bytes(&[init.2]));
    let blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let blocks_u32 = u32::try_from(blocks).map_err(|_| Error::LengthTooLarge { len: blocks })?;
    unsafe {
        tuple3_apply_init_kernel::launch_unchecked::<A, B, C, Op, R>(
            client,
            CubeCount::Static(blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            unsafe { BufferArg::from_raw_parts(first.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(second.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(third.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_first.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_second.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_third.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(out_first.clone(), len) },
            unsafe { BufferArg::from_raw_parts(out_second.clone(), len) },
            unsafe { BufferArg::from_raw_parts(out_third.clone(), len) },
        );
    }
    Ok((
        DeviceVec::from_handle(policy.id(), out_first, len),
        DeviceVec::from_handle(policy.id(), out_second, len),
        DeviceVec::from_handle(policy.id(), out_third, len),
    ))
}

pub(crate) fn reduce_input_handle<R, T, Op>(
    policy: &CubePolicy<R>,
    input_handle: cubecl::server::Handle,
    _storage_len: usize,
    len: usize,
    init: T,
) -> Result<T, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
    Input<T>: GpuExpr<T>,
{
    if len == 0 {
        return Ok(init);
    }

    let client = policy.client();
    let partial_handle = inclusive_scan_handle::<R, T, Op>(policy, input_handle, len)?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    let output_handle = client.empty(std::mem::size_of::<T>());
    unsafe {
        scalar_reduce_last_finalize_kernel::launch_unchecked::<T, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(partial_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), 1) },
        );
    }
    read_one(policy, output_handle)
}

fn inclusive_scan_handle<R, T, Op>(
    policy: &CubePolicy<R>,
    input_handle: cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
{
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let client = policy.client();
    let workspace = Workspace::new(policy);
    let output_handle = workspace.alloc::<T>(len);
    let num_blocks = len.div_ceil(scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums = workspace.alloc::<T>(num_blocks);

    unsafe {
        scalar_inclusive_scan_block_kernel::launch_unchecked::<T, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(scan::BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(input_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let block_prefixes = inclusive_scan_handle::<R, T, Op>(policy, block_sums, num_blocks)?;
        unsafe {
            scalar_scan_add_block_prefix_kernel::launch_unchecked::<T, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(scan::BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }
    }

    Ok(output_handle)
}

fn read_one<R, T>(policy: &CubePolicy<R>, handle: cubecl::server::Handle) -> Result<T, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let bytes = policy
        .client()
        .read_one(handle)
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    Ok(T::from_bytes(&bytes)[0].clone())
}

pub(crate) fn reduce_tuple1_device_expr<R, A, ExprA, Op>(
    policy: &CubePolicy<R>,
    a: &KernelColumnBindings,
    len: usize,
    init: (A,),
) -> Result<(A,), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    Op: BinaryOp<(A,)>,
{
    if len == 0 {
        return Ok(init);
    }

    let client = policy.client();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let partial = scan::inclusive_scan_tuple1_device_expr::<R, A, ExprA, Op>(policy, a, len)?;

    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let output_a = client.empty(std::mem::size_of::<A>());
    unsafe {
        tuple1_reduce_last_finalize_kernel::launch_unchecked::<A, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(partial.source.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), 1) },
        );
    }

    Ok((read_one(policy, output_a)?,))
}

pub(crate) fn reduce_tuple2_device_expr<R, A, B, ExprA, ExprB, Op>(
    policy: &CubePolicy<R>,
    a: &KernelColumnBindings,
    b: &KernelColumnBindings,
    len: usize,
    init: (A, B),
) -> Result<(A, B), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp<(A, B)>,
{
    if len == 0 {
        return Ok(init);
    }

    let client = policy.client();
    let workspace = Workspace::new(policy);
    let partial_len = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let partial_len_u32 =
        u32::try_from(partial_len).map_err(|_| Error::LengthTooLarge { len: partial_len })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let partial_a = workspace.alloc::<A>(partial_len);
    let partial_b = workspace.alloc::<B>(partial_len);
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

    unsafe {
        tuple2_device_reduce_expr_partials_kernel::launch_unchecked::<A, B, ExprA, ExprB, Op, R>(
            client,
            CubeCount::Static(partial_len_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            unsafe { BufferArg::from_raw_parts(a0.0.clone(), a0.1) },
            unsafe { BufferArg::from_raw_parts(a1.0.clone(), a1.1) },
            unsafe { BufferArg::from_raw_parts(a2.0.clone(), a2.1) },
            unsafe { BufferArg::from_raw_parts(a3.0.clone(), a3.1) },
            unsafe { BufferArg::from_raw_parts(a_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b0.0.clone(), b0.1) },
            unsafe { BufferArg::from_raw_parts(b1.0.clone(), b1.1) },
            unsafe { BufferArg::from_raw_parts(b2.0.clone(), b2.1) },
            unsafe { BufferArg::from_raw_parts(b3.0.clone(), b3.1) },
            unsafe { BufferArg::from_raw_parts(b_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(partial_a.clone(), partial_len) },
            unsafe { BufferArg::from_raw_parts(partial_b.clone(), partial_len) },
        );
    }

    let mut current_a = partial_a;
    let mut current_b = partial_b;
    let mut current_len = partial_len;
    while current_len > 1 {
        let next_len = current_len.div_ceil(BLOCK_REDUCE_SIZE as usize);
        let next_len_u32 =
            u32::try_from(next_len).map_err(|_| Error::LengthTooLarge { len: next_len })?;
        let current_len_u32 =
            u32::try_from(current_len).map_err(|_| Error::LengthTooLarge { len: current_len })?;
        let current_len_handle = client.create_from_slice(u32::as_bytes(&[current_len_u32]));
        let next_a = workspace.alloc::<A>(next_len);
        let next_b = workspace.alloc::<B>(next_len);
        unsafe {
            tuple2_reduce_partials_kernel::launch_unchecked::<A, B, Op, R>(
                client,
                CubeCount::Static(next_len_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                unsafe { BufferArg::from_raw_parts(current_a.clone(), current_len) },
                unsafe { BufferArg::from_raw_parts(current_b.clone(), current_len) },
                unsafe { BufferArg::from_raw_parts(current_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(next_a.clone(), next_len) },
                unsafe { BufferArg::from_raw_parts(next_b.clone(), next_len) },
            );
        }
        current_a = next_a;
        current_b = next_b;
        current_len = next_len;
    }

    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let output_a = client.empty(std::mem::size_of::<A>());
    let output_b = client.empty(std::mem::size_of::<B>());
    unsafe {
        tuple2_reduce_finalize_kernel::launch_unchecked::<A, B, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(current_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(current_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), 1) },
        );
    }

    Ok((read_one(policy, output_a)?, read_one(policy, output_b)?))
}

pub(crate) fn reduce_tuple3_device_expr<R, A, B, C, ExprA, ExprB, ExprC, Op>(
    policy: &CubePolicy<R>,
    a: &KernelColumnBindings,
    b: &KernelColumnBindings,
    c: &KernelColumnBindings,
    len: usize,
    init: (A, B, C),
) -> Result<(A, B, C), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    Op: BinaryOp<(A, B, C)>,
{
    if len == 0 {
        return Ok(init);
    }

    let client = policy.client();
    let workspace = Workspace::new(policy);
    let partial_len = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let partial_len_u32 =
        u32::try_from(partial_len).map_err(|_| Error::LengthTooLarge { len: partial_len })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let partial_a = workspace.alloc::<A>(partial_len);
    let partial_b = workspace.alloc::<B>(partial_len);
    let partial_c = workspace.alloc::<C>(partial_len);
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

    unsafe {
        tuple3_device_reduce_expr_partials_kernel::launch_unchecked::<
            A,
            B,
            C,
            ExprA,
            ExprB,
            ExprC,
            Op,
            R,
        >(
            client,
            CubeCount::Static(partial_len_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            unsafe { BufferArg::from_raw_parts(a0.0.clone(), a0.1) },
            unsafe { BufferArg::from_raw_parts(a1.0.clone(), a1.1) },
            unsafe { BufferArg::from_raw_parts(a2.0.clone(), a2.1) },
            unsafe { BufferArg::from_raw_parts(a3.0.clone(), a3.1) },
            unsafe { BufferArg::from_raw_parts(a_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b0.0.clone(), b0.1) },
            unsafe { BufferArg::from_raw_parts(b1.0.clone(), b1.1) },
            unsafe { BufferArg::from_raw_parts(b2.0.clone(), b2.1) },
            unsafe { BufferArg::from_raw_parts(b3.0.clone(), b3.1) },
            unsafe { BufferArg::from_raw_parts(b_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(c0.0.clone(), c0.1) },
            unsafe { BufferArg::from_raw_parts(c1.0.clone(), c1.1) },
            unsafe { BufferArg::from_raw_parts(c2.0.clone(), c2.1) },
            unsafe { BufferArg::from_raw_parts(c3.0.clone(), c3.1) },
            unsafe { BufferArg::from_raw_parts(c_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(partial_a.clone(), partial_len) },
            unsafe { BufferArg::from_raw_parts(partial_b.clone(), partial_len) },
            unsafe { BufferArg::from_raw_parts(partial_c.clone(), partial_len) },
        );
    }

    let mut current_a = partial_a;
    let mut current_b = partial_b;
    let mut current_c = partial_c;
    let mut current_len = partial_len;
    while current_len > 1 {
        let next_len = current_len.div_ceil(BLOCK_REDUCE_SIZE as usize);
        let next_len_u32 =
            u32::try_from(next_len).map_err(|_| Error::LengthTooLarge { len: next_len })?;
        let current_len_u32 =
            u32::try_from(current_len).map_err(|_| Error::LengthTooLarge { len: current_len })?;
        let current_len_handle = client.create_from_slice(u32::as_bytes(&[current_len_u32]));
        let next_a = workspace.alloc::<A>(next_len);
        let next_b = workspace.alloc::<B>(next_len);
        let next_c = workspace.alloc::<C>(next_len);
        unsafe {
            tuple3_reduce_partials_kernel::launch_unchecked::<A, B, C, Op, R>(
                client,
                CubeCount::Static(next_len_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                unsafe { BufferArg::from_raw_parts(current_a.clone(), current_len) },
                unsafe { BufferArg::from_raw_parts(current_b.clone(), current_len) },
                unsafe { BufferArg::from_raw_parts(current_c.clone(), current_len) },
                unsafe { BufferArg::from_raw_parts(current_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(next_a.clone(), next_len) },
                unsafe { BufferArg::from_raw_parts(next_b.clone(), next_len) },
                unsafe { BufferArg::from_raw_parts(next_c.clone(), next_len) },
            );
        }
        current_a = next_a;
        current_b = next_b;
        current_c = next_c;
        current_len = next_len;
    }

    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(C::as_bytes(&[init.2]));
    let output_a = client.empty(std::mem::size_of::<A>());
    let output_b = client.empty(std::mem::size_of::<B>());
    let output_c = client.empty(std::mem::size_of::<C>());
    unsafe {
        tuple3_reduce_finalize_kernel::launch_unchecked::<A, B, C, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(current_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(current_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(current_c.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_c.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), 1) },
        );
    }

    Ok((
        read_one(policy, output_a)?,
        read_one(policy, output_b)?,
        read_one(policy, output_c)?,
    ))
}

pub(crate) fn reduce_by_key_handle<R, K, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    value_handle: cubecl::server::Handle,
    init: T,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let (keys, values, _) =
        reduce_by_key_handle_with_control::<R, K, T, KeyEq, Op>(policy, keys, value_handle, init)?;
    Ok((keys, values))
}

pub(crate) fn reduce_by_key_handle_with_control<R, K, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    value_handle: cubecl::server::Handle,
    init: T,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>, ReduceByKeyControl), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            ReduceByKeyControl::empty(policy)?,
        ));
    }

    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let workspace = Workspace::new(policy);
    let local_inclusive_handle = workspace.alloc::<T>(len);
    let scan_blocks = len.div_ceil(scan::BLOCK_SCAN_SIZE as usize);
    let scan_blocks_u32 =
        u32::try_from(scan_blocks).map_err(|_| Error::LengthTooLarge { len: scan_blocks })?;
    let block_tail_keys = workspace.alloc::<K>(scan_blocks);
    let block_tail_values = workspace.alloc::<T>(scan_blocks);
    let flag_handle = workspace.alloc::<u32>(len);
    let reduced_value_handle = workspace.alloc::<T>(len);
    let num_blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        scan_by_key_block_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(scan_blocks_u32, 1, 1),
            CubeDim::new_1d(scan::BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(keys.handle.clone(), keys.len()) },
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(local_inclusive_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), scan_blocks) },
            unsafe { BufferArg::from_raw_parts(block_tail_values.clone(), scan_blocks) },
        );
    }

    if scan_blocks > 1 {
        let block_tail_keys_vec =
            DeviceVec::from_handle(policy.id(), block_tail_keys.clone(), scan_blocks);
        let block_prefixes = scan::inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(
            policy,
            &block_tail_keys_vec,
            &block_tail_values,
        )?;
        unsafe {
            reduce_by_key_end_flags_with_block_prefix_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                unsafe { BufferArg::from_raw_parts(keys.handle.clone(), keys.len()) },
                unsafe { BufferArg::from_raw_parts(local_inclusive_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), scan_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes.clone(), scan_blocks) },
                unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_value_handle.clone(), len) },
            );
        }
    } else {
        unsafe {
            reduce_by_key_end_flags_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                unsafe { BufferArg::from_raw_parts(keys.handle.clone(), keys.len()) },
                unsafe { BufferArg::from_raw_parts(local_inclusive_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(reduced_value_handle.clone(), len) },
            );
        }
    }

    let control =
        ReduceByKeyControl::from_end_flags(policy, len, len_u32, flag_handle, keys.handle.clone())?;
    let (out_keys, out_values) =
        control.compact_pair::<R, K, T>(policy, keys.handle.clone(), reduced_value_handle)?;
    Ok((out_keys, out_values, control))
}

pub(crate) fn reduce_tuple2_by_key_device_vec<R, A, B, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    key_a: &DeviceVec<R, A>,
    key_b: &DeviceVec<R, B>,
    values: &DeviceVec<R, T>,
    init: T,
) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<(A, B)>,
    Op: BinaryOp<T>,
{
    super::ensure_same_len(key_b.len(), key_a.len())?;
    super::ensure_same_len(values.len(), key_a.len())?;

    let len = key_a.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
        ));
    }

    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    let inclusive_handle = scan::inclusive_scan_tuple2_by_key_handle::<R, A, B, T, KeyEq, Op>(
        policy,
        key_a,
        key_b,
        &values.handle,
    )?;
    let workspace = Workspace::new(policy);
    let flag_handle = workspace.alloc::<u32>(len);
    let reduced_value_handle = workspace.alloc::<T>(len);
    let num_blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        reduce_tuple2_by_key_end_flags_kernel::launch_unchecked::<A, B, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            unsafe { BufferArg::from_raw_parts(key_a.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(key_b.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(reduced_value_handle.clone(), len) },
        );
    }

    let control = segmented::SegmentControl::from_end_flags(
        policy,
        len,
        len_u32,
        flag_handle,
        key_a.handle.clone(),
    )?;
    let out_key_a = control.compact_first::<R, A>(policy)?;
    let out_key_b = control.compact_value::<R, B>(policy, key_b.handle.clone())?;
    let out_values = control.compact_value::<R, T>(policy, reduced_value_handle)?;
    Ok((out_key_a, out_key_b, out_values))
}

pub(crate) fn reduce_tuple3_by_key_device_vec<R, A, B, C, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    key_a: &DeviceVec<R, A>,
    key_b: &DeviceVec<R, B>,
    key_c: &DeviceVec<R, C>,
    values: &DeviceVec<R, T>,
    init: T,
) -> Result<
    (
        DeviceVec<R, A>,
        DeviceVec<R, B>,
        DeviceVec<R, C>,
        DeviceVec<R, T>,
    ),
    Error,
>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<(A, B, C)>,
    Op: BinaryOp<T>,
{
    super::ensure_same_len(key_b.len(), key_a.len())?;
    super::ensure_same_len(key_c.len(), key_a.len())?;
    super::ensure_same_len(values.len(), key_a.len())?;

    let len = key_a.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
        ));
    }

    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    let inclusive_handle = scan::inclusive_scan_tuple3_by_key_handle::<R, A, B, C, T, KeyEq, Op>(
        policy,
        key_a,
        key_b,
        key_c,
        &values.handle,
    )?;
    let workspace = Workspace::new(policy);
    let flag_handle = workspace.alloc::<u32>(len);
    let reduced_value_handle = workspace.alloc::<T>(len);
    let num_blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        reduce_tuple3_by_key_end_flags_kernel::launch_unchecked::<A, B, C, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            unsafe { BufferArg::from_raw_parts(key_a.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(key_b.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(key_c.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(reduced_value_handle.clone(), len) },
        );
    }

    let control = segmented::SegmentControl::from_end_flags(
        policy,
        len,
        len_u32,
        flag_handle,
        key_a.handle.clone(),
    )?;
    let out_key_a = control.compact_first::<R, A>(policy)?;
    let out_key_b = control.compact_value::<R, B>(policy, key_b.handle.clone())?;
    let out_key_c = control.compact_value::<R, C>(policy, key_c.handle.clone())?;
    let out_values = control.compact_value::<R, T>(policy, reduced_value_handle)?;
    Ok((out_key_a, out_key_b, out_key_c, out_values))
}

pub(crate) fn reduce_by_key_handle_with_existing_control<R, K, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    value_handle: cubecl::server::Handle,
    init: T,
    control: &ReduceByKeyControl,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let reduced_value_handle = reduce_by_key_values_at_ends_from_input::<R, K, T, KeyEq, Op>(
        policy,
        keys,
        value_handle,
        init,
    )?;
    control.compact_value::<R, T>(policy, reduced_value_handle)
}

fn reduce_by_key_values_at_ends_from_input<R, K, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    value_handle: cubecl::server::Handle,
    init: T,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let client = policy.client();
    let workspace = Workspace::new(policy);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let local_inclusive_handle = workspace.alloc::<T>(len);
    let scan_blocks = len.div_ceil(scan::BLOCK_SCAN_SIZE as usize);
    let scan_blocks_u32 =
        u32::try_from(scan_blocks).map_err(|_| Error::LengthTooLarge { len: scan_blocks })?;
    let block_tail_keys = workspace.alloc::<K>(scan_blocks);
    let block_tail_values = workspace.alloc::<T>(scan_blocks);
    let output_handle = workspace.alloc::<T>(len);
    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    let num_blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        scan_by_key_block_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(scan_blocks_u32, 1, 1),
            CubeDim::new_1d(scan::BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(keys.handle.clone(), keys.len()) },
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(local_inclusive_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), scan_blocks) },
            unsafe { BufferArg::from_raw_parts(block_tail_values.clone(), scan_blocks) },
        );
    }

    if scan_blocks > 1 {
        let block_tail_keys_vec =
            DeviceVec::from_handle(policy.id(), block_tail_keys.clone(), scan_blocks);
        let block_prefixes = scan::inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(
            policy,
            &block_tail_keys_vec,
            &block_tail_values,
        )?;
        unsafe {
            reduce_by_key_values_at_ends_with_block_prefix_kernel::launch_unchecked::<
                K,
                T,
                KeyEq,
                Op,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                unsafe { BufferArg::from_raw_parts(keys.handle.clone(), keys.len()) },
                unsafe { BufferArg::from_raw_parts(local_inclusive_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), scan_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes.clone(), scan_blocks) },
                unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }
    } else {
        unsafe {
            reduce_by_key_values_at_ends_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                unsafe { BufferArg::from_raw_parts(keys.handle.clone(), keys.len()) },
                unsafe { BufferArg::from_raw_parts(local_inclusive_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }
    }

    Ok(output_handle)
}
