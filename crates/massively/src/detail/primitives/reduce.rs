use crate::{
    detail::op::kernel::BinaryOp,
    device::KernelColumnBindings,
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    policy::CubePolicy,
    primitives::{scan, workspace::Workspace},
};
use cubecl::prelude::*;

pub(crate) const BLOCK_REDUCE_SIZE: u32 = 256;

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
