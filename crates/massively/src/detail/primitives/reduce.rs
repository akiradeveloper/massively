use crate::{
    detail::op::kernel::BinaryOp,
    device::KernelColumnBindings,
    error::Error,
    expr::{
        DeviceGpuExpr, LogicalDeviceExpr3, LogicalDeviceExpr7, LogicalHostPack3, LogicalHostPack7,
    },
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

pub(crate) fn reduce_logical3_device_expr<R, Item, LeafA, LeafB, LeafC, Expr, Pack, Op>(
    policy: &CubePolicy<R>,
    bindings: &KernelColumnBindings,
    len: usize,
    init: Item,
) -> Result<Item, Error>
where
    R: Runtime,
    Item: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive + CubeElement,
    LeafB: CubePrimitive + CubeElement,
    LeafC: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr3<Item, LeafA, LeafB, LeafC>,
    Pack: crate::expr::LogicalDevicePack3<Item, LeafA, LeafB, LeafC>
        + LogicalHostPack3<Item, LeafA, LeafB, LeafC>
        + 'static
        + Send
        + Sync,
    Op: BinaryOp<Item>,
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
    let partial_a = workspace.alloc::<LeafA>(partial_len);
    let partial_b = workspace.alloc::<LeafB>(partial_len);
    let partial_c = workspace.alloc::<LeafC>(partial_len);
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let offsets = bindings.slot_offsets_handle(client)?;

    unsafe {
        logical3_reduce_expr_partials_kernel::launch_unchecked::<
            Item,
            LeafA,
            LeafB,
            LeafC,
            Expr,
            Pack,
            Op,
            R,
        >(
            client,
            CubeCount::Static(partial_len_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(offsets.clone(), 4),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(partial_a.clone(), partial_len),
            BufferArg::from_raw_parts(partial_b.clone(), partial_len),
            BufferArg::from_raw_parts(partial_c.clone(), partial_len),
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
        let next_a = workspace.alloc::<LeafA>(next_len);
        let next_b = workspace.alloc::<LeafB>(next_len);
        let next_c = workspace.alloc::<LeafC>(next_len);
        unsafe {
            logical3_reduce_partials_kernel::launch_unchecked::<
                Item,
                LeafA,
                LeafB,
                LeafC,
                Pack,
                Op,
                R,
            >(
                client,
                CubeCount::Static(next_len_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                BufferArg::from_raw_parts(current_a.clone(), current_len),
                BufferArg::from_raw_parts(current_b.clone(), current_len),
                BufferArg::from_raw_parts(current_c.clone(), current_len),
                BufferArg::from_raw_parts(current_len_handle.clone(), 1),
                BufferArg::from_raw_parts(next_a.clone(), next_len),
                BufferArg::from_raw_parts(next_b.clone(), next_len),
                BufferArg::from_raw_parts(next_c.clone(), next_len),
            );
        }
        current_a = next_a;
        current_b = next_b;
        current_c = next_c;
        current_len = next_len;
    }

    let (init_a, init_b, init_c) = Pack::leaves_host(init);
    let init_a = client.create_from_slice(LeafA::as_bytes(&[init_a]));
    let init_b = client.create_from_slice(LeafB::as_bytes(&[init_b]));
    let init_c = client.create_from_slice(LeafC::as_bytes(&[init_c]));
    let output_a = client.empty(std::mem::size_of::<LeafA>());
    let output_b = client.empty(std::mem::size_of::<LeafB>());
    let output_c = client.empty(std::mem::size_of::<LeafC>());
    unsafe {
        logical3_reduce_finalize_kernel::launch_unchecked::<Item, LeafA, LeafB, LeafC, Pack, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(current_a.clone(), 1),
            BufferArg::from_raw_parts(current_b.clone(), 1),
            BufferArg::from_raw_parts(current_c.clone(), 1),
            BufferArg::from_raw_parts(init_a.clone(), 1),
            BufferArg::from_raw_parts(init_b.clone(), 1),
            BufferArg::from_raw_parts(init_c.clone(), 1),
            BufferArg::from_raw_parts(output_a.clone(), 1),
            BufferArg::from_raw_parts(output_b.clone(), 1),
            BufferArg::from_raw_parts(output_c.clone(), 1),
        );
    }

    Ok(Pack::pack_host(
        read_one(policy, output_a)?,
        read_one(policy, output_b)?,
        read_one(policy, output_c)?,
    ))
}

pub(crate) fn reduce_logical7_device_expr<
    R,
    Item,
    Leaf0,
    Leaf1,
    Leaf2,
    Leaf3,
    Leaf4,
    Leaf5,
    Leaf6,
    Expr,
    Pack,
    Op,
>(
    policy: &CubePolicy<R>,
    bindings: &KernelColumnBindings,
    len: usize,
    init: Item,
) -> Result<Item, Error>
where
    R: Runtime,
    Item: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive + CubeElement,
    Leaf1: CubePrimitive + CubeElement,
    Leaf2: CubePrimitive + CubeElement,
    Leaf3: CubePrimitive + CubeElement,
    Leaf4: CubePrimitive + CubeElement,
    Leaf5: CubePrimitive + CubeElement,
    Leaf6: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    Pack: crate::expr::LogicalDevicePack7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
        + LogicalHostPack7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
        + 'static
        + Send
        + Sync,
    Op: BinaryOp<Item>,
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
    let partial0 = workspace.alloc::<Leaf0>(partial_len);
    let partial1 = workspace.alloc::<Leaf1>(partial_len);
    let partial2 = workspace.alloc::<Leaf2>(partial_len);
    let partial3 = workspace.alloc::<Leaf3>(partial_len);
    let partial4 = workspace.alloc::<Leaf4>(partial_len);
    let partial5 = workspace.alloc::<Leaf5>(partial_len);
    let partial6 = workspace.alloc::<Leaf6>(partial_len);
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot4 = bindings.slot_or_first(4);
    let slot5 = bindings.slot_or_first(5);
    let slot6 = bindings.slot_or_first(6);
    let offsets = bindings.slot_offsets7_handle(client)?;

    unsafe {
        logical7_reduce_expr_partials_kernel::launch_unchecked::<
            Item,
            Leaf0,
            Leaf1,
            Leaf2,
            Leaf3,
            Leaf4,
            Leaf5,
            Leaf6,
            Expr,
            Pack,
            Op,
            R,
        >(
            client,
            CubeCount::Static(partial_len_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
            BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
            BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
            BufferArg::from_raw_parts(offsets.clone(), 7),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(partial0.clone(), partial_len),
            BufferArg::from_raw_parts(partial1.clone(), partial_len),
            BufferArg::from_raw_parts(partial2.clone(), partial_len),
            BufferArg::from_raw_parts(partial3.clone(), partial_len),
            BufferArg::from_raw_parts(partial4.clone(), partial_len),
            BufferArg::from_raw_parts(partial5.clone(), partial_len),
            BufferArg::from_raw_parts(partial6.clone(), partial_len),
        );
    }

    let mut current0 = partial0;
    let mut current1 = partial1;
    let mut current2 = partial2;
    let mut current3 = partial3;
    let mut current4 = partial4;
    let mut current5 = partial5;
    let mut current6 = partial6;
    let mut current_len = partial_len;
    while current_len > 1 {
        let next_len = current_len.div_ceil(BLOCK_REDUCE_SIZE as usize);
        let next_len_u32 =
            u32::try_from(next_len).map_err(|_| Error::LengthTooLarge { len: next_len })?;
        let current_len_u32 =
            u32::try_from(current_len).map_err(|_| Error::LengthTooLarge { len: current_len })?;
        let current_len_handle = client.create_from_slice(u32::as_bytes(&[current_len_u32]));
        let next0 = workspace.alloc::<Leaf0>(next_len);
        let next1 = workspace.alloc::<Leaf1>(next_len);
        let next2 = workspace.alloc::<Leaf2>(next_len);
        let next3 = workspace.alloc::<Leaf3>(next_len);
        let next4 = workspace.alloc::<Leaf4>(next_len);
        let next5 = workspace.alloc::<Leaf5>(next_len);
        let next6 = workspace.alloc::<Leaf6>(next_len);
        unsafe {
            logical7_reduce_partials_kernel::launch_unchecked::<
                Item,
                Leaf0,
                Leaf1,
                Leaf2,
                Leaf3,
                Leaf4,
                Leaf5,
                Leaf6,
                Pack,
                Op,
                R,
            >(
                client,
                CubeCount::Static(next_len_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                BufferArg::from_raw_parts(current0.clone(), current_len),
                BufferArg::from_raw_parts(current1.clone(), current_len),
                BufferArg::from_raw_parts(current2.clone(), current_len),
                BufferArg::from_raw_parts(current3.clone(), current_len),
                BufferArg::from_raw_parts(current4.clone(), current_len),
                BufferArg::from_raw_parts(current5.clone(), current_len),
                BufferArg::from_raw_parts(current6.clone(), current_len),
                BufferArg::from_raw_parts(current_len_handle.clone(), 1),
                BufferArg::from_raw_parts(next0.clone(), next_len),
                BufferArg::from_raw_parts(next1.clone(), next_len),
                BufferArg::from_raw_parts(next2.clone(), next_len),
                BufferArg::from_raw_parts(next3.clone(), next_len),
                BufferArg::from_raw_parts(next4.clone(), next_len),
                BufferArg::from_raw_parts(next5.clone(), next_len),
                BufferArg::from_raw_parts(next6.clone(), next_len),
            );
        }
        current0 = next0;
        current1 = next1;
        current2 = next2;
        current3 = next3;
        current4 = next4;
        current5 = next5;
        current6 = next6;
        current_len = next_len;
    }

    let (init0, init1, init2, init3, init4, init5, init6) = Pack::leaves_host(init);
    let init0 = client.create_from_slice(Leaf0::as_bytes(&[init0]));
    let init1 = client.create_from_slice(Leaf1::as_bytes(&[init1]));
    let init2 = client.create_from_slice(Leaf2::as_bytes(&[init2]));
    let init3 = client.create_from_slice(Leaf3::as_bytes(&[init3]));
    let init4 = client.create_from_slice(Leaf4::as_bytes(&[init4]));
    let init5 = client.create_from_slice(Leaf5::as_bytes(&[init5]));
    let init6 = client.create_from_slice(Leaf6::as_bytes(&[init6]));
    let output0 = client.empty(std::mem::size_of::<Leaf0>());
    let output1 = client.empty(std::mem::size_of::<Leaf1>());
    let output2 = client.empty(std::mem::size_of::<Leaf2>());
    let output3 = client.empty(std::mem::size_of::<Leaf3>());
    let output4 = client.empty(std::mem::size_of::<Leaf4>());
    let output5 = client.empty(std::mem::size_of::<Leaf5>());
    let output6 = client.empty(std::mem::size_of::<Leaf6>());
    unsafe {
        logical7_reduce_finalize_kernel::launch_unchecked::<
            Item,
            Leaf0,
            Leaf1,
            Leaf2,
            Leaf3,
            Leaf4,
            Leaf5,
            Leaf6,
            Pack,
            Op,
            R,
        >(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(current0.clone(), 1),
            BufferArg::from_raw_parts(current1.clone(), 1),
            BufferArg::from_raw_parts(current2.clone(), 1),
            BufferArg::from_raw_parts(current3.clone(), 1),
            BufferArg::from_raw_parts(current4.clone(), 1),
            BufferArg::from_raw_parts(current5.clone(), 1),
            BufferArg::from_raw_parts(current6.clone(), 1),
            BufferArg::from_raw_parts(init0.clone(), 1),
            BufferArg::from_raw_parts(init1.clone(), 1),
            BufferArg::from_raw_parts(init2.clone(), 1),
            BufferArg::from_raw_parts(init3.clone(), 1),
            BufferArg::from_raw_parts(init4.clone(), 1),
            BufferArg::from_raw_parts(init5.clone(), 1),
            BufferArg::from_raw_parts(init6.clone(), 1),
            BufferArg::from_raw_parts(output0.clone(), 1),
            BufferArg::from_raw_parts(output1.clone(), 1),
            BufferArg::from_raw_parts(output2.clone(), 1),
            BufferArg::from_raw_parts(output3.clone(), 1),
            BufferArg::from_raw_parts(output4.clone(), 1),
            BufferArg::from_raw_parts(output5.clone(), 1),
            BufferArg::from_raw_parts(output6.clone(), 1),
        );
    }

    Ok(Pack::pack_host(
        read_one(policy, output0)?,
        read_one(policy, output1)?,
        read_one(policy, output2)?,
        read_one(policy, output3)?,
        read_one(policy, output4)?,
        read_one(policy, output5)?,
        read_one(policy, output6)?,
    ))
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn reduce_tuple7_device_expr<
    R,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    ExprA,
    ExprB,
    ExprC,
    ExprD,
    ExprE,
    ExprF,
    ExprG,
    Op,
>(
    policy: &CubePolicy<R>,
    a: &KernelColumnBindings,
    b: &KernelColumnBindings,
    c: &KernelColumnBindings,
    d: &KernelColumnBindings,
    e: &KernelColumnBindings,
    f: &KernelColumnBindings,
    g: &KernelColumnBindings,
    len: usize,
    init: (A, B, C, D, E, F, G),
) -> Result<(A, B, C, D, E, F, G), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    ExprD: DeviceGpuExpr<D>,
    ExprE: DeviceGpuExpr<E>,
    ExprF: DeviceGpuExpr<F>,
    ExprG: DeviceGpuExpr<G>,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
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
    let partial_d = workspace.alloc::<D>(partial_len);
    let partial_e = workspace.alloc::<E>(partial_len);
    let partial_f = workspace.alloc::<F>(partial_len);
    let partial_g = workspace.alloc::<G>(partial_len);
    let a_offsets = a.slot_offsets_handle(client)?;
    let b_offsets = b.slot_offsets_handle(client)?;
    let c_offsets = c.slot_offsets_handle(client)?;
    let d_offsets = d.slot_offsets_handle(client)?;
    let e_offsets = e.slot_offsets_handle(client)?;
    let f_offsets = f.slot_offsets_handle(client)?;
    let g_offsets = g.slot_offsets_handle(client)?;
    let a0 = a.slot_or_first(0);
    let a1 = a.slot_or_first(1);
    let a2 = a.slot_or_first(2);
    let a3 = a.slot_or_first(3);
    let b0 = b.slot_or_first(0);
    let b1 = b.slot_or_first(1);
    let b2 = b.slot_or_first(2);
    let b3 = b.slot_or_first(3);
    let c0 = c.slot_or_first(0);
    let c1 = c.slot_or_first(1);
    let c2 = c.slot_or_first(2);
    let c3 = c.slot_or_first(3);
    let d0 = d.slot_or_first(0);
    let d1 = d.slot_or_first(1);
    let d2 = d.slot_or_first(2);
    let d3 = d.slot_or_first(3);
    let e0 = e.slot_or_first(0);
    let e1 = e.slot_or_first(1);
    let e2 = e.slot_or_first(2);
    let e3 = e.slot_or_first(3);
    let f0 = f.slot_or_first(0);
    let f1 = f.slot_or_first(1);
    let f2 = f.slot_or_first(2);
    let f3 = f.slot_or_first(3);
    let g0 = g.slot_or_first(0);
    let g1 = g.slot_or_first(1);
    let g2 = g.slot_or_first(2);
    let g3 = g.slot_or_first(3);

    unsafe {
        tuple7_device_reduce_expr_partials_kernel::launch_unchecked::<
            A,
            B,
            C,
            D,
            E,
            F,
            G,
            ExprA,
            ExprB,
            ExprC,
            ExprD,
            ExprE,
            ExprF,
            ExprG,
            Op,
            R,
        >(
            client,
            CubeCount::Static(partial_len_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
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
            BufferArg::from_raw_parts(d0.0.clone(), d0.1),
            BufferArg::from_raw_parts(d1.0.clone(), d1.1),
            BufferArg::from_raw_parts(d2.0.clone(), d2.1),
            BufferArg::from_raw_parts(d3.0.clone(), d3.1),
            BufferArg::from_raw_parts(d_offsets.clone(), 4),
            BufferArg::from_raw_parts(e0.0.clone(), e0.1),
            BufferArg::from_raw_parts(e1.0.clone(), e1.1),
            BufferArg::from_raw_parts(e2.0.clone(), e2.1),
            BufferArg::from_raw_parts(e3.0.clone(), e3.1),
            BufferArg::from_raw_parts(e_offsets.clone(), 4),
            BufferArg::from_raw_parts(f0.0.clone(), f0.1),
            BufferArg::from_raw_parts(f1.0.clone(), f1.1),
            BufferArg::from_raw_parts(f2.0.clone(), f2.1),
            BufferArg::from_raw_parts(f3.0.clone(), f3.1),
            BufferArg::from_raw_parts(f_offsets.clone(), 4),
            BufferArg::from_raw_parts(g0.0.clone(), g0.1),
            BufferArg::from_raw_parts(g1.0.clone(), g1.1),
            BufferArg::from_raw_parts(g2.0.clone(), g2.1),
            BufferArg::from_raw_parts(g3.0.clone(), g3.1),
            BufferArg::from_raw_parts(g_offsets.clone(), 4),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(partial_a.clone(), partial_len),
            BufferArg::from_raw_parts(partial_b.clone(), partial_len),
            BufferArg::from_raw_parts(partial_c.clone(), partial_len),
            BufferArg::from_raw_parts(partial_d.clone(), partial_len),
            BufferArg::from_raw_parts(partial_e.clone(), partial_len),
            BufferArg::from_raw_parts(partial_f.clone(), partial_len),
            BufferArg::from_raw_parts(partial_g.clone(), partial_len),
        );
    }

    let mut current_a = partial_a;
    let mut current_b = partial_b;
    let mut current_c = partial_c;
    let mut current_d = partial_d;
    let mut current_e = partial_e;
    let mut current_f = partial_f;
    let mut current_g = partial_g;
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
        let next_d = workspace.alloc::<D>(next_len);
        let next_e = workspace.alloc::<E>(next_len);
        let next_f = workspace.alloc::<F>(next_len);
        let next_g = workspace.alloc::<G>(next_len);
        unsafe {
            tuple7_reduce_partials_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
                client,
                CubeCount::Static(next_len_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                BufferArg::from_raw_parts(current_a.clone(), current_len),
                BufferArg::from_raw_parts(current_b.clone(), current_len),
                BufferArg::from_raw_parts(current_c.clone(), current_len),
                BufferArg::from_raw_parts(current_d.clone(), current_len),
                BufferArg::from_raw_parts(current_e.clone(), current_len),
                BufferArg::from_raw_parts(current_f.clone(), current_len),
                BufferArg::from_raw_parts(current_g.clone(), current_len),
                BufferArg::from_raw_parts(current_len_handle.clone(), 1),
                BufferArg::from_raw_parts(next_a.clone(), next_len),
                BufferArg::from_raw_parts(next_b.clone(), next_len),
                BufferArg::from_raw_parts(next_c.clone(), next_len),
                BufferArg::from_raw_parts(next_d.clone(), next_len),
                BufferArg::from_raw_parts(next_e.clone(), next_len),
                BufferArg::from_raw_parts(next_f.clone(), next_len),
                BufferArg::from_raw_parts(next_g.clone(), next_len),
            );
        }
        current_a = next_a;
        current_b = next_b;
        current_c = next_c;
        current_d = next_d;
        current_e = next_e;
        current_f = next_f;
        current_g = next_g;
        current_len = next_len;
    }

    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(C::as_bytes(&[init.2]));
    let init_d = client.create_from_slice(D::as_bytes(&[init.3]));
    let init_e = client.create_from_slice(E::as_bytes(&[init.4]));
    let init_f = client.create_from_slice(F::as_bytes(&[init.5]));
    let init_g = client.create_from_slice(G::as_bytes(&[init.6]));
    let output_a = client.empty(std::mem::size_of::<A>());
    let output_b = client.empty(std::mem::size_of::<B>());
    let output_c = client.empty(std::mem::size_of::<C>());
    let output_d = client.empty(std::mem::size_of::<D>());
    let output_e = client.empty(std::mem::size_of::<E>());
    let output_f = client.empty(std::mem::size_of::<F>());
    let output_g = client.empty(std::mem::size_of::<G>());
    unsafe {
        tuple7_reduce_finalize_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(current_a.clone(), 1),
            BufferArg::from_raw_parts(current_b.clone(), 1),
            BufferArg::from_raw_parts(current_c.clone(), 1),
            BufferArg::from_raw_parts(current_d.clone(), 1),
            BufferArg::from_raw_parts(current_e.clone(), 1),
            BufferArg::from_raw_parts(current_f.clone(), 1),
            BufferArg::from_raw_parts(current_g.clone(), 1),
            BufferArg::from_raw_parts(init_a.clone(), 1),
            BufferArg::from_raw_parts(init_b.clone(), 1),
            BufferArg::from_raw_parts(init_c.clone(), 1),
            BufferArg::from_raw_parts(init_d.clone(), 1),
            BufferArg::from_raw_parts(init_e.clone(), 1),
            BufferArg::from_raw_parts(init_f.clone(), 1),
            BufferArg::from_raw_parts(init_g.clone(), 1),
            BufferArg::from_raw_parts(output_a.clone(), 1),
            BufferArg::from_raw_parts(output_b.clone(), 1),
            BufferArg::from_raw_parts(output_c.clone(), 1),
            BufferArg::from_raw_parts(output_d.clone(), 1),
            BufferArg::from_raw_parts(output_e.clone(), 1),
            BufferArg::from_raw_parts(output_f.clone(), 1),
            BufferArg::from_raw_parts(output_g.clone(), 1),
        );
    }

    Ok((
        read_one(policy, output_a)?,
        read_one(policy, output_b)?,
        read_one(policy, output_c)?,
        read_one(policy, output_d)?,
        read_one(policy, output_e)?,
        read_one(policy, output_f)?,
        read_one(policy, output_g)?,
    ))
}
