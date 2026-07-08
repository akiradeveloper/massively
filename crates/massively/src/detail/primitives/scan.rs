#![allow(dead_code)]

use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp},
    device::{DeviceColumnView, DeviceVec, KernelColumnBindings, Zip1, Zip2, Zip3},
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::{range, workspace::Workspace},
};
use cubecl::prelude::*;

pub(crate) const BLOCK_SCAN_SIZE: u32 = 256;

fn binding_slot_or_first(
    bindings: &KernelColumnBindings,
    index: usize,
) -> &(cubecl::server::Handle, usize) {
    let first = bindings
        .slots
        .first()
        .expect("kernel column has at least one slot");
    bindings.slots.get(index).unwrap_or(first)
}

pub(crate) fn inclusive_scan_u32<R: Runtime>(
    client: &ComputeClient<R>,
    input_handle: &cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error> {
    if len == 0 {
        return Ok(crate::policy::empty_handle(client));
    }

    let output_handle = client.empty(len * std::mem::size_of::<u32>());
    let workspace = Workspace::from_client(client);
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_SCAN_SIZE)?;
    let num_blocks = launch.logical_blocks;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let block_sums_handle = workspace.alloc::<u32>(num_blocks);

    unsafe {
        u32_block_inclusive_scan_kernel::launch_unchecked::<R>(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(input_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums_handle.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let block_prefixes_handle =
            inclusive_scan_u32::<R>(client, &block_sums_handle, num_blocks)?;
        unsafe {
            u32_add_block_prefix_kernel::launch_unchecked::<R>(
                client,
                launch.cube_count(),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_handle.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }
    }

    Ok(output_handle)
}

pub(crate) fn adjacent_difference_tuple2_device_expr<R, A, B, ExprA, ExprB, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<Zip2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp<(A, B)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok(Zip2 {
            left: policy.empty_device_vec(),
            right: policy.empty_device_vec(),
        });
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let a_slot0 = binding_slot_or_first(a_bindings, 0);
    let a_slot1 = binding_slot_or_first(a_bindings, 1);
    let a_slot2 = binding_slot_or_first(a_bindings, 2);
    let a_slot3 = binding_slot_or_first(a_bindings, 3);
    let b_slot0 = binding_slot_or_first(b_bindings, 0);
    let b_slot1 = binding_slot_or_first(b_bindings, 1);
    let b_slot2 = binding_slot_or_first(b_bindings, 2);
    let b_slot3 = binding_slot_or_first(b_bindings, 3);
    let a_slot_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_slot_offsets = b_bindings.slot_offsets_handle(client)?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_SCAN_SIZE)?;

    unsafe {
        tuple2_adjacent_difference_expr_kernel::launch_unchecked::<A, B, ExprA, ExprB, Op, R>(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(a_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
        );
    }

    Ok(Zip2 {
        left: DeviceVec::from_handle(policy.id(), output_a, len),
        right: DeviceVec::from_handle(policy.id(), output_b, len),
    })
}

pub(crate) fn adjacent_difference_tuple3_device_expr<R, A, B, C, ExprA, ExprB, ExprC, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<Zip3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
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
    let client = policy.client();
    if len == 0 {
        return Ok(Zip3 {
            first: policy.empty_device_vec(),
            second: policy.empty_device_vec(),
            third: policy.empty_device_vec(),
        });
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let a_slot0 = binding_slot_or_first(a_bindings, 0);
    let a_slot1 = binding_slot_or_first(a_bindings, 1);
    let a_slot2 = binding_slot_or_first(a_bindings, 2);
    let a_slot3 = binding_slot_or_first(a_bindings, 3);
    let b_slot0 = binding_slot_or_first(b_bindings, 0);
    let b_slot1 = binding_slot_or_first(b_bindings, 1);
    let b_slot2 = binding_slot_or_first(b_bindings, 2);
    let b_slot3 = binding_slot_or_first(b_bindings, 3);
    let c_slot0 = binding_slot_or_first(c_bindings, 0);
    let c_slot1 = binding_slot_or_first(c_bindings, 1);
    let c_slot2 = binding_slot_or_first(c_bindings, 2);
    let c_slot3 = binding_slot_or_first(c_bindings, 3);
    let a_slot_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_slot_offsets = b_bindings.slot_offsets_handle(client)?;
    let c_slot_offsets = c_bindings.slot_offsets_handle(client)?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_SCAN_SIZE)?;

    unsafe {
        tuple3_adjacent_difference_expr_kernel::launch_unchecked::<
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
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(a_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(c_slot0.0.clone(), c_slot0.1) },
            unsafe { BufferArg::from_raw_parts(c_slot1.0.clone(), c_slot1.1) },
            unsafe { BufferArg::from_raw_parts(c_slot2.0.clone(), c_slot2.1) },
            unsafe { BufferArg::from_raw_parts(c_slot3.0.clone(), c_slot3.1) },
            unsafe { BufferArg::from_raw_parts(c_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
        );
    }

    Ok(Zip3 {
        first: DeviceVec::from_handle(policy.id(), output_a, len),
        second: DeviceVec::from_handle(policy.id(), output_b, len),
        third: DeviceVec::from_handle(policy.id(), output_c, len),
    })
}

pub(crate) fn inclusive_scan_by_key_device_expr<R, K, T, KeyExpr, ValueExpr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    value_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ValueExpr: DeviceGpuExpr<T>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let output_handle =
        inclusive_scan_by_key_device_expr_handle::<R, K, T, KeyExpr, ValueExpr, KeyEq, Op>(
            policy,
            key_bindings,
            value_bindings,
            len,
        )?;
    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn exclusive_scan_by_key_device_expr<R, K, T, KeyExpr, ValueExpr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    value_bindings: &KernelColumnBindings,
    len: usize,
    init: T,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ValueExpr: DeviceGpuExpr<T>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let client = policy.client();
    let inclusive_handle =
        inclusive_scan_by_key_device_expr_handle::<R, K, T, KeyExpr, ValueExpr, KeyEq, Op>(
            policy,
            key_bindings,
            value_bindings,
            len,
        )?;
    let output_handle = make_scan_by_key_device_expr_exclusive::<R, K, T, KeyExpr, KeyEq, Op>(
        client,
        key_bindings,
        len,
        &inclusive_handle,
        init,
    )?;
    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn inclusive_scan_tuple2_by_key_values_device_expr<
    R,
    K,
    A,
    B,
    KeyExpr,
    ExprA,
    ExprB,
    KeyEq,
    Op,
>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<Zip2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
{
    let (left, right) = inclusive_scan_tuple2_by_key_values_device_expr_handle::<
        R,
        K,
        A,
        B,
        KeyExpr,
        ExprA,
        ExprB,
        KeyEq,
        Op,
    >(policy, key_bindings, a_bindings, b_bindings, len)?;
    Ok(Zip2 {
        left: DeviceVec::from_handle(policy.id(), left, len),
        right: DeviceVec::from_handle(policy.id(), right, len),
    })
}

pub(crate) fn exclusive_scan_tuple2_by_key_values_device_expr<
    R,
    K,
    A,
    B,
    KeyExpr,
    ExprA,
    ExprB,
    KeyEq,
    Op,
>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    len: usize,
    init: (A, B),
) -> Result<Zip2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
{
    let inclusive = inclusive_scan_tuple2_by_key_values_device_expr_handle::<
        R,
        K,
        A,
        B,
        KeyExpr,
        ExprA,
        ExprB,
        KeyEq,
        Op,
    >(policy, key_bindings, a_bindings, b_bindings, len)?;
    let (left, right) =
        make_scan_tuple2_by_key_values_device_expr_exclusive::<R, K, A, B, KeyExpr, KeyEq, Op>(
            policy,
            key_bindings,
            len,
            inclusive,
            init,
        )?;
    Ok(Zip2 {
        left: DeviceVec::from_handle(policy.id(), left, len),
        right: DeviceVec::from_handle(policy.id(), right, len),
    })
}

pub(crate) fn inclusive_scan_tuple3_by_key_values_device_expr<
    R,
    K,
    A,
    B,
    C,
    KeyExpr,
    ExprA,
    ExprB,
    ExprC,
    KeyEq,
    Op,
>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<Zip3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
{
    let (first, second, third) = inclusive_scan_tuple3_by_key_values_device_expr_handle::<
        R,
        K,
        A,
        B,
        C,
        KeyExpr,
        ExprA,
        ExprB,
        ExprC,
        KeyEq,
        Op,
    >(
        policy,
        key_bindings,
        a_bindings,
        b_bindings,
        c_bindings,
        len,
    )?;
    Ok(Zip3 {
        first: DeviceVec::from_handle(policy.id(), first, len),
        second: DeviceVec::from_handle(policy.id(), second, len),
        third: DeviceVec::from_handle(policy.id(), third, len),
    })
}

pub(crate) fn exclusive_scan_tuple3_by_key_values_device_expr<
    R,
    K,
    A,
    B,
    C,
    KeyExpr,
    ExprA,
    ExprB,
    ExprC,
    KeyEq,
    Op,
>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
    init: (A, B, C),
) -> Result<Zip3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
{
    let inclusive = inclusive_scan_tuple3_by_key_values_device_expr_handle::<
        R,
        K,
        A,
        B,
        C,
        KeyExpr,
        ExprA,
        ExprB,
        ExprC,
        KeyEq,
        Op,
    >(
        policy,
        key_bindings,
        a_bindings,
        b_bindings,
        c_bindings,
        len,
    )?;
    let (first, second, third) =
        make_scan_tuple3_by_key_values_device_expr_exclusive::<R, K, A, B, C, KeyExpr, KeyEq, Op>(
            policy,
            key_bindings,
            len,
            inclusive,
            init,
        )?;
    Ok(Zip3 {
        first: DeviceVec::from_handle(policy.id(), first, len),
        second: DeviceVec::from_handle(policy.id(), second, len),
        third: DeviceVec::from_handle(policy.id(), third, len),
    })
}

macro_rules! define_scan_tuple_value_by_key_handle {
    (
        $handle_fn:ident,
        $handles:ty,
        $block_kernel:ident,
        $add_prefix_kernel:ident,
        ( $( $ty:ident: $value:ident: $out:ident: $tail:ident: $tail_vec:ident: $prefix:ident ),+ )
    ) => {
        fn $handle_fn<R, K, $( $ty ),+, KeyEq, Op>(
            policy: &CubePolicy<R>,
            keys: &DeviceVec<R, K>,
            $( $value: &cubecl::server::Handle, )+
        ) -> Result<$handles, Error>
        where
            R: Runtime,
            K: CubePrimitive + CubeElement,
            $( $ty: CubePrimitive + CubeElement, )+
            KeyEq: BinaryPredicateOp<K>,
            Op: BinaryOp<($( $ty ),+)>,
        {
            let len = keys.len();
            let client = policy.client();
            if len == 0 {
                return Ok(($( {
                    let _ = core::mem::size_of::<$ty>();
                    policy.empty_handle()
                }, )+));
            }
            if len == 1 {
                return Ok(($( range::copy_handle::<R, $ty>(policy, $value, len)?, )+));
            }

            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
            let workspace = Workspace::new(policy);
            $(
                let $out = workspace.alloc::<$ty>(len);
                let $tail = workspace.alloc::<$ty>(num_blocks);
            )+
            let block_tail_keys = workspace.alloc::<K>(num_blocks);

            unsafe {
                $block_kernel::launch_unchecked::<K, $( $ty, )+ KeyEq, Op, R>(
                    client,
                    CubeCount::Static(num_blocks_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SCAN_SIZE),
                    unsafe { BufferArg::from_raw_parts(keys.handle.clone(), len) },
                    $(
                        unsafe { BufferArg::from_raw_parts($value.clone(), len) },
                    )+
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    $(
                        unsafe { BufferArg::from_raw_parts($out.clone(), len) },
                    )+
                    unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
                    $(
                        unsafe { BufferArg::from_raw_parts($tail.clone(), num_blocks) },
                    )+
                );
            }

            if num_blocks > 1 {
                let tail_keys = DeviceVec::from_handle(policy.id(), block_tail_keys.clone(), num_blocks);
                $(
                    let $tail_vec = DeviceVec::<R, $ty>::from_handle(
                        policy.id(),
                        $tail.clone(),
                        num_blocks,
                    );
                )+
                let ($( $prefix, )+) = $handle_fn::<R, K, $( $ty, )+ KeyEq, Op>(
                    policy,
                    &tail_keys,
                    $( &$tail_vec.handle, )+
                )?;
                unsafe {
                    $add_prefix_kernel::launch_unchecked::<K, $( $ty, )+ KeyEq, Op, R>(
                        client,
                        CubeCount::Static(num_blocks_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SCAN_SIZE),
                        unsafe { BufferArg::from_raw_parts(keys.handle.clone(), len) },
                        unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
                        $(
                            unsafe { BufferArg::from_raw_parts($prefix.clone(), num_blocks) },
                        )+
                        unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                        $(
                            unsafe { BufferArg::from_raw_parts($out.clone(), len) },
                        )+
                    );
                }
            }

            Ok(($( $out, )+))
        }
    };
}

define_scan_tuple_value_by_key_handle!(
    inclusive_scan_tuple2_by_key_values_handle,
    (cubecl::server::Handle, cubecl::server::Handle),
    scan_by_key_tuple2_block_kernel,
    scan_by_key_tuple2_add_block_prefix_kernel,
    (A: left: output_a: block_tail_a: block_tail_a_vec: prefix_a, B: right: output_b: block_tail_b: block_tail_b_vec: prefix_b)
);
define_scan_tuple_value_by_key_handle!(
    inclusive_scan_tuple3_by_key_values_handle,
    (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ),
    scan_by_key_tuple3_block_kernel,
    scan_by_key_tuple3_add_block_prefix_kernel,
    (A: first: output_a: block_tail_a: block_tail_a_vec: prefix_a, B: second: output_b: block_tail_b: block_tail_b_vec: prefix_b, C: third: output_c: block_tail_c: block_tail_c_vec: prefix_c)
);

macro_rules! define_scan_tuple_by_key_device_vec {
    (
        $inclusive_fn:ident,
        $exclusive_fn:ident,
        $handle_fn:ident,
        $exclusive_handle_fn:ident,
        $block_kernel:ident,
        $add_prefix_kernel:ident,
        $exclusive_kernel:ident,
        ( $( $ty:ident: $key:ident: $tail_handle:ident: $tail_vec:ident ),+ )
    ) => {
        #[allow(dead_code)]
        pub(crate) fn $inclusive_fn<R, $( $ty ),+, T, KeyEq, Op>(
            policy: &CubePolicy<R>,
            $( $key: &DeviceVec<R, $ty>, )+
            values: &DeviceVec<R, T>,
            _key_eq: GpuOp<KeyEq>,
            _op: GpuOp<Op>,
        ) -> Result<DeviceVec<R, T>, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let len = values.len();
            $(
                super::ensure_same_len($key.len(), len)?;
            )+

            let output_handle = $handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                policy,
                $( $key, )+
                &values.handle,
            )?;
            Ok(DeviceVec::from_handle(policy.id(),
                output_handle,
                values.len(),
            ))
        }

        #[allow(dead_code)]
        pub(crate) fn $exclusive_fn<R, $( $ty ),+, T, KeyEq, Op>(
            policy: &CubePolicy<R>,
            $( $key: &DeviceVec<R, $ty>, )+
            values: &DeviceVec<R, T>,
            init: T,
            _key_eq: GpuOp<KeyEq>,
            _op: GpuOp<Op>,
        ) -> Result<DeviceVec<R, T>, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let len = values.len();
            $(
                super::ensure_same_len($key.len(), len)?;
            )+

            let client = policy.client();
            let inclusive_handle = $handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                policy,
                $( $key, )+
                &values.handle,
            )?;
            let output_handle = $exclusive_handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                client,
                $( $key, )+
                &inclusive_handle,
                init,
            )?;
            Ok(DeviceVec::from_handle(policy.id(),
                output_handle,
                values.len(),
            ))
        }

        pub(crate) fn $handle_fn<R, $( $ty ),+, T, KeyEq, Op>(
            policy: &CubePolicy<R>,
            $( $key: &DeviceVec<R, $ty>, )+
            value_handle: &cubecl::server::Handle,
        ) -> Result<cubecl::server::Handle, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let mut len = None;
            $(
                let key_len = $key.len();
                if let Some(expected) = len {
                    super::ensure_same_len(key_len, expected)?;
                } else {
                    len = Some(key_len);
                }
            )+
            let len = len.unwrap_or(0);

            let client = policy.client();
            if len == 0 {
                return Ok(policy.empty_handle());
            }
            if len == 1 {
                return range::copy_handle::<R, T>(policy, value_handle, len);
            }

            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
            let workspace = Workspace::new(policy);
            let output_handle = workspace.alloc::<T>(len);
            $(
                let $tail_handle = workspace.alloc::<$ty>(num_blocks);
            )+
            let block_tail_values = workspace.alloc::<T>(num_blocks);

            unsafe {
                $block_kernel::launch_unchecked::<$( $ty, )+ T, KeyEq, Op, R>(
                    client,
                    CubeCount::Static(num_blocks_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SCAN_SIZE),
                    $(
                        unsafe { BufferArg::from_raw_parts($key.handle.clone(), len) },
                    )+
                    unsafe { BufferArg::from_raw_parts(value_handle.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
                    $(
                        unsafe { BufferArg::from_raw_parts($tail_handle.clone(), num_blocks) },
                    )+
                    unsafe { BufferArg::from_raw_parts(block_tail_values.clone(), num_blocks) },
                );
            }

            if num_blocks > 1 {
                $(
                    let $tail_vec = DeviceVec::from_handle(policy.id(),
                        $tail_handle.clone(),
                        num_blocks,
                    );
                )+
                let block_prefixes = $handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                    policy,
                    $( &$tail_vec, )+
                    &block_tail_values,
                )?;
                unsafe {
                    $add_prefix_kernel::launch_unchecked::<$( $ty, )+ T, KeyEq, Op, R>(
                        client,
                        CubeCount::Static(num_blocks_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SCAN_SIZE),
                        $(
                            unsafe { BufferArg::from_raw_parts($key.handle.clone(), len) },
                        )+
                        $(
                            unsafe { BufferArg::from_raw_parts($tail_handle.clone(), num_blocks) },
                        )+
                        unsafe { BufferArg::from_raw_parts(block_prefixes.clone(), num_blocks) },
                        unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
                    );
                }
            }

            Ok(output_handle)
        }

        #[allow(dead_code)]
        pub(crate) fn $exclusive_handle_fn<R, $( $ty ),+, T, KeyEq, Op>(
            client: &ComputeClient<R>,
            $( $key: &DeviceVec<R, $ty>, )+
            inclusive_handle: &cubecl::server::Handle,
            init: T,
        ) -> Result<cubecl::server::Handle, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let mut len = None;
            $(
                let key_len = $key.len();
                if let Some(expected) = len {
                    super::ensure_same_len(key_len, expected)?;
                } else {
                    len = Some(key_len);
                }
            )+
            let len = len.unwrap_or(0);
            if len == 0 {
                return Ok(crate::policy::empty_handle(client));
            }

            let output_handle = client.empty(len * std::mem::size_of::<T>());
            u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
            let init_handle = client.create_from_slice(T::as_bytes(&[init]));
            unsafe {
                $exclusive_kernel::launch_unchecked::<$( $ty, )+ T, KeyEq, Op, R>(
                    client,
                    CubeCount::Static(num_blocks_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SCAN_SIZE),
                    $(
                        unsafe { BufferArg::from_raw_parts($key.handle.clone(), len) },
                    )+
                    unsafe { BufferArg::from_raw_parts(inclusive_handle.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
                );
            }

            Ok(output_handle)
        }
    };
}

define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple2_by_key_device_vec,
    exclusive_scan_tuple2_by_key_device_vec,
    inclusive_scan_tuple2_by_key_handle,
    make_scan_tuple2_by_key_exclusive,
    scan_tuple2_by_key_block_kernel,
    scan_tuple2_by_key_add_block_prefix_kernel,
    scan_tuple2_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple3_by_key_device_vec,
    exclusive_scan_tuple3_by_key_device_vec,
    inclusive_scan_tuple3_by_key_handle,
    make_scan_tuple3_by_key_exclusive,
    scan_tuple3_by_key_block_kernel,
    scan_tuple3_by_key_add_block_prefix_kernel,
    scan_tuple3_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec)
);

pub(crate) fn inclusive_scan_tuple1_device_expr<R, A, ExprA, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<Zip1<DeviceVec<R, A>>, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    Op: BinaryOp<(A,)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok(Zip1 {
            source: policy.empty_device_vec(),
        });
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let workspace = Workspace::new(policy);
    let a_slot0 = a_bindings.slots.first().unwrap();
    let a_slot1 = a_bindings.slots.get(1).unwrap_or(a_slot0);
    let a_slot2 = a_bindings.slots.get(2).unwrap_or(a_slot0);
    let a_slot3 = a_bindings.slots.get(3).unwrap_or(a_slot0);
    let a_slot_offsets = a_bindings.slot_offsets_handle(client)?;
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_SCAN_SIZE)?;
    let num_blocks = launch.logical_blocks;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);

    unsafe {
        tuple1_device_inclusive_scan_expr_block_kernel::launch_unchecked::<A, ExprA, Op, R>(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(a_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let (block_prefixes_a,) =
            inclusive_scan_tuple1_handles::<R, A, Op>(policy, &block_sums_a, num_blocks)?;
        unsafe {
            tuple1_scan_add_block_prefix_kernel::launch_unchecked::<A, Op, R>(
                client,
                launch.cube_count(),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            );
        }
    }

    Ok(Zip1 {
        source: DeviceVec::from_handle(policy.id(), output_a, len),
    })
}

pub(crate) fn exclusive_scan_tuple1_device_expr<R, A, ExprA, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    len: usize,
    init: (A,),
) -> Result<Zip1<DeviceVec<R, A>>, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    Op: BinaryOp<(A,)>,
{
    let inclusive = inclusive_scan_tuple1_device_expr::<R, A, ExprA, Op>(policy, a_bindings, len)?;
    let (output_a,) =
        make_tuple1_exclusive::<R, A, Op>(policy, &inclusive.source.handle, len, init)?;
    Ok(Zip1 {
        source: DeviceVec::from_handle(policy.id(), output_a, len),
    })
}

fn inclusive_scan_tuple1_handles<R, A, Op>(
    policy: &CubePolicy<R>,
    input_a: &cubecl::server::Handle,
    len: usize,
) -> Result<(cubecl::server::Handle,), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    Op: BinaryOp<(A,)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((policy.empty_handle(),));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let workspace = Workspace::new(policy);
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_SCAN_SIZE)?;
    let num_blocks = launch.logical_blocks;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);

    unsafe {
        tuple1_inclusive_scan_block_kernel::launch_unchecked::<A, Op, R>(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(input_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let (block_prefixes_a,) =
            inclusive_scan_tuple1_handles::<R, A, Op>(policy, &block_sums_a, num_blocks)?;
        unsafe {
            tuple1_scan_add_block_prefix_kernel::launch_unchecked::<A, Op, R>(
                client,
                launch.cube_count(),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            );
        }
    }

    Ok((output_a,))
}

fn make_tuple1_exclusive<R, A, Op>(
    policy: &CubePolicy<R>,
    inclusive_a: &cubecl::server::Handle,
    len: usize,
    init: (A,),
) -> Result<(cubecl::server::Handle,), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    Op: BinaryOp<(A,)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((policy.empty_handle(),));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_SCAN_SIZE)?;
    unsafe {
        tuple1_scan_make_exclusive_kernel::launch_unchecked::<A, Op, R>(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(inclusive_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
        );
    }

    Ok((output_a,))
}

pub(crate) fn inclusive_scan_tuple2_device_expr<R, A, B, ExprA, ExprB, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<Zip2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp<(A, B)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok(Zip2 {
            left: policy.empty_device_vec(),
            right: policy.empty_device_vec(),
        });
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let workspace = Workspace::new(policy);
    let a_slot0 = a_bindings.slots.first().unwrap();
    let a_slot1 = a_bindings.slots.get(1).unwrap_or(a_slot0);
    let a_slot2 = a_bindings.slots.get(2).unwrap_or(a_slot0);
    let a_slot3 = a_bindings.slots.get(3).unwrap_or(a_slot0);
    let b_slot0 = b_bindings.slots.first().unwrap();
    let b_slot1 = b_bindings.slots.get(1).unwrap_or(b_slot0);
    let b_slot2 = b_bindings.slots.get(2).unwrap_or(b_slot0);
    let b_slot3 = b_bindings.slots.get(3).unwrap_or(b_slot0);
    let a_slot_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_slot_offsets = b_bindings.slot_offsets_handle(client)?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);
    let block_sums_b = workspace.alloc::<B>(num_blocks);

    unsafe {
        tuple2_device_inclusive_scan_expr_block_kernel::launch_unchecked::<A, B, ExprA, ExprB, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(a_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_sums_b.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let (block_prefixes_a, block_prefixes_b) = inclusive_scan_tuple2_handles::<R, A, B, Op>(
            policy,
            &block_sums_a,
            &block_sums_b,
            num_blocks,
        )?;
        unsafe {
            tuple2_scan_add_block_prefix_kernel::launch_unchecked::<A, B, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes_b.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            );
        }
    }

    Ok(Zip2 {
        left: DeviceVec::from_handle(policy.id(), output_a, len),
        right: DeviceVec::from_handle(policy.id(), output_b, len),
    })
}

pub(crate) fn exclusive_scan_tuple2_device_expr<R, A, B, ExprA, ExprB, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    len: usize,
    init: (A, B),
) -> Result<Zip2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp<(A, B)>,
{
    let inclusive = inclusive_scan_tuple2_device_expr::<R, A, B, ExprA, ExprB, Op>(
        policy, a_bindings, b_bindings, len,
    )?;
    let (output_a, output_b) = make_tuple2_exclusive::<R, A, B, Op>(
        policy,
        &inclusive.left.handle,
        &inclusive.right.handle,
        len,
        init,
    )?;
    Ok(Zip2 {
        left: DeviceVec::from_handle(policy.id(), output_a, len),
        right: DeviceVec::from_handle(policy.id(), output_b, len),
    })
}

fn inclusive_scan_tuple2_handles<R, A, B, Op>(
    policy: &CubePolicy<R>,
    input_a: &cubecl::server::Handle,
    input_b: &cubecl::server::Handle,
    len: usize,
) -> Result<(cubecl::server::Handle, cubecl::server::Handle), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((policy.empty_handle(), policy.empty_handle()));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let workspace = Workspace::new(policy);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);
    let block_sums_b = workspace.alloc::<B>(num_blocks);

    unsafe {
        tuple2_inclusive_scan_block_kernel::launch_unchecked::<A, B, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(input_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(input_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_sums_b.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let (block_prefixes_a, block_prefixes_b) = inclusive_scan_tuple2_handles::<R, A, B, Op>(
            policy,
            &block_sums_a,
            &block_sums_b,
            num_blocks,
        )?;
        unsafe {
            tuple2_scan_add_block_prefix_kernel::launch_unchecked::<A, B, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes_b.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            );
        }
    }

    Ok((output_a, output_b))
}

fn make_tuple2_exclusive<R, A, B, Op>(
    policy: &CubePolicy<R>,
    inclusive_a: &cubecl::server::Handle,
    inclusive_b: &cubecl::server::Handle,
    len: usize,
    init: (A, B),
) -> Result<(cubecl::server::Handle, cubecl::server::Handle), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((policy.empty_handle(), policy.empty_handle()));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    unsafe {
        tuple2_scan_make_exclusive_kernel::launch_unchecked::<A, B, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(inclusive_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
        );
    }

    Ok((output_a, output_b))
}

pub(crate) fn inclusive_scan_tuple3_device_expr<R, A, B, C, ExprA, ExprB, ExprC, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<Zip3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
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
    let client = policy.client();
    if len == 0 {
        return Ok(Zip3 {
            first: policy.empty_device_vec(),
            second: policy.empty_device_vec(),
            third: policy.empty_device_vec(),
        });
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let workspace = Workspace::new(policy);
    let a_slot0 = a_bindings.slots.first().unwrap();
    let a_slot1 = a_bindings.slots.get(1).unwrap_or(a_slot0);
    let a_slot2 = a_bindings.slots.get(2).unwrap_or(a_slot0);
    let a_slot3 = a_bindings.slots.get(3).unwrap_or(a_slot0);
    let b_slot0 = b_bindings.slots.first().unwrap();
    let b_slot1 = b_bindings.slots.get(1).unwrap_or(b_slot0);
    let b_slot2 = b_bindings.slots.get(2).unwrap_or(b_slot0);
    let b_slot3 = b_bindings.slots.get(3).unwrap_or(b_slot0);
    let c_slot0 = c_bindings.slots.first().unwrap();
    let c_slot1 = c_bindings.slots.get(1).unwrap_or(c_slot0);
    let c_slot2 = c_bindings.slots.get(2).unwrap_or(c_slot0);
    let c_slot3 = c_bindings.slots.get(3).unwrap_or(c_slot0);
    let a_slot_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_slot_offsets = b_bindings.slot_offsets_handle(client)?;
    let c_slot_offsets = c_bindings.slot_offsets_handle(client)?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);
    let block_sums_b = workspace.alloc::<B>(num_blocks);
    let block_sums_c = workspace.alloc::<C>(num_blocks);

    unsafe {
        tuple3_device_inclusive_scan_expr_block_kernel::launch_unchecked::<
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
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(a_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(c_slot0.0.clone(), c_slot0.1) },
            unsafe { BufferArg::from_raw_parts(c_slot1.0.clone(), c_slot1.1) },
            unsafe { BufferArg::from_raw_parts(c_slot2.0.clone(), c_slot2.1) },
            unsafe { BufferArg::from_raw_parts(c_slot3.0.clone(), c_slot3.1) },
            unsafe { BufferArg::from_raw_parts(c_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_sums_b.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_sums_c.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let (block_prefixes_a, block_prefixes_b, block_prefixes_c) =
            inclusive_scan_tuple3_handles::<R, A, B, C, Op>(
                policy,
                &block_sums_a,
                &block_sums_b,
                &block_sums_c,
                num_blocks,
            )?;
        unsafe {
            tuple3_scan_add_block_prefix_kernel::launch_unchecked::<A, B, C, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes_b.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes_c.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
            );
        }
    }

    Ok(Zip3 {
        first: DeviceVec::from_handle(policy.id(), output_a, len),
        second: DeviceVec::from_handle(policy.id(), output_b, len),
        third: DeviceVec::from_handle(policy.id(), output_c, len),
    })
}

pub(crate) fn exclusive_scan_tuple3_device_expr<R, A, B, C, ExprA, ExprB, ExprC, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
    init: (A, B, C),
) -> Result<Zip3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
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
    let inclusive = inclusive_scan_tuple3_device_expr::<R, A, B, C, ExprA, ExprB, ExprC, Op>(
        policy, a_bindings, b_bindings, c_bindings, len,
    )?;
    let (output_a, output_b, output_c) = make_tuple3_exclusive::<R, A, B, C, Op>(
        policy,
        &inclusive.first.handle,
        &inclusive.second.handle,
        &inclusive.third.handle,
        len,
        init,
    )?;
    Ok(Zip3 {
        first: DeviceVec::from_handle(policy.id(), output_a, len),
        second: DeviceVec::from_handle(policy.id(), output_b, len),
        third: DeviceVec::from_handle(policy.id(), output_c, len),
    })
}

fn inclusive_scan_tuple3_handles<R, A, B, C, Op>(
    policy: &CubePolicy<R>,
    input_a: &cubecl::server::Handle,
    input_b: &cubecl::server::Handle,
    input_c: &cubecl::server::Handle,
    len: usize,
) -> Result<
    (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ),
    Error,
>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
        ));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let workspace = Workspace::new(policy);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);
    let block_sums_b = workspace.alloc::<B>(num_blocks);
    let block_sums_c = workspace.alloc::<C>(num_blocks);

    unsafe {
        tuple3_inclusive_scan_block_kernel::launch_unchecked::<A, B, C, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(input_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(input_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(input_c.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_sums_b.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_sums_c.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let (block_prefixes_a, block_prefixes_b, block_prefixes_c) =
            inclusive_scan_tuple3_handles::<R, A, B, C, Op>(
                policy,
                &block_sums_a,
                &block_sums_b,
                &block_sums_c,
                num_blocks,
            )?;
        unsafe {
            tuple3_scan_add_block_prefix_kernel::launch_unchecked::<A, B, C, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes_b.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes_c.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
            );
        }
    }

    Ok((output_a, output_b, output_c))
}

fn make_tuple3_exclusive<R, A, B, C, Op>(
    policy: &CubePolicy<R>,
    inclusive_a: &cubecl::server::Handle,
    inclusive_b: &cubecl::server::Handle,
    inclusive_c: &cubecl::server::Handle,
    len: usize,
    init: (A, B, C),
) -> Result<
    (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ),
    Error,
>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
        ));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(C::as_bytes(&[init.2]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    unsafe {
        tuple3_scan_make_exclusive_kernel::launch_unchecked::<A, B, C, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(inclusive_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive_c.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_c.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
        );
    }

    Ok((output_a, output_b, output_c))
}

#[allow(clippy::type_complexity)]
pub(crate) fn inclusive_scan_tuple7_device_expr<
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
    let client = policy.client();
    if len == 0 {
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

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let output_d = client.empty(len * std::mem::size_of::<D>());
    let output_e = client.empty(len * std::mem::size_of::<E>());
    let output_f = client.empty(len * std::mem::size_of::<F>());
    let output_g = client.empty(len * std::mem::size_of::<G>());
    let workspace = Workspace::new(policy);
    let a0 = binding_slot_or_first(a, 0);
    let a1 = binding_slot_or_first(a, 1);
    let a2 = binding_slot_or_first(a, 2);
    let a3 = binding_slot_or_first(a, 3);
    let b0 = binding_slot_or_first(b, 0);
    let b1 = binding_slot_or_first(b, 1);
    let b2 = binding_slot_or_first(b, 2);
    let b3 = binding_slot_or_first(b, 3);
    let c0 = binding_slot_or_first(c, 0);
    let c1 = binding_slot_or_first(c, 1);
    let c2 = binding_slot_or_first(c, 2);
    let c3 = binding_slot_or_first(c, 3);
    let d0 = binding_slot_or_first(d, 0);
    let d1 = binding_slot_or_first(d, 1);
    let d2 = binding_slot_or_first(d, 2);
    let d3 = binding_slot_or_first(d, 3);
    let e0 = binding_slot_or_first(e, 0);
    let e1 = binding_slot_or_first(e, 1);
    let e2 = binding_slot_or_first(e, 2);
    let e3 = binding_slot_or_first(e, 3);
    let f0 = binding_slot_or_first(f, 0);
    let f1 = binding_slot_or_first(f, 1);
    let f2 = binding_slot_or_first(f, 2);
    let f3 = binding_slot_or_first(f, 3);
    let g0 = binding_slot_or_first(g, 0);
    let g1 = binding_slot_or_first(g, 1);
    let g2 = binding_slot_or_first(g, 2);
    let g3 = binding_slot_or_first(g, 3);
    let a_offsets = a.slot_offsets_handle(client)?;
    let b_offsets = b.slot_offsets_handle(client)?;
    let c_offsets = c.slot_offsets_handle(client)?;
    let d_offsets = d.slot_offsets_handle(client)?;
    let e_offsets = e.slot_offsets_handle(client)?;
    let f_offsets = f.slot_offsets_handle(client)?;
    let g_offsets = g.slot_offsets_handle(client)?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);
    let block_sums_b = workspace.alloc::<B>(num_blocks);
    let block_sums_c = workspace.alloc::<C>(num_blocks);
    let block_sums_d = workspace.alloc::<D>(num_blocks);
    let block_sums_e = workspace.alloc::<E>(num_blocks);
    let block_sums_f = workspace.alloc::<F>(num_blocks);
    let block_sums_g = workspace.alloc::<G>(num_blocks);

    unsafe {
        tuple7_device_inclusive_scan_expr_block_kernel::launch_unchecked::<
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
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
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
            BufferArg::from_raw_parts(output_a.clone(), len),
            BufferArg::from_raw_parts(output_b.clone(), len),
            BufferArg::from_raw_parts(output_c.clone(), len),
            BufferArg::from_raw_parts(output_d.clone(), len),
            BufferArg::from_raw_parts(output_e.clone(), len),
            BufferArg::from_raw_parts(output_f.clone(), len),
            BufferArg::from_raw_parts(output_g.clone(), len),
            BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_b.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_c.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_d.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_e.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_f.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_g.clone(), num_blocks),
        );
    }

    if num_blocks > 1 {
        let block_prefixes = inclusive_scan_tuple7_handles::<R, A, B, C, D, E, F, G, Op>(
            policy,
            &block_sums_a,
            &block_sums_b,
            &block_sums_c,
            &block_sums_d,
            &block_sums_e,
            &block_sums_f,
            &block_sums_g,
            num_blocks,
        )?;
        add_tuple7_block_prefixes::<R, A, B, C, D, E, F, G, Op>(
            policy,
            &block_prefixes,
            num_blocks,
            len,
            &len_handle,
            &output_a,
            &output_b,
            &output_c,
            &output_d,
            &output_e,
            &output_f,
            &output_g,
        )?;
    }

    Ok((
        DeviceVec::from_handle(policy.id(), output_a, len),
        DeviceVec::from_handle(policy.id(), output_b, len),
        DeviceVec::from_handle(policy.id(), output_c, len),
        DeviceVec::from_handle(policy.id(), output_d, len),
        DeviceVec::from_handle(policy.id(), output_e, len),
        DeviceVec::from_handle(policy.id(), output_f, len),
        DeviceVec::from_handle(policy.id(), output_g, len),
    ))
}

#[allow(clippy::type_complexity)]
pub(crate) fn exclusive_scan_tuple7_device_expr<
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
    let inclusive = inclusive_scan_tuple7_device_expr::<
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
    >(policy, a, b, c, d, e, f, g, len)?;
    let output = make_tuple7_exclusive::<R, A, B, C, D, E, F, G, Op>(
        policy,
        &inclusive.0.handle,
        &inclusive.1.handle,
        &inclusive.2.handle,
        &inclusive.3.handle,
        &inclusive.4.handle,
        &inclusive.5.handle,
        &inclusive.6.handle,
        len,
        init,
    )?;
    Ok((
        DeviceVec::from_handle(policy.id(), output.0, len),
        DeviceVec::from_handle(policy.id(), output.1, len),
        DeviceVec::from_handle(policy.id(), output.2, len),
        DeviceVec::from_handle(policy.id(), output.3, len),
        DeviceVec::from_handle(policy.id(), output.4, len),
        DeviceVec::from_handle(policy.id(), output.5, len),
        DeviceVec::from_handle(policy.id(), output.6, len),
    ))
}

type Tuple7Handles = (
    cubecl::server::Handle,
    cubecl::server::Handle,
    cubecl::server::Handle,
    cubecl::server::Handle,
    cubecl::server::Handle,
    cubecl::server::Handle,
    cubecl::server::Handle,
);

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn inclusive_scan_tuple7_device_views<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    a: &DeviceColumnView<R, A>,
    b: &DeviceColumnView<R, B>,
    c: &DeviceColumnView<R, C>,
    d: &DeviceColumnView<R, D>,
    e: &DeviceColumnView<R, E>,
    f: &DeviceColumnView<R, F>,
    g: &DeviceColumnView<R, G>,
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
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    let len = a.len;
    let client = policy.client();
    if len == 0 {
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

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let output_d = client.empty(len * std::mem::size_of::<D>());
    let output_e = client.empty(len * std::mem::size_of::<E>());
    let output_f = client.empty(len * std::mem::size_of::<F>());
    let output_g = client.empty(len * std::mem::size_of::<G>());
    let workspace = Workspace::new(policy);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
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
    let block_sums_a = workspace.alloc::<A>(num_blocks);
    let block_sums_b = workspace.alloc::<B>(num_blocks);
    let block_sums_c = workspace.alloc::<C>(num_blocks);
    let block_sums_d = workspace.alloc::<D>(num_blocks);
    let block_sums_e = workspace.alloc::<E>(num_blocks);
    let block_sums_f = workspace.alloc::<F>(num_blocks);
    let block_sums_g = workspace.alloc::<G>(num_blocks);

    unsafe {
        tuple7_view_inclusive_scan_block_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a.source.handle.clone(), a.source.len()),
            BufferArg::from_raw_parts(b.source.handle.clone(), b.source.len()),
            BufferArg::from_raw_parts(c.source.handle.clone(), c.source.len()),
            BufferArg::from_raw_parts(d.source.handle.clone(), d.source.len()),
            BufferArg::from_raw_parts(e.source.handle.clone(), e.source.len()),
            BufferArg::from_raw_parts(f.source.handle.clone(), f.source.len()),
            BufferArg::from_raw_parts(g.source.handle.clone(), g.source.len()),
            BufferArg::from_raw_parts(offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_a.clone(), len),
            BufferArg::from_raw_parts(output_b.clone(), len),
            BufferArg::from_raw_parts(output_c.clone(), len),
            BufferArg::from_raw_parts(output_d.clone(), len),
            BufferArg::from_raw_parts(output_e.clone(), len),
            BufferArg::from_raw_parts(output_f.clone(), len),
            BufferArg::from_raw_parts(output_g.clone(), len),
            BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_b.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_c.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_d.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_e.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_f.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_g.clone(), num_blocks),
        );
    }

    if num_blocks > 1 {
        let block_prefixes = inclusive_scan_tuple7_handles::<R, A, B, C, D, E, F, G, Op>(
            policy,
            &block_sums_a,
            &block_sums_b,
            &block_sums_c,
            &block_sums_d,
            &block_sums_e,
            &block_sums_f,
            &block_sums_g,
            num_blocks,
        )?;
        add_tuple7_block_prefixes::<R, A, B, C, D, E, F, G, Op>(
            policy,
            &block_prefixes,
            num_blocks,
            len,
            &len_handle,
            &output_a,
            &output_b,
            &output_c,
            &output_d,
            &output_e,
            &output_f,
            &output_g,
        )?;
    }

    Ok((
        DeviceVec::from_handle(policy.id(), output_a, len),
        DeviceVec::from_handle(policy.id(), output_b, len),
        DeviceVec::from_handle(policy.id(), output_c, len),
        DeviceVec::from_handle(policy.id(), output_d, len),
        DeviceVec::from_handle(policy.id(), output_e, len),
        DeviceVec::from_handle(policy.id(), output_f, len),
        DeviceVec::from_handle(policy.id(), output_g, len),
    ))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn exclusive_scan_tuple7_device_views<R, A, B, C, D, E, F, G, Op>(
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
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    let inclusive = inclusive_scan_tuple7_device_views::<R, A, B, C, D, E, F, G, Op>(
        policy, a, b, c, d, e, f, g,
    )?;
    let output = make_tuple7_exclusive::<R, A, B, C, D, E, F, G, Op>(
        policy,
        &inclusive.0.handle,
        &inclusive.1.handle,
        &inclusive.2.handle,
        &inclusive.3.handle,
        &inclusive.4.handle,
        &inclusive.5.handle,
        &inclusive.6.handle,
        a.len,
        init,
    )?;
    Ok((
        DeviceVec::from_handle(policy.id(), output.0, a.len),
        DeviceVec::from_handle(policy.id(), output.1, a.len),
        DeviceVec::from_handle(policy.id(), output.2, a.len),
        DeviceVec::from_handle(policy.id(), output.3, a.len),
        DeviceVec::from_handle(policy.id(), output.4, a.len),
        DeviceVec::from_handle(policy.id(), output.5, a.len),
        DeviceVec::from_handle(policy.id(), output.6, a.len),
    ))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn adjacent_difference_tuple7_device_views<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    a: &DeviceColumnView<R, A>,
    b: &DeviceColumnView<R, B>,
    c: &DeviceColumnView<R, C>,
    d: &DeviceColumnView<R, D>,
    e: &DeviceColumnView<R, E>,
    f: &DeviceColumnView<R, F>,
    g: &DeviceColumnView<R, G>,
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
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    let len = a.len;
    let client = policy.client();
    if len == 0 {
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

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let output_d = client.empty(len * std::mem::size_of::<D>());
    let output_e = client.empty(len * std::mem::size_of::<E>());
    let output_f = client.empty(len * std::mem::size_of::<F>());
    let output_g = client.empty(len * std::mem::size_of::<G>());
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
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        tuple7_view_adjacent_difference_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(a.source.handle.clone(), a.source.len()),
            BufferArg::from_raw_parts(b.source.handle.clone(), b.source.len()),
            BufferArg::from_raw_parts(c.source.handle.clone(), c.source.len()),
            BufferArg::from_raw_parts(d.source.handle.clone(), d.source.len()),
            BufferArg::from_raw_parts(e.source.handle.clone(), e.source.len()),
            BufferArg::from_raw_parts(f.source.handle.clone(), f.source.len()),
            BufferArg::from_raw_parts(g.source.handle.clone(), g.source.len()),
            BufferArg::from_raw_parts(offsets_handle.clone(), 7),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_a.clone(), len),
            BufferArg::from_raw_parts(output_b.clone(), len),
            BufferArg::from_raw_parts(output_c.clone(), len),
            BufferArg::from_raw_parts(output_d.clone(), len),
            BufferArg::from_raw_parts(output_e.clone(), len),
            BufferArg::from_raw_parts(output_f.clone(), len),
            BufferArg::from_raw_parts(output_g.clone(), len),
        );
    }

    Ok((
        DeviceVec::from_handle(policy.id(), output_a, len),
        DeviceVec::from_handle(policy.id(), output_b, len),
        DeviceVec::from_handle(policy.id(), output_c, len),
        DeviceVec::from_handle(policy.id(), output_d, len),
        DeviceVec::from_handle(policy.id(), output_e, len),
        DeviceVec::from_handle(policy.id(), output_f, len),
        DeviceVec::from_handle(policy.id(), output_g, len),
    ))
}

#[allow(clippy::too_many_arguments)]
fn inclusive_scan_tuple7_handles<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    input_a: &cubecl::server::Handle,
    input_b: &cubecl::server::Handle,
    input_c: &cubecl::server::Handle,
    input_d: &cubecl::server::Handle,
    input_e: &cubecl::server::Handle,
    input_f: &cubecl::server::Handle,
    input_g: &cubecl::server::Handle,
    len: usize,
) -> Result<Tuple7Handles, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
        ));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let output_d = client.empty(len * std::mem::size_of::<D>());
    let output_e = client.empty(len * std::mem::size_of::<E>());
    let output_f = client.empty(len * std::mem::size_of::<F>());
    let output_g = client.empty(len * std::mem::size_of::<G>());
    let workspace = Workspace::new(policy);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);
    let block_sums_b = workspace.alloc::<B>(num_blocks);
    let block_sums_c = workspace.alloc::<C>(num_blocks);
    let block_sums_d = workspace.alloc::<D>(num_blocks);
    let block_sums_e = workspace.alloc::<E>(num_blocks);
    let block_sums_f = workspace.alloc::<F>(num_blocks);
    let block_sums_g = workspace.alloc::<G>(num_blocks);

    unsafe {
        tuple7_inclusive_scan_block_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(input_a.clone(), len),
            BufferArg::from_raw_parts(input_b.clone(), len),
            BufferArg::from_raw_parts(input_c.clone(), len),
            BufferArg::from_raw_parts(input_d.clone(), len),
            BufferArg::from_raw_parts(input_e.clone(), len),
            BufferArg::from_raw_parts(input_f.clone(), len),
            BufferArg::from_raw_parts(input_g.clone(), len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_a.clone(), len),
            BufferArg::from_raw_parts(output_b.clone(), len),
            BufferArg::from_raw_parts(output_c.clone(), len),
            BufferArg::from_raw_parts(output_d.clone(), len),
            BufferArg::from_raw_parts(output_e.clone(), len),
            BufferArg::from_raw_parts(output_f.clone(), len),
            BufferArg::from_raw_parts(output_g.clone(), len),
            BufferArg::from_raw_parts(block_sums_a.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_b.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_c.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_d.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_e.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_f.clone(), num_blocks),
            BufferArg::from_raw_parts(block_sums_g.clone(), num_blocks),
        );
    }

    if num_blocks > 1 {
        let block_prefixes = inclusive_scan_tuple7_handles::<R, A, B, C, D, E, F, G, Op>(
            policy,
            &block_sums_a,
            &block_sums_b,
            &block_sums_c,
            &block_sums_d,
            &block_sums_e,
            &block_sums_f,
            &block_sums_g,
            num_blocks,
        )?;
        add_tuple7_block_prefixes::<R, A, B, C, D, E, F, G, Op>(
            policy,
            &block_prefixes,
            num_blocks,
            len,
            &len_handle,
            &output_a,
            &output_b,
            &output_c,
            &output_d,
            &output_e,
            &output_f,
            &output_g,
        )?;
    }

    Ok((
        output_a, output_b, output_c, output_d, output_e, output_f, output_g,
    ))
}

#[allow(clippy::too_many_arguments)]
fn add_tuple7_block_prefixes<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    block_prefixes: &Tuple7Handles,
    num_blocks: usize,
    len: usize,
    len_handle: &cubecl::server::Handle,
    output_a: &cubecl::server::Handle,
    output_b: &cubecl::server::Handle,
    output_c: &cubecl::server::Handle,
    output_d: &cubecl::server::Handle,
    output_e: &cubecl::server::Handle,
    output_f: &cubecl::server::Handle,
    output_g: &cubecl::server::Handle,
) -> Result<(), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    unsafe {
        tuple7_scan_add_block_prefix_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            policy.client(),
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(block_prefixes.0.clone(), num_blocks),
            BufferArg::from_raw_parts(block_prefixes.1.clone(), num_blocks),
            BufferArg::from_raw_parts(block_prefixes.2.clone(), num_blocks),
            BufferArg::from_raw_parts(block_prefixes.3.clone(), num_blocks),
            BufferArg::from_raw_parts(block_prefixes.4.clone(), num_blocks),
            BufferArg::from_raw_parts(block_prefixes.5.clone(), num_blocks),
            BufferArg::from_raw_parts(block_prefixes.6.clone(), num_blocks),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output_a.clone(), len),
            BufferArg::from_raw_parts(output_b.clone(), len),
            BufferArg::from_raw_parts(output_c.clone(), len),
            BufferArg::from_raw_parts(output_d.clone(), len),
            BufferArg::from_raw_parts(output_e.clone(), len),
            BufferArg::from_raw_parts(output_f.clone(), len),
            BufferArg::from_raw_parts(output_g.clone(), len),
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn make_tuple7_exclusive<R, A, B, C, D, E, F, G, Op>(
    policy: &CubePolicy<R>,
    inclusive_a: &cubecl::server::Handle,
    inclusive_b: &cubecl::server::Handle,
    inclusive_c: &cubecl::server::Handle,
    inclusive_d: &cubecl::server::Handle,
    inclusive_e: &cubecl::server::Handle,
    inclusive_f: &cubecl::server::Handle,
    inclusive_g: &cubecl::server::Handle,
    len: usize,
    init: (A, B, C, D, E, F, G),
) -> Result<Tuple7Handles, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
        ));
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let output_d = client.empty(len * std::mem::size_of::<D>());
    let output_e = client.empty(len * std::mem::size_of::<E>());
    let output_f = client.empty(len * std::mem::size_of::<F>());
    let output_g = client.empty(len * std::mem::size_of::<G>());
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(C::as_bytes(&[init.2]));
    let init_d = client.create_from_slice(D::as_bytes(&[init.3]));
    let init_e = client.create_from_slice(E::as_bytes(&[init.4]));
    let init_f = client.create_from_slice(F::as_bytes(&[init.5]));
    let init_g = client.create_from_slice(G::as_bytes(&[init.6]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    unsafe {
        tuple7_scan_make_exclusive_kernel::launch_unchecked::<A, B, C, D, E, F, G, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(inclusive_a.clone(), len),
            BufferArg::from_raw_parts(inclusive_b.clone(), len),
            BufferArg::from_raw_parts(inclusive_c.clone(), len),
            BufferArg::from_raw_parts(inclusive_d.clone(), len),
            BufferArg::from_raw_parts(inclusive_e.clone(), len),
            BufferArg::from_raw_parts(inclusive_f.clone(), len),
            BufferArg::from_raw_parts(inclusive_g.clone(), len),
            BufferArg::from_raw_parts(init_a.clone(), 1),
            BufferArg::from_raw_parts(init_b.clone(), 1),
            BufferArg::from_raw_parts(init_c.clone(), 1),
            BufferArg::from_raw_parts(init_d.clone(), 1),
            BufferArg::from_raw_parts(init_e.clone(), 1),
            BufferArg::from_raw_parts(init_f.clone(), 1),
            BufferArg::from_raw_parts(init_g.clone(), 1),
            BufferArg::from_raw_parts(output_a.clone(), len),
            BufferArg::from_raw_parts(output_b.clone(), len),
            BufferArg::from_raw_parts(output_c.clone(), len),
            BufferArg::from_raw_parts(output_d.clone(), len),
            BufferArg::from_raw_parts(output_e.clone(), len),
            BufferArg::from_raw_parts(output_f.clone(), len),
            BufferArg::from_raw_parts(output_g.clone(), len),
        );
    }

    Ok((
        output_a, output_b, output_c, output_d, output_e, output_f, output_g,
    ))
}

pub(crate) fn inclusive_scan_by_key_handle<R, K, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    value_handle: &cubecl::server::Handle,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }
    if len == 1 {
        return range::copy_handle::<R, T>(policy, value_handle, len);
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let output_handle = workspace.alloc::<T>(len);
    let block_tail_keys = workspace.alloc::<K>(num_blocks);
    let block_tail_values = workspace.alloc::<T>(num_blocks);

    unsafe {
        scan_by_key_block_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(keys.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_tail_values.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let block_tail_keys_vec =
            DeviceVec::from_handle(policy.id(), block_tail_keys.clone(), num_blocks);
        let block_prefixes = inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(
            policy,
            &block_tail_keys_vec,
            &block_tail_values,
        )?;
        unsafe {
            scan_by_key_add_block_prefix_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(keys.handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }
    }

    Ok(output_handle)
}

pub(crate) fn inclusive_scan_by_key_device_expr_handle<R, K, T, KeyExpr, ValueExpr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    value_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ValueExpr: DeviceGpuExpr<T>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let output_handle = workspace.alloc::<T>(len);
    let block_tail_keys = workspace.alloc::<K>(num_blocks);
    let block_tail_values = workspace.alloc::<T>(num_blocks);
    let key_slot0 = binding_slot_or_first(key_bindings, 0);
    let key_slot1 = binding_slot_or_first(key_bindings, 1);
    let key_slot2 = binding_slot_or_first(key_bindings, 2);
    let key_slot3 = binding_slot_or_first(key_bindings, 3);
    let value_slot0 = binding_slot_or_first(value_bindings, 0);
    let value_slot1 = binding_slot_or_first(value_bindings, 1);
    let value_slot2 = binding_slot_or_first(value_bindings, 2);
    let value_slot3 = binding_slot_or_first(value_bindings, 3);
    let key_slot_offsets = key_bindings.slot_offsets_handle(client)?;
    let value_slot_offsets = value_bindings.slot_offsets_handle(client)?;

    unsafe {
        scan_by_key_device_expr_block_kernel::launch_unchecked::<
            K,
            T,
            KeyExpr,
            ValueExpr,
            KeyEq,
            Op,
            R,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
            unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
            unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
            unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
            unsafe { BufferArg::from_raw_parts(key_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(value_slot0.0.clone(), value_slot0.1) },
            unsafe { BufferArg::from_raw_parts(value_slot1.0.clone(), value_slot1.1) },
            unsafe { BufferArg::from_raw_parts(value_slot2.0.clone(), value_slot2.1) },
            unsafe { BufferArg::from_raw_parts(value_slot3.0.clone(), value_slot3.1) },
            unsafe { BufferArg::from_raw_parts(value_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(block_tail_values.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let block_tail_keys_vec =
            DeviceVec::from_handle(policy.id(), block_tail_keys.clone(), num_blocks);
        let block_prefixes = inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(
            policy,
            &block_tail_keys_vec,
            &block_tail_values,
        )?;
        unsafe {
            scan_by_key_device_expr_add_block_prefix_kernel::launch_unchecked::<
                K,
                T,
                KeyExpr,
                KeyEq,
                Op,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(block_prefixes.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }
    }

    Ok(output_handle)
}

fn make_scan_by_key_device_expr_exclusive<R, K, T, KeyExpr, KeyEq, Op>(
    client: &ComputeClient<R>,
    key_bindings: &KernelColumnBindings,
    len: usize,
    inclusive_handle: &cubecl::server::Handle,
    init: T,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    if len == 0 {
        return Ok(crate::policy::empty_handle(client));
    }

    let output_handle = client.empty(len * std::mem::size_of::<T>());
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    let key_slot0 = binding_slot_or_first(key_bindings, 0);
    let key_slot1 = binding_slot_or_first(key_bindings, 1);
    let key_slot2 = binding_slot_or_first(key_bindings, 2);
    let key_slot3 = binding_slot_or_first(key_bindings, 3);
    let key_slot_offsets = key_bindings.slot_offsets_handle(client)?;

    unsafe {
        scan_by_key_device_expr_make_exclusive_kernel::launch_unchecked::<
            K,
            T,
            KeyExpr,
            KeyEq,
            Op,
            R,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
            unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
            unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
            unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
            unsafe { BufferArg::from_raw_parts(key_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(inclusive_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(output_handle)
}

pub(crate) fn inclusive_scan_tuple2_by_key_values_device_expr_handle<
    R,
    K,
    A,
    B,
    KeyExpr,
    ExprA,
    ExprB,
    KeyEq,
    Op,
>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<(cubecl::server::Handle, cubecl::server::Handle), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((policy.empty_handle(), policy.empty_handle()));
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let output_a = workspace.alloc::<A>(len);
    let output_b = workspace.alloc::<B>(len);
    let block_tail_keys = workspace.alloc::<K>(num_blocks);
    let tail_a = workspace.alloc::<A>(num_blocks);
    let tail_b = workspace.alloc::<B>(num_blocks);
    let key_slot0 = binding_slot_or_first(key_bindings, 0);
    let key_slot1 = binding_slot_or_first(key_bindings, 1);
    let key_slot2 = binding_slot_or_first(key_bindings, 2);
    let key_slot3 = binding_slot_or_first(key_bindings, 3);
    let a_slot0 = binding_slot_or_first(a_bindings, 0);
    let a_slot1 = binding_slot_or_first(a_bindings, 1);
    let a_slot2 = binding_slot_or_first(a_bindings, 2);
    let a_slot3 = binding_slot_or_first(a_bindings, 3);
    let b_slot0 = binding_slot_or_first(b_bindings, 0);
    let b_slot1 = binding_slot_or_first(b_bindings, 1);
    let b_slot2 = binding_slot_or_first(b_bindings, 2);
    let b_slot3 = binding_slot_or_first(b_bindings, 3);
    let key_offsets = key_bindings.slot_offsets_handle(client)?;
    let a_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_offsets = b_bindings.slot_offsets_handle(client)?;

    unsafe {
        scan_by_key_tuple2_device_expr_block_kernel::launch_unchecked::<
            K,
            A,
            B,
            KeyExpr,
            ExprA,
            ExprB,
            KeyEq,
            Op,
            R,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
            unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
            unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
            unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
            unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(a_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(tail_a.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(tail_b.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let tail_keys = DeviceVec::from_handle(policy.id(), block_tail_keys.clone(), num_blocks);
        let tail_a_vec: DeviceVec<R, A> =
            DeviceVec::from_handle(policy.id(), tail_a.clone(), num_blocks);
        let tail_b_vec: DeviceVec<R, B> =
            DeviceVec::from_handle(policy.id(), tail_b.clone(), num_blocks);
        let (prefix_a, prefix_b) =
            inclusive_scan_tuple2_by_key_values_handle::<R, K, A, B, KeyEq, Op>(
                policy,
                &tail_keys,
                &tail_a_vec.handle,
                &tail_b_vec.handle,
            )?;
        unsafe {
            scan_by_key_tuple2_device_expr_add_block_prefix_kernel::launch_unchecked::<
                K,
                A,
                B,
                KeyExpr,
                KeyEq,
                Op,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(prefix_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(prefix_b.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            );
        }
    }

    Ok((output_a, output_b))
}

fn make_scan_tuple2_by_key_values_device_expr_exclusive<R, K, A, B, KeyExpr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    len: usize,
    inclusive: (cubecl::server::Handle, cubecl::server::Handle),
    init: (A, B),
) -> Result<(cubecl::server::Handle, cubecl::server::Handle), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
{
    if len == 0 {
        return Ok((policy.empty_handle(), policy.empty_handle()));
    }
    let client = policy.client();
    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let key_slot0 = binding_slot_or_first(key_bindings, 0);
    let key_slot1 = binding_slot_or_first(key_bindings, 1);
    let key_slot2 = binding_slot_or_first(key_bindings, 2);
    let key_slot3 = binding_slot_or_first(key_bindings, 3);
    let key_offsets = key_bindings.slot_offsets_handle(client)?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        scan_by_key_tuple2_device_expr_make_exclusive_kernel::launch_unchecked::<
            K,
            A,
            B,
            KeyExpr,
            KeyEq,
            Op,
            R,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
            unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
            unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
            unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
            unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(inclusive.0.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive.1.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
        );
    }

    Ok((output_a, output_b))
}

pub(crate) fn inclusive_scan_tuple3_by_key_values_device_expr_handle<
    R,
    K,
    A,
    B,
    C,
    KeyExpr,
    ExprA,
    ExprB,
    ExprC,
    KeyEq,
    Op,
>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<
    (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ),
    Error,
>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok((
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
        ));
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let output_a = workspace.alloc::<A>(len);
    let output_b = workspace.alloc::<B>(len);
    let output_c = workspace.alloc::<C>(len);
    let block_tail_keys = workspace.alloc::<K>(num_blocks);
    let tail_a = workspace.alloc::<A>(num_blocks);
    let tail_b = workspace.alloc::<B>(num_blocks);
    let tail_c = workspace.alloc::<C>(num_blocks);
    let key_slot0 = binding_slot_or_first(key_bindings, 0);
    let key_slot1 = binding_slot_or_first(key_bindings, 1);
    let key_slot2 = binding_slot_or_first(key_bindings, 2);
    let key_slot3 = binding_slot_or_first(key_bindings, 3);
    let a_slot0 = binding_slot_or_first(a_bindings, 0);
    let a_slot1 = binding_slot_or_first(a_bindings, 1);
    let a_slot2 = binding_slot_or_first(a_bindings, 2);
    let a_slot3 = binding_slot_or_first(a_bindings, 3);
    let b_slot0 = binding_slot_or_first(b_bindings, 0);
    let b_slot1 = binding_slot_or_first(b_bindings, 1);
    let b_slot2 = binding_slot_or_first(b_bindings, 2);
    let b_slot3 = binding_slot_or_first(b_bindings, 3);
    let c_slot0 = binding_slot_or_first(c_bindings, 0);
    let c_slot1 = binding_slot_or_first(c_bindings, 1);
    let c_slot2 = binding_slot_or_first(c_bindings, 2);
    let c_slot3 = binding_slot_or_first(c_bindings, 3);
    let key_offsets = key_bindings.slot_offsets_handle(client)?;
    let a_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_offsets = b_bindings.slot_offsets_handle(client)?;
    let c_offsets = c_bindings.slot_offsets_handle(client)?;

    unsafe {
        scan_by_key_tuple3_device_expr_block_kernel::launch_unchecked::<
            K,
            A,
            B,
            C,
            KeyExpr,
            ExprA,
            ExprB,
            ExprC,
            KeyEq,
            Op,
            R,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
            unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
            unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
            unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
            unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(a_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(c_slot0.0.clone(), c_slot0.1) },
            unsafe { BufferArg::from_raw_parts(c_slot1.0.clone(), c_slot1.1) },
            unsafe { BufferArg::from_raw_parts(c_slot2.0.clone(), c_slot2.1) },
            unsafe { BufferArg::from_raw_parts(c_slot3.0.clone(), c_slot3.1) },
            unsafe { BufferArg::from_raw_parts(c_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
            unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(tail_a.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(tail_b.clone(), num_blocks) },
            unsafe { BufferArg::from_raw_parts(tail_c.clone(), num_blocks) },
        );
    }

    if num_blocks > 1 {
        let tail_keys = DeviceVec::from_handle(policy.id(), block_tail_keys.clone(), num_blocks);
        let tail_a_vec: DeviceVec<R, A> =
            DeviceVec::from_handle(policy.id(), tail_a.clone(), num_blocks);
        let tail_b_vec: DeviceVec<R, B> =
            DeviceVec::from_handle(policy.id(), tail_b.clone(), num_blocks);
        let tail_c_vec: DeviceVec<R, C> =
            DeviceVec::from_handle(policy.id(), tail_c.clone(), num_blocks);
        let (prefix_a, prefix_b, prefix_c) =
            inclusive_scan_tuple3_by_key_values_handle::<R, K, A, B, C, KeyEq, Op>(
                policy,
                &tail_keys,
                &tail_a_vec.handle,
                &tail_b_vec.handle,
                &tail_c_vec.handle,
            )?;
        unsafe {
            scan_by_key_tuple3_device_expr_add_block_prefix_kernel::launch_unchecked::<
                K,
                A,
                B,
                C,
                KeyExpr,
                KeyEq,
                Op,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
                unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
                unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
                unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
                unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(block_tail_keys.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(prefix_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(prefix_b.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(prefix_c.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
            );
        }
    }

    Ok((output_a, output_b, output_c))
}

fn make_scan_tuple3_by_key_values_device_expr_exclusive<R, K, A, B, C, KeyExpr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    key_bindings: &KernelColumnBindings,
    len: usize,
    inclusive: (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ),
    init: (A, B, C),
) -> Result<
    (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ),
    Error,
>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
{
    if len == 0 {
        return Ok((
            policy.empty_handle(),
            policy.empty_handle(),
            policy.empty_handle(),
        ));
    }
    let client = policy.client();
    let output_a = client.empty(len * std::mem::size_of::<A>());
    let output_b = client.empty(len * std::mem::size_of::<B>());
    let output_c = client.empty(len * std::mem::size_of::<C>());
    let init_a = client.create_from_slice(A::as_bytes(&[init.0]));
    let init_b = client.create_from_slice(B::as_bytes(&[init.1]));
    let init_c = client.create_from_slice(C::as_bytes(&[init.2]));
    let key_slot0 = binding_slot_or_first(key_bindings, 0);
    let key_slot1 = binding_slot_or_first(key_bindings, 1);
    let key_slot2 = binding_slot_or_first(key_bindings, 2);
    let key_slot3 = binding_slot_or_first(key_bindings, 3);
    let key_offsets = key_bindings.slot_offsets_handle(client)?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        scan_by_key_tuple3_device_expr_make_exclusive_kernel::launch_unchecked::<
            K,
            A,
            B,
            C,
            KeyExpr,
            KeyEq,
            Op,
            R,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(key_slot0.0.clone(), key_slot0.1) },
            unsafe { BufferArg::from_raw_parts(key_slot1.0.clone(), key_slot1.1) },
            unsafe { BufferArg::from_raw_parts(key_slot2.0.clone(), key_slot2.1) },
            unsafe { BufferArg::from_raw_parts(key_slot3.0.clone(), key_slot3.1) },
            unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(inclusive.0.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive.1.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive.2.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_a.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_b.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(init_c.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
        );
    }

    Ok((output_a, output_b, output_c))
}

pub(crate) fn read_u32_scalar<R: Runtime>(
    client: &ComputeClient<R>,
    handle: cubecl::server::Handle,
) -> Result<u32, Error> {
    let bytes = client.read_one(handle).map_err(|err| Error::Launch {
        message: format!("{err:?}"),
    })?;
    Ok(u32::from_bytes(&bytes)[0])
}
