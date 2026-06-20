use crate::{
    device::{DeviceVec, KernelColumnBindings, SoA1, SoA2, SoA3},
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{BinaryOp, BinaryPredicateOp, GpuOp},
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
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let block_sums_handle = workspace.alloc::<u32>(num_blocks);

    unsafe {
        u32_block_inclusive_scan_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
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
                CubeCount::Static(num_blocks_u32, 1, 1),
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
) -> Result<SoA2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
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
        return Ok(SoA2 {
            left: DeviceVec::empty(policy.clone()),
            right: DeviceVec::empty(policy.clone()),
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
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        tuple2_adjacent_difference_expr_kernel::launch_unchecked::<A, B, ExprA, ExprB, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
        );
    }

    Ok(SoA2 {
        left: DeviceVec::from_handle(policy.clone(), output_a, len),
        right: DeviceVec::from_handle(policy.clone(), output_b, len),
    })
}

pub(crate) fn adjacent_difference_tuple3_device_expr<R, A, B, C, ExprA, ExprB, ExprC, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
) -> Result<SoA3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
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
        return Ok(SoA3 {
            first: DeviceVec::empty(policy.clone()),
            second: DeviceVec::empty(policy.clone()),
            third: DeviceVec::empty(policy.clone()),
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
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

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
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(c_slot0.0.clone(), c_slot0.1) },
            unsafe { BufferArg::from_raw_parts(c_slot1.0.clone(), c_slot1.1) },
            unsafe { BufferArg::from_raw_parts(c_slot2.0.clone(), c_slot2.1) },
            unsafe { BufferArg::from_raw_parts(c_slot3.0.clone(), c_slot3.1) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
        );
    }

    Ok(SoA3 {
        first: DeviceVec::from_handle(policy.clone(), output_a, len),
        second: DeviceVec::from_handle(policy.clone(), output_b, len),
        third: DeviceVec::from_handle(policy.clone(), output_c, len),
    })
}

pub(crate) fn inclusive_scan_by_key_device_vec<R, K, T, KeyEq, Op>(
    keys: &DeviceVec<R, K>,
    values: &DeviceVec<R, T>,
    _key_eq: GpuOp<KeyEq>,
    _op: GpuOp<Op>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    super::ensure_same_len(values.len(), keys.len())?;

    let output_handle =
        inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(values.policy(), keys, &values.handle)?;
    Ok(DeviceVec::from_handle(
        values.policy().clone(),
        output_handle,
        values.len(),
    ))
}

pub(crate) fn exclusive_scan_by_key_device_vec<R, K, T, KeyEq, Op>(
    keys: &DeviceVec<R, K>,
    values: &DeviceVec<R, T>,
    init: T,
    _key_eq: GpuOp<KeyEq>,
    _op: GpuOp<Op>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    super::ensure_same_len(values.len(), keys.len())?;

    let client = values.policy().client();
    let inclusive_handle =
        inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(values.policy(), keys, &values.handle)?;
    let output_handle =
        make_scan_by_key_exclusive::<R, K, T, KeyEq, Op>(client, keys, &inclusive_handle, init)?;
    Ok(DeviceVec::from_handle(
        values.policy().clone(),
        output_handle,
        values.len(),
    ))
}

macro_rules! define_scan_tuple_value_by_key_device_vec {
    (
        $inclusive_fn:ident,
        $exclusive_fn:ident,
        $handle_fn:ident,
        $exclusive_handle_fn:ident,
        $output:ident,
        $handles:ty,
        $block_kernel:ident,
        $add_prefix_kernel:ident,
        $exclusive_kernel:ident,
        ( $( $ty:ident: $value:ident: $out:ident: $tail:ident: $tail_vec:ident: $prefix:ident: $init:tt ),+ )
    ) => {
        pub(crate) fn $inclusive_fn<R, K, $( $ty ),+, KeyEq, Op>(
            keys: &DeviceVec<R, K>,
            $( $value: &DeviceVec<R, $ty>, )+
            _key_eq: GpuOp<KeyEq>,
            _op: GpuOp<Op>,
        ) -> Result<$output<$( DeviceVec<R, $ty> ),+>, Error>
        where
            R: Runtime,
            K: CubePrimitive + CubeElement,
            $( $ty: CubePrimitive + CubeElement, )+
            KeyEq: BinaryPredicateOp<K>,
            Op: BinaryOp<($( $ty ),+)>,
        {
            $(
                super::ensure_same_len($value.len(), keys.len())?;
            )+
            let ($( $out, )+) = $handle_fn::<R, K, $( $ty, )+ KeyEq, Op>(
                keys.policy(),
                keys,
                $( &$value.handle, )+
            )?;
            Ok($output {
                $( $value: DeviceVec::from_handle($value.policy().clone(), $out, $value.len()), )+
            })
        }

        pub(crate) fn $exclusive_fn<R, K, $( $ty ),+, KeyEq, Op>(
            keys: &DeviceVec<R, K>,
            $( $value: &DeviceVec<R, $ty>, )+
            init: ($( $ty ),+),
            _key_eq: GpuOp<KeyEq>,
            _op: GpuOp<Op>,
        ) -> Result<$output<$( DeviceVec<R, $ty> ),+>, Error>
        where
            R: Runtime,
            K: CubePrimitive + CubeElement,
            $( $ty: CubePrimitive + CubeElement, )+
            KeyEq: BinaryPredicateOp<K>,
            Op: BinaryOp<($( $ty ),+)>,
        {
            $(
                super::ensure_same_len($value.len(), keys.len())?;
            )+
            let inclusive = $handle_fn::<R, K, $( $ty, )+ KeyEq, Op>(
                keys.policy(),
                keys,
                $( &$value.handle, )+
            )?;
            let ($( $out, )+) = $exclusive_handle_fn::<R, K, $( $ty, )+ KeyEq, Op>(
                keys.policy(),
                keys,
                inclusive,
                init,
            )?;
            Ok($output {
                $( $value: DeviceVec::from_handle($value.policy().clone(), $out, $value.len()), )+
            })
        }

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
                let tail_keys = DeviceVec::from_handle(policy.clone(), block_tail_keys.clone(), num_blocks);
                $(
                    let $tail_vec = DeviceVec::<R, $ty>::from_handle(
                        policy.clone(),
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

        fn $exclusive_handle_fn<R, K, $( $ty ),+, KeyEq, Op>(
            policy: &CubePolicy<R>,
            keys: &DeviceVec<R, K>,
            inclusive: $handles,
            init: ($( $ty ),+),
        ) -> Result<$handles, Error>
        where
            R: Runtime,
            K: CubePrimitive + CubeElement,
            $( $ty: CubePrimitive + CubeElement, )+
            KeyEq: BinaryPredicateOp<K>,
            Op: BinaryOp<($( $ty ),+)>,
        {
            let len = keys.len();
            if len == 0 {
                return Ok(($( {
                    let _ = core::mem::size_of::<$ty>();
                    policy.empty_handle()
                }, )+));
            }

            let client = policy.client();
            $(
                let $out = client.empty(len * std::mem::size_of::<$ty>());
            )+
            let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
            $(
                let $value = client.create_from_slice($ty::as_bytes(&[init.$init]));
            )+
            unsafe {
                $exclusive_kernel::launch_unchecked::<K, $( $ty, )+ KeyEq, Op, R>(
                    client,
                    CubeCount::Static(num_blocks_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SCAN_SIZE),
                    unsafe { BufferArg::from_raw_parts(keys.handle.clone(), len) },
                    $(
                        unsafe { BufferArg::from_raw_parts(inclusive.$init.clone(), len) },
                    )+
                    $(
                        unsafe { BufferArg::from_raw_parts($value.clone(), 1) },
                    )+
                    $(
                        unsafe { BufferArg::from_raw_parts($out.clone(), len) },
                    )+
                );
            }

            Ok(($( $out, )+))
        }
    };
}

define_scan_tuple_value_by_key_device_vec!(
    inclusive_scan_tuple2_by_key_values_device_vec,
    exclusive_scan_tuple2_by_key_values_device_vec,
    inclusive_scan_tuple2_by_key_values_handle,
    make_scan_tuple2_by_key_values_exclusive,
    SoA2,
    (cubecl::server::Handle, cubecl::server::Handle),
    scan_by_key_tuple2_block_kernel,
    scan_by_key_tuple2_add_block_prefix_kernel,
    scan_by_key_tuple2_make_exclusive_kernel,
    (A: left: output_a: block_tail_a: block_tail_a_vec: prefix_a: 0, B: right: output_b: block_tail_b: block_tail_b_vec: prefix_b: 1)
);
define_scan_tuple_value_by_key_device_vec!(
    inclusive_scan_tuple3_by_key_values_device_vec,
    exclusive_scan_tuple3_by_key_values_device_vec,
    inclusive_scan_tuple3_by_key_values_handle,
    make_scan_tuple3_by_key_values_exclusive,
    SoA3,
    (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ),
    scan_by_key_tuple3_block_kernel,
    scan_by_key_tuple3_add_block_prefix_kernel,
    scan_by_key_tuple3_make_exclusive_kernel,
    (A: first: output_a: block_tail_a: block_tail_a_vec: prefix_a: 0, B: second: output_b: block_tail_b: block_tail_b_vec: prefix_b: 1, C: third: output_c: block_tail_c: block_tail_c_vec: prefix_c: 2)
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
        pub(crate) fn $inclusive_fn<R, $( $ty ),+, T, KeyEq, Op>(
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
                values.policy(),
                $( $key, )+
                &values.handle,
            )?;
            Ok(DeviceVec::from_handle(
                values.policy().clone(),
                output_handle,
                values.len(),
            ))
        }

        pub(crate) fn $exclusive_fn<R, $( $ty ),+, T, KeyEq, Op>(
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

            let client = values.policy().client();
            let inclusive_handle = $handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                values.policy(),
                $( $key, )+
                &values.handle,
            )?;
            let output_handle = $exclusive_handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                client,
                $( $key, )+
                &inclusive_handle,
                init,
            )?;
            Ok(DeviceVec::from_handle(
                values.policy().clone(),
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
                    let $tail_vec = DeviceVec::from_handle(
                        policy.clone(),
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
) -> Result<SoA1<DeviceVec<R, A>>, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    Op: BinaryOp<(A,)>,
{
    let client = policy.client();
    if len == 0 {
        return Ok(SoA1 {
            source: DeviceVec::empty(policy.clone()),
        });
    }

    let output_a = client.empty(len * std::mem::size_of::<A>());
    let workspace = Workspace::new(policy);
    let a_slot0 = a_bindings.slots.first().unwrap();
    let a_slot1 = a_bindings.slots.get(1).unwrap_or(a_slot0);
    let a_slot2 = a_bindings.slots.get(2).unwrap_or(a_slot0);
    let a_slot3 = a_bindings.slots.get(3).unwrap_or(a_slot0);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);

    unsafe {
        tuple1_device_inclusive_scan_expr_block_kernel::launch_unchecked::<A, ExprA, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(a_slot0.0.clone(), a_slot0.1) },
            unsafe { BufferArg::from_raw_parts(a_slot1.0.clone(), a_slot1.1) },
            unsafe { BufferArg::from_raw_parts(a_slot2.0.clone(), a_slot2.1) },
            unsafe { BufferArg::from_raw_parts(a_slot3.0.clone(), a_slot3.1) },
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
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                unsafe { BufferArg::from_raw_parts(block_prefixes_a.clone(), num_blocks) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
            );
        }
    }

    Ok(SoA1 {
        source: DeviceVec::from_handle(policy.clone(), output_a, len),
    })
}

pub(crate) fn exclusive_scan_tuple1_device_expr<R, A, ExprA, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    len: usize,
    init: (A,),
) -> Result<SoA1<DeviceVec<R, A>>, Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    ExprA: DeviceGpuExpr<A>,
    Op: BinaryOp<(A,)>,
{
    let inclusive = inclusive_scan_tuple1_device_expr::<R, A, ExprA, Op>(policy, a_bindings, len)?;
    let (output_a,) =
        make_tuple1_exclusive::<R, A, Op>(policy, &inclusive.source.handle, len, init)?;
    Ok(SoA1 {
        source: DeviceVec::from_handle(policy.clone(), output_a, len),
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
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_a = workspace.alloc::<A>(num_blocks);

    unsafe {
        tuple1_inclusive_scan_block_kernel::launch_unchecked::<A, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
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
                CubeCount::Static(num_blocks_u32, 1, 1),
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
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    unsafe {
        tuple1_scan_make_exclusive_kernel::launch_unchecked::<A, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
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
) -> Result<SoA2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
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
        return Ok(SoA2 {
            left: DeviceVec::empty(policy.clone()),
            right: DeviceVec::empty(policy.clone()),
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
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
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

    Ok(SoA2 {
        left: DeviceVec::from_handle(policy.clone(), output_a, len),
        right: DeviceVec::from_handle(policy.clone(), output_b, len),
    })
}

pub(crate) fn exclusive_scan_tuple2_device_expr<R, A, B, ExprA, ExprB, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    len: usize,
    init: (A, B),
) -> Result<SoA2<DeviceVec<R, A>, DeviceVec<R, B>>, Error>
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
    Ok(SoA2 {
        left: DeviceVec::from_handle(policy.clone(), output_a, len),
        right: DeviceVec::from_handle(policy.clone(), output_b, len),
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
) -> Result<SoA3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
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
        return Ok(SoA3 {
            first: DeviceVec::empty(policy.clone()),
            second: DeviceVec::empty(policy.clone()),
            third: DeviceVec::empty(policy.clone()),
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
            unsafe { BufferArg::from_raw_parts(b_slot0.0.clone(), b_slot0.1) },
            unsafe { BufferArg::from_raw_parts(b_slot1.0.clone(), b_slot1.1) },
            unsafe { BufferArg::from_raw_parts(b_slot2.0.clone(), b_slot2.1) },
            unsafe { BufferArg::from_raw_parts(b_slot3.0.clone(), b_slot3.1) },
            unsafe { BufferArg::from_raw_parts(c_slot0.0.clone(), c_slot0.1) },
            unsafe { BufferArg::from_raw_parts(c_slot1.0.clone(), c_slot1.1) },
            unsafe { BufferArg::from_raw_parts(c_slot2.0.clone(), c_slot2.1) },
            unsafe { BufferArg::from_raw_parts(c_slot3.0.clone(), c_slot3.1) },
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

    Ok(SoA3 {
        first: DeviceVec::from_handle(policy.clone(), output_a, len),
        second: DeviceVec::from_handle(policy.clone(), output_b, len),
        third: DeviceVec::from_handle(policy.clone(), output_c, len),
    })
}

pub(crate) fn exclusive_scan_tuple3_device_expr<R, A, B, C, ExprA, ExprB, ExprC, Op>(
    policy: &CubePolicy<R>,
    a_bindings: &KernelColumnBindings,
    b_bindings: &KernelColumnBindings,
    c_bindings: &KernelColumnBindings,
    len: usize,
    init: (A, B, C),
) -> Result<SoA3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>, Error>
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
    Ok(SoA3 {
        first: DeviceVec::from_handle(policy.clone(), output_a, len),
        second: DeviceVec::from_handle(policy.clone(), output_b, len),
        third: DeviceVec::from_handle(policy.clone(), output_c, len),
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
            DeviceVec::from_handle(policy.clone(), block_tail_keys.clone(), num_blocks);
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

fn make_scan_by_key_exclusive<R, K, T, KeyEq, Op>(
    client: &ComputeClient<R>,
    keys: &DeviceVec<R, K>,
    inclusive_handle: &cubecl::server::Handle,
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
        return Ok(crate::policy::empty_handle(client));
    }

    let output_handle = client.empty(len * std::mem::size_of::<T>());
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    unsafe {
        scan_by_key_make_exclusive_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            unsafe { BufferArg::from_raw_parts(keys.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(inclusive_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(init_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(output_handle)
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
