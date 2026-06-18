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

pub(crate) fn reduce_input_handle<R, T, Op>(
    policy: &CubePolicy<R>,
    input_handle: cubecl::server::Handle,
    storage_len: usize,
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
    let workspace = Workspace::new(policy);
    let dummy_indices = [0_u32];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
    let mut current_handle = input_handle;
    let mut current_len = len;
    let mut current_storage_len = storage_len;

    while current_len > 1 {
        let partial_len = current_len.div_ceil(BLOCK_REDUCE_SIZE as usize);
        let partial_len_u32 =
            u32::try_from(partial_len).map_err(|_| Error::LengthTooLarge { len: partial_len })?;
        let current_len_u32 =
            u32::try_from(current_len).map_err(|_| Error::LengthTooLarge { len: current_len })?;
        let len_values = [current_len_u32];
        let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
        let partial_handle = workspace.alloc::<T>(partial_len);

        unsafe {
            reduce_expr_partials_kernel::launch_unchecked::<T, Input<T>, Op, R>(
                client,
                CubeCount::Static(partial_len_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                ArrayArg::from_raw_parts::<T>(&current_handle, current_storage_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<T>(&current_handle, current_storage_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&partial_handle, partial_len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        current_handle = partial_handle;
        current_len = partial_len;
        current_storage_len = partial_len;
    }

    finalize_handle::<R, T, Op>(policy, current_handle, init)
}

pub(crate) fn reduce_device_expr<R, T, Expr, Op>(
    policy: &CubePolicy<R>,
    bindings: &KernelColumnBindings,
    len: usize,
    init: T,
) -> Result<T, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Expr: DeviceGpuExpr<T>,
    Op: BinaryOp<T>,
{
    if len == 0 {
        return Ok(init);
    }

    let client = policy.client();
    let workspace = Workspace::new(policy);
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let partial_len = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let partial_len_u32 =
        u32::try_from(partial_len).map_err(|_| Error::LengthTooLarge { len: partial_len })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let partial_handle = workspace.alloc::<T>(partial_len);

    unsafe {
        device_reduce_expr_partials_kernel::launch_unchecked::<T, Expr, Op, R>(
            client,
            CubeCount::Static(partial_len_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            ArrayArg::from_raw_parts::<T>(&slot0.0, slot0.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot1.0, slot1.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot2.0, slot2.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot3.0, slot3.1, 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&partial_handle, partial_len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    reduce_input_handle::<R, T, Op>(policy, partial_handle, partial_len, partial_len, init)
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
            DeviceVec::empty(policy.clone()),
            DeviceVec::empty(policy.clone()),
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
            ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len(), 1),
            ArrayArg::from_raw_parts::<T>(&value_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&local_inclusive_handle, len, 1),
            ArrayArg::from_raw_parts::<K>(&block_tail_keys, scan_blocks, 1),
            ArrayArg::from_raw_parts::<T>(&block_tail_values, scan_blocks, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    if scan_blocks > 1 {
        let block_tail_keys_vec =
            DeviceVec::from_handle(policy.clone(), block_tail_keys.clone(), scan_blocks);
        let block_prefixes = scan::inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(
            policy,
            &block_tail_keys_vec,
            &block_tail_values,
        )?;
        unsafe {
            reduce_by_key_end_flags_with_block_prefix_kernel::launch_unchecked::<
                K,
                T,
                KeyEq,
                Op,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&local_inclusive_handle, len, 1),
                ArrayArg::from_raw_parts::<K>(&block_tail_keys, scan_blocks, 1),
                ArrayArg::from_raw_parts::<T>(&block_prefixes, scan_blocks, 1),
                ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&reduced_value_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    } else {
        unsafe {
            reduce_by_key_end_flags_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&local_inclusive_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&reduced_value_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    let control =
        ReduceByKeyControl::from_end_flags(policy, len, len_u32, flag_handle, keys.handle.clone())?;
    let (out_keys, out_values) =
        control.compact_pair::<R, K, T>(policy, keys.handle.clone(), reduced_value_handle)?;
    Ok((out_keys, out_values, control))
}

pub(crate) fn reduce_tuple2_by_key_device_vec<R, A, B, T, KeyEq, Op>(
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

    let policy = values.policy();
    let len = key_a.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    if len == 0 {
        return Ok((
            DeviceVec::empty(policy.clone()),
            DeviceVec::empty(policy.clone()),
            DeviceVec::empty(policy.clone()),
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
            ArrayArg::from_raw_parts::<A>(&key_a.handle, len, 1),
            ArrayArg::from_raw_parts::<B>(&key_b.handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&inclusive_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&reduced_value_handle, len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
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

    let policy = values.policy();
    let len = key_a.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    if len == 0 {
        return Ok((
            DeviceVec::empty(policy.clone()),
            DeviceVec::empty(policy.clone()),
            DeviceVec::empty(policy.clone()),
            DeviceVec::empty(policy.clone()),
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
            ArrayArg::from_raw_parts::<A>(&key_a.handle, len, 1),
            ArrayArg::from_raw_parts::<B>(&key_b.handle, len, 1),
            ArrayArg::from_raw_parts::<C>(&key_c.handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&inclusive_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&reduced_value_handle, len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
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

macro_rules! define_reduce_tuple_by_key_device_vec {
    (
        $fn_name:ident,
        $scan_handle:ident,
        $end_flags_kernel:ident,
        ( $( $ty:ident: $key:ident: $out_key:ident: $key_handles:ident ),+ )
    ) => {
        pub(crate) fn $fn_name<R, $( $ty ),+, T, KeyEq, Op>(
            $( $key: &DeviceVec<R, $ty>, )+
            values: &DeviceVec<R, T>,
            init: T,
        ) -> Result<( $( DeviceVec<R, $ty>, )+ DeviceVec<R, T> ), Error>
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

            let policy = values.policy();
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let client = policy.client();
            if len == 0 {
                return Ok((
                    $( {
                        let _ = $key;
                        DeviceVec::empty(policy.clone())
                    }, )+
                    DeviceVec::empty(policy.clone()),
                ));
            }

            let init_handle = client.create_from_slice(T::as_bytes(&[init]));
            let inclusive_handle = scan::$scan_handle::<R, $( $ty, )+ T, KeyEq, Op>(
                policy,
                $( $key, )+
                &values.handle,
            )?;
            let workspace = Workspace::new(policy);
            let flag_handle = workspace.alloc::<u32>(len);
            let reduced_value_handle = workspace.alloc::<T>(len);
            let num_blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

            unsafe {
                $end_flags_kernel::launch_unchecked::<$( $ty, )+ T, KeyEq, Op, R>(
                    client,
                    CubeCount::Static(num_blocks_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                    $(
                        ArrayArg::from_raw_parts::<$ty>(&$key.handle, len, 1),
                    )+
                    ArrayArg::from_raw_parts::<T>(&inclusive_handle, len, 1),
                    ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    ArrayArg::from_raw_parts::<T>(&reduced_value_handle, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }

            let control = segmented::SegmentControl::from_end_flags(
                policy,
                len,
                len_u32,
                flag_handle,
                values.handle.clone(),
            )?;
            $(
                let $key_handles = $key.handle.clone();
                let $out_key = control.compact_value::<R, $ty>(policy, $key_handles)?;
            )+
            let out_values = control.compact_value::<R, T>(policy, reduced_value_handle)?;
            Ok(( $( $out_key, )+ out_values ))
        }
    };
}

define_reduce_tuple_by_key_device_vec!(
    reduce_tuple4_by_key_device_vec,
    inclusive_scan_tuple4_by_key_handle,
    reduce_tuple4_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple5_by_key_device_vec,
    inclusive_scan_tuple5_by_key_handle,
    reduce_tuple5_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple6_by_key_device_vec,
    inclusive_scan_tuple6_by_key_handle,
    reduce_tuple6_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles, F: key_f: out_key_f: key_f_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple7_by_key_device_vec,
    inclusive_scan_tuple7_by_key_handle,
    reduce_tuple7_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles, F: key_f: out_key_f: key_f_handles, G: key_g: out_key_g: key_g_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple8_by_key_device_vec,
    inclusive_scan_tuple8_by_key_handle,
    reduce_tuple8_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles, F: key_f: out_key_f: key_f_handles, G: key_g: out_key_g: key_g_handles, I: key_h: out_key_h: key_h_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple9_by_key_device_vec,
    inclusive_scan_tuple9_by_key_handle,
    reduce_tuple9_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles, F: key_f: out_key_f: key_f_handles, G: key_g: out_key_g: key_g_handles, I: key_h: out_key_h: key_h_handles, J: key_i: out_key_i: key_i_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple10_by_key_device_vec,
    inclusive_scan_tuple10_by_key_handle,
    reduce_tuple10_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles, F: key_f: out_key_f: key_f_handles, G: key_g: out_key_g: key_g_handles, I: key_h: out_key_h: key_h_handles, J: key_i: out_key_i: key_i_handles, K: key_j: out_key_j: key_j_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple11_by_key_device_vec,
    inclusive_scan_tuple11_by_key_handle,
    reduce_tuple11_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles, F: key_f: out_key_f: key_f_handles, G: key_g: out_key_g: key_g_handles, I: key_h: out_key_h: key_h_handles, J: key_i: out_key_i: key_i_handles, K: key_j: out_key_j: key_j_handles, L: key_k: out_key_k: key_k_handles)
);
define_reduce_tuple_by_key_device_vec!(
    reduce_tuple12_by_key_device_vec,
    inclusive_scan_tuple12_by_key_handle,
    reduce_tuple12_by_key_end_flags_kernel,
    (A: key_a: out_key_a: key_a_handles, B: key_b: out_key_b: key_b_handles, C: key_c: out_key_c: key_c_handles, D: key_d: out_key_d: key_d_handles, E: key_e: out_key_e: key_e_handles, F: key_f: out_key_f: key_f_handles, G: key_g: out_key_g: key_g_handles, I: key_h: out_key_h: key_h_handles, J: key_i: out_key_i: key_i_handles, K: key_j: out_key_j: key_j_handles, L: key_k: out_key_k: key_k_handles, M: key_l: out_key_l: key_l_handles)
);

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
        return Ok(DeviceVec::empty(policy.clone()));
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
            ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len(), 1),
            ArrayArg::from_raw_parts::<T>(&value_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&local_inclusive_handle, len, 1),
            ArrayArg::from_raw_parts::<K>(&block_tail_keys, scan_blocks, 1),
            ArrayArg::from_raw_parts::<T>(&block_tail_values, scan_blocks, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    if scan_blocks > 1 {
        let block_tail_keys_vec =
            DeviceVec::from_handle(policy.clone(), block_tail_keys.clone(), scan_blocks);
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
                ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&local_inclusive_handle, len, 1),
                ArrayArg::from_raw_parts::<K>(&block_tail_keys, scan_blocks, 1),
                ArrayArg::from_raw_parts::<T>(&block_prefixes, scan_blocks, 1),
                ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    } else {
        unsafe {
            reduce_by_key_values_at_ends_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&local_inclusive_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(output_handle)
}

pub(crate) fn reduce_by_key_expr_handle<R, K, T, Expr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    input_handle: cubecl::server::Handle,
    input_len: usize,
    rhs_handle: cubecl::server::Handle,
    rhs_len: usize,
    init: T,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Expr: GpuExpr<T>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let value_handle = collect_reduce_by_key_expr_handle::<R, T, Expr>(
        policy,
        len,
        input_handle,
        input_len,
        rhs_handle,
        rhs_len,
    )?;

    reduce_by_key_handle::<R, K, T, KeyEq, Op>(policy, keys, value_handle, init)
}

pub(crate) fn reduce_by_key_expr_handle_with_control<R, K, T, Expr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    input_handle: cubecl::server::Handle,
    input_len: usize,
    rhs_handle: cubecl::server::Handle,
    rhs_len: usize,
    init: T,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>, ReduceByKeyControl), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Expr: GpuExpr<T>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let value_handle = collect_reduce_by_key_expr_handle::<R, T, Expr>(
        policy,
        len,
        input_handle,
        input_len,
        rhs_handle,
        rhs_len,
    )?;

    reduce_by_key_handle_with_control::<R, K, T, KeyEq, Op>(policy, keys, value_handle, init)
}

pub(crate) fn reduce_by_key_expr_handle_with_existing_control<R, K, T, Expr, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    input_handle: cubecl::server::Handle,
    input_len: usize,
    rhs_handle: cubecl::server::Handle,
    rhs_len: usize,
    init: T,
    control: &ReduceByKeyControl,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Expr: GpuExpr<T>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let value_handle = collect_reduce_by_key_expr_handle::<R, T, Expr>(
        policy,
        len,
        input_handle,
        input_len,
        rhs_handle,
        rhs_len,
    )?;

    reduce_by_key_handle_with_existing_control::<R, K, T, KeyEq, Op>(
        policy,
        keys,
        value_handle,
        init,
        control,
    )
}

fn collect_reduce_by_key_expr_handle<R, T, Expr>(
    policy: &CubePolicy<R>,
    len: usize,
    input_handle: cubecl::server::Handle,
    input_len: usize,
    rhs_handle: cubecl::server::Handle,
    rhs_len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Expr: GpuExpr<T>,
{
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let client = policy.client();
    let workspace = Workspace::new(policy);
    let value_handle = workspace.alloc::<T>(len);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let block_size = 256_u32;
    let block_count = len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let dummy_indices = [0_u32];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
    unsafe {
        collect_expr_block_kernel::launch_unchecked::<T, Expr, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            ArrayArg::from_raw_parts::<T>(&value_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&input_handle, input_len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<T>(&rhs_handle, rhs_len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(value_handle)
}

pub(crate) fn finalize_handle<R, T, Op>(
    policy: &CubePolicy<R>,
    partial_handle: cubecl::server::Handle,
    init: T,
) -> Result<T, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
{
    let client = policy.client();
    let init_values = [init];
    let init_handle = client.create_from_slice(T::as_bytes(&init_values));
    let output_handle = client.empty(std::mem::size_of::<T>());

    unsafe {
        reduce_finalize_kernel::launch_unchecked::<T, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<T>(&partial_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, 1, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    let bytes = client.read_one(output_handle);
    Ok(T::from_bytes(&bytes)[0].clone())
}
