use crate::{detail::op::kernel::BinaryPredicateOp, expr::DeviceGpuExpr};
use cubecl::prelude::*;

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_path_device_expr_kernel<
    T: CubePrimitive,
    LeftExpr: DeviceGpuExpr<T>,
    RightExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    output: &mut [T],
    lhs_slot0: &[T],
    lhs_slot1: &[T],
    lhs_slot2: &[T],
    lhs_slot3: &[T],
    lhs_slot_offsets: &[u32],
    lhs_len: &[u32],
    rhs_slot0: &[T],
    rhs_slot1: &[T],
    rhs_slot2: &[T],
    rhs_slot3: &[T],
    rhs_slot_offsets: &[u32],
    rhs_len: &[u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let lhs_logical_len = lhs_len[0] as usize;
    let rhs_logical_len = rhs_len[0] as usize;
    if out < output.len() {
        let low_init = RuntimeCell::<usize>::new(0usize);
        if out > rhs_logical_len {
            low_init.store(out - rhs_logical_len);
        }

        let high_init = RuntimeCell::<usize>::new(out);
        if high_init.read() > lhs_logical_len {
            high_init.store(lhs_logical_len);
        }

        let low = RuntimeCell::<usize>::new(low_init.read());
        let high = RuntimeCell::<usize>::new(high_init.read());
        while low.read() < high.read() {
            let mid = (low.read() + high.read()) / 2usize;
            let rhs_index = out - mid;
            if mid < lhs_logical_len
                && rhs_index > 0usize
                && !Less::apply(
                    RightExpr::eval(
                        rhs_slot0,
                        rhs_slot1,
                        rhs_slot2,
                        rhs_slot3,
                        rhs_slot_offsets,
                        rhs_index - 1usize,
                    ),
                    LeftExpr::eval(
                        lhs_slot0,
                        lhs_slot1,
                        lhs_slot2,
                        lhs_slot3,
                        lhs_slot_offsets,
                        mid,
                    ),
                )
            {
                low.store(mid + 1usize);
            } else {
                high.store(mid);
            }
        }

        let lhs_index = low.read();
        let rhs_index = out - lhs_index;
        if lhs_index < lhs_logical_len {
            let lhs_value = LeftExpr::eval(
                lhs_slot0,
                lhs_slot1,
                lhs_slot2,
                lhs_slot3,
                lhs_slot_offsets,
                lhs_index,
            );
            if rhs_index >= rhs_logical_len {
                output[out] = lhs_value;
            } else {
                let rhs_value = RightExpr::eval(
                    rhs_slot0,
                    rhs_slot1,
                    rhs_slot2,
                    rhs_slot3,
                    rhs_slot_offsets,
                    rhs_index,
                );
                if !Less::apply(rhs_value, lhs_value) {
                    output[out] = lhs_value;
                } else {
                    output[out] = rhs_value;
                }
            }
        } else {
            output[out] = RightExpr::eval(
                rhs_slot0,
                rhs_slot1,
                rhs_slot2,
                rhs_slot3,
                rhs_slot_offsets,
                rhs_index,
            );
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_path_control_device_expr_kernel<
    T: CubePrimitive,
    LeftExpr: DeviceGpuExpr<T>,
    RightExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    lhs_slot0: &[T],
    lhs_slot1: &[T],
    lhs_slot2: &[T],
    lhs_slot3: &[T],
    lhs_slot_offsets: &[u32],
    lhs_len: &[u32],
    rhs_slot0: &[T],
    rhs_slot1: &[T],
    rhs_slot2: &[T],
    rhs_slot3: &[T],
    rhs_slot_offsets: &[u32],
    rhs_len: &[u32],
    source_sides: &mut [u32],
    source_indices: &mut [u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let lhs_logical_len = lhs_len[0] as usize;
    let rhs_logical_len = rhs_len[0] as usize;
    if out < source_sides.len() {
        let low_init = RuntimeCell::<usize>::new(0usize);
        if out > rhs_logical_len {
            low_init.store(out - rhs_logical_len);
        }

        let high_init = RuntimeCell::<usize>::new(out);
        if high_init.read() > lhs_logical_len {
            high_init.store(lhs_logical_len);
        }

        let low = RuntimeCell::<usize>::new(low_init.read());
        let high = RuntimeCell::<usize>::new(high_init.read());
        while low.read() < high.read() {
            let mid = (low.read() + high.read()) / 2usize;
            let rhs_index = out - mid;
            if mid < lhs_logical_len
                && rhs_index > 0usize
                && !Less::apply(
                    RightExpr::eval(
                        rhs_slot0,
                        rhs_slot1,
                        rhs_slot2,
                        rhs_slot3,
                        rhs_slot_offsets,
                        rhs_index - 1usize,
                    ),
                    LeftExpr::eval(
                        lhs_slot0,
                        lhs_slot1,
                        lhs_slot2,
                        lhs_slot3,
                        lhs_slot_offsets,
                        mid,
                    ),
                )
            {
                low.store(mid + 1usize);
            } else {
                high.store(mid);
            }
        }

        let lhs_index = low.read();
        let rhs_index = out - lhs_index;
        if lhs_index < lhs_logical_len {
            if rhs_index >= rhs_logical_len {
                source_sides[out] = 0u32;
                source_indices[out] = lhs_index as u32;
            } else {
                let lhs_value = LeftExpr::eval(
                    lhs_slot0,
                    lhs_slot1,
                    lhs_slot2,
                    lhs_slot3,
                    lhs_slot_offsets,
                    lhs_index,
                );
                let rhs_value = RightExpr::eval(
                    rhs_slot0,
                    rhs_slot1,
                    rhs_slot2,
                    rhs_slot3,
                    rhs_slot_offsets,
                    rhs_index,
                );
                if !Less::apply(rhs_value, lhs_value) {
                    source_sides[out] = 0u32;
                    source_indices[out] = lhs_index as u32;
                } else {
                    source_sides[out] = 1u32;
                    source_indices[out] = rhs_index as u32;
                }
            }
        } else {
            source_sides[out] = 1u32;
            source_indices[out] = rhs_index as u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_by_key_control_device_expr_kernel<
    K: CubePrimitive,
    LeftExpr: DeviceGpuExpr<K>,
    RightExpr: DeviceGpuExpr<K>,
    Less: BinaryPredicateOp<K>,
>(
    lhs_slot0: &[K],
    lhs_slot1: &[K],
    lhs_slot2: &[K],
    lhs_slot3: &[K],
    lhs_slot_offsets: &[u32],
    lhs_len: &[u32],
    rhs_slot0: &[K],
    rhs_slot1: &[K],
    rhs_slot2: &[K],
    rhs_slot3: &[K],
    rhs_slot_offsets: &[u32],
    rhs_len: &[u32],
    out_keys: &mut [K],
    source_sides: &mut [u32],
    source_indices: &mut [u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let lhs_logical_len = lhs_len[0] as usize;
    let rhs_logical_len = rhs_len[0] as usize;
    if out < out_keys.len() {
        let low_init = RuntimeCell::<usize>::new(0usize);
        if out > rhs_logical_len {
            low_init.store(out - rhs_logical_len);
        }

        let high_init = RuntimeCell::<usize>::new(out);
        if high_init.read() > lhs_logical_len {
            high_init.store(lhs_logical_len);
        }

        let low = RuntimeCell::<usize>::new(low_init.read());
        let high = RuntimeCell::<usize>::new(high_init.read());
        while low.read() < high.read() {
            let mid = (low.read() + high.read()) / 2usize;
            let rhs_index = out - mid;
            if mid < lhs_logical_len
                && rhs_index > 0usize
                && !Less::apply(
                    RightExpr::eval(
                        rhs_slot0,
                        rhs_slot1,
                        rhs_slot2,
                        rhs_slot3,
                        rhs_slot_offsets,
                        rhs_index - 1usize,
                    ),
                    LeftExpr::eval(
                        lhs_slot0,
                        lhs_slot1,
                        lhs_slot2,
                        lhs_slot3,
                        lhs_slot_offsets,
                        mid,
                    ),
                )
            {
                low.store(mid + 1usize);
            } else {
                high.store(mid);
            }
        }

        let lhs_index = low.read();
        let rhs_index = out - lhs_index;
        if lhs_index < lhs_logical_len {
            let lhs_key = LeftExpr::eval(
                lhs_slot0,
                lhs_slot1,
                lhs_slot2,
                lhs_slot3,
                lhs_slot_offsets,
                lhs_index,
            );
            if rhs_index >= rhs_logical_len {
                out_keys[out] = lhs_key;
                source_sides[out] = 0u32;
                source_indices[out] = lhs_index as u32;
            } else {
                let rhs_key = RightExpr::eval(
                    rhs_slot0,
                    rhs_slot1,
                    rhs_slot2,
                    rhs_slot3,
                    rhs_slot_offsets,
                    rhs_index,
                );
                if !Less::apply(rhs_key, lhs_key) {
                    out_keys[out] = lhs_key;
                    source_sides[out] = 0u32;
                    source_indices[out] = lhs_index as u32;
                } else {
                    out_keys[out] = rhs_key;
                    source_sides[out] = 1u32;
                    source_indices[out] = rhs_index as u32;
                }
            }
        } else {
            out_keys[out] = RightExpr::eval(
                rhs_slot0,
                rhs_slot1,
                rhs_slot2,
                rhs_slot3,
                rhs_slot_offsets,
                rhs_index,
            );
            source_sides[out] = 1u32;
            source_indices[out] = rhs_index as u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_tuple2_by_key_control_device_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    LeftAExpr: DeviceGpuExpr<A>,
    LeftBExpr: DeviceGpuExpr<B>,
    RightAExpr: DeviceGpuExpr<A>,
    RightBExpr: DeviceGpuExpr<B>,
    Less: BinaryPredicateOp<(A, B)>,
>(
    lhs_a_slot0: &[A],
    lhs_a_slot1: &[A],
    lhs_a_slot2: &[A],
    lhs_a_slot3: &[A],
    lhs_a_slot_offsets: &[u32],
    lhs_b_slot0: &[B],
    lhs_b_slot1: &[B],
    lhs_b_slot2: &[B],
    lhs_b_slot3: &[B],
    lhs_b_slot_offsets: &[u32],
    lhs_len: &[u32],
    rhs_a_slot0: &[A],
    rhs_a_slot1: &[A],
    rhs_a_slot2: &[A],
    rhs_a_slot3: &[A],
    rhs_a_slot_offsets: &[u32],
    rhs_b_slot0: &[B],
    rhs_b_slot1: &[B],
    rhs_b_slot2: &[B],
    rhs_b_slot3: &[B],
    rhs_b_slot_offsets: &[u32],
    rhs_len: &[u32],
    out_a: &mut [A],
    out_b: &mut [B],
    source_sides: &mut [u32],
    source_indices: &mut [u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let lhs_logical_len = lhs_len[0] as usize;
    let rhs_logical_len = rhs_len[0] as usize;
    if out < out_a.len() {
        let low_init = RuntimeCell::<usize>::new(0usize);
        if out > rhs_logical_len {
            low_init.store(out - rhs_logical_len);
        }

        let high_init = RuntimeCell::<usize>::new(out);
        if high_init.read() > lhs_logical_len {
            high_init.store(lhs_logical_len);
        }

        let low = RuntimeCell::<usize>::new(low_init.read());
        let high = RuntimeCell::<usize>::new(high_init.read());
        while low.read() < high.read() {
            let mid = (low.read() + high.read()) / 2usize;
            let rhs_index = out - mid;
            if mid < lhs_logical_len && rhs_index > 0usize {
                let right_a = RightAExpr::eval(
                    rhs_a_slot0,
                    rhs_a_slot1,
                    rhs_a_slot2,
                    rhs_a_slot3,
                    rhs_a_slot_offsets,
                    rhs_index - 1usize,
                );
                let right_b = RightBExpr::eval(
                    rhs_b_slot0,
                    rhs_b_slot1,
                    rhs_b_slot2,
                    rhs_b_slot3,
                    rhs_b_slot_offsets,
                    rhs_index - 1usize,
                );
                let left_a = LeftAExpr::eval(
                    lhs_a_slot0,
                    lhs_a_slot1,
                    lhs_a_slot2,
                    lhs_a_slot3,
                    lhs_a_slot_offsets,
                    mid,
                );
                let left_b = LeftBExpr::eval(
                    lhs_b_slot0,
                    lhs_b_slot1,
                    lhs_b_slot2,
                    lhs_b_slot3,
                    lhs_b_slot_offsets,
                    mid,
                );
                if !Less::apply((right_a, right_b), (left_a, left_b)) {
                    low.store(mid + 1usize);
                } else {
                    high.store(mid);
                }
            } else {
                high.store(mid);
            }
        }

        let lhs_index = low.read();
        let rhs_index = out - lhs_index;
        if lhs_index < lhs_logical_len {
            let left_a = LeftAExpr::eval(
                lhs_a_slot0,
                lhs_a_slot1,
                lhs_a_slot2,
                lhs_a_slot3,
                lhs_a_slot_offsets,
                lhs_index,
            );
            let left_b = LeftBExpr::eval(
                lhs_b_slot0,
                lhs_b_slot1,
                lhs_b_slot2,
                lhs_b_slot3,
                lhs_b_slot_offsets,
                lhs_index,
            );
            if rhs_index >= rhs_logical_len {
                out_a[out] = left_a;
                out_b[out] = left_b;
                source_sides[out] = 0u32;
                source_indices[out] = lhs_index as u32;
            } else {
                let right_a = RightAExpr::eval(
                    rhs_a_slot0,
                    rhs_a_slot1,
                    rhs_a_slot2,
                    rhs_a_slot3,
                    rhs_a_slot_offsets,
                    rhs_index,
                );
                let right_b = RightBExpr::eval(
                    rhs_b_slot0,
                    rhs_b_slot1,
                    rhs_b_slot2,
                    rhs_b_slot3,
                    rhs_b_slot_offsets,
                    rhs_index,
                );
                if !Less::apply((right_a, right_b), (left_a, left_b)) {
                    out_a[out] = left_a;
                    out_b[out] = left_b;
                    source_sides[out] = 0u32;
                    source_indices[out] = lhs_index as u32;
                } else {
                    out_a[out] = right_a;
                    out_b[out] = right_b;
                    source_sides[out] = 1u32;
                    source_indices[out] = rhs_index as u32;
                }
            }
        } else {
            out_a[out] = RightAExpr::eval(
                rhs_a_slot0,
                rhs_a_slot1,
                rhs_a_slot2,
                rhs_a_slot3,
                rhs_a_slot_offsets,
                rhs_index,
            );
            out_b[out] = RightBExpr::eval(
                rhs_b_slot0,
                rhs_b_slot1,
                rhs_b_slot2,
                rhs_b_slot3,
                rhs_b_slot_offsets,
                rhs_index,
            );
            source_sides[out] = 1u32;
            source_indices[out] = rhs_index as u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_tuple3_by_key_control_device_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    LeftAExpr: DeviceGpuExpr<A>,
    LeftBExpr: DeviceGpuExpr<B>,
    LeftCExpr: DeviceGpuExpr<C>,
    RightAExpr: DeviceGpuExpr<A>,
    RightBExpr: DeviceGpuExpr<B>,
    RightCExpr: DeviceGpuExpr<C>,
    Less: BinaryPredicateOp<(A, B, C)>,
>(
    lhs_a_slot0: &[A],
    lhs_a_slot1: &[A],
    lhs_a_slot2: &[A],
    lhs_a_slot3: &[A],
    lhs_a_slot_offsets: &[u32],
    lhs_b_slot0: &[B],
    lhs_b_slot1: &[B],
    lhs_b_slot2: &[B],
    lhs_b_slot3: &[B],
    lhs_b_slot_offsets: &[u32],
    lhs_c_slot0: &[C],
    lhs_c_slot1: &[C],
    lhs_c_slot2: &[C],
    lhs_c_slot3: &[C],
    lhs_c_slot_offsets: &[u32],
    lhs_len: &[u32],
    rhs_a_slot0: &[A],
    rhs_a_slot1: &[A],
    rhs_a_slot2: &[A],
    rhs_a_slot3: &[A],
    rhs_a_slot_offsets: &[u32],
    rhs_b_slot0: &[B],
    rhs_b_slot1: &[B],
    rhs_b_slot2: &[B],
    rhs_b_slot3: &[B],
    rhs_b_slot_offsets: &[u32],
    rhs_c_slot0: &[C],
    rhs_c_slot1: &[C],
    rhs_c_slot2: &[C],
    rhs_c_slot3: &[C],
    rhs_c_slot_offsets: &[u32],
    rhs_len: &[u32],
    out_a: &mut [A],
    out_b: &mut [B],
    out_c: &mut [C],
    source_sides: &mut [u32],
    source_indices: &mut [u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let lhs_logical_len = lhs_len[0] as usize;
    let rhs_logical_len = rhs_len[0] as usize;
    if out < out_a.len() {
        let low_init = RuntimeCell::<usize>::new(0usize);
        if out > rhs_logical_len {
            low_init.store(out - rhs_logical_len);
        }

        let high_init = RuntimeCell::<usize>::new(out);
        if high_init.read() > lhs_logical_len {
            high_init.store(lhs_logical_len);
        }

        let low = RuntimeCell::<usize>::new(low_init.read());
        let high = RuntimeCell::<usize>::new(high_init.read());
        while low.read() < high.read() {
            let mid = (low.read() + high.read()) / 2usize;
            let rhs_index = out - mid;
            if mid < lhs_logical_len && rhs_index > 0usize {
                let right_a = RightAExpr::eval(
                    rhs_a_slot0,
                    rhs_a_slot1,
                    rhs_a_slot2,
                    rhs_a_slot3,
                    rhs_a_slot_offsets,
                    rhs_index - 1usize,
                );
                let right_b = RightBExpr::eval(
                    rhs_b_slot0,
                    rhs_b_slot1,
                    rhs_b_slot2,
                    rhs_b_slot3,
                    rhs_b_slot_offsets,
                    rhs_index - 1usize,
                );
                let right_c = RightCExpr::eval(
                    rhs_c_slot0,
                    rhs_c_slot1,
                    rhs_c_slot2,
                    rhs_c_slot3,
                    rhs_c_slot_offsets,
                    rhs_index - 1usize,
                );
                let left_a = LeftAExpr::eval(
                    lhs_a_slot0,
                    lhs_a_slot1,
                    lhs_a_slot2,
                    lhs_a_slot3,
                    lhs_a_slot_offsets,
                    mid,
                );
                let left_b = LeftBExpr::eval(
                    lhs_b_slot0,
                    lhs_b_slot1,
                    lhs_b_slot2,
                    lhs_b_slot3,
                    lhs_b_slot_offsets,
                    mid,
                );
                let left_c = LeftCExpr::eval(
                    lhs_c_slot0,
                    lhs_c_slot1,
                    lhs_c_slot2,
                    lhs_c_slot3,
                    lhs_c_slot_offsets,
                    mid,
                );
                if !Less::apply((right_a, right_b, right_c), (left_a, left_b, left_c)) {
                    low.store(mid + 1usize);
                } else {
                    high.store(mid);
                }
            } else {
                high.store(mid);
            }
        }

        let lhs_index = low.read();
        let rhs_index = out - lhs_index;
        if lhs_index < lhs_logical_len {
            let left_a = LeftAExpr::eval(
                lhs_a_slot0,
                lhs_a_slot1,
                lhs_a_slot2,
                lhs_a_slot3,
                lhs_a_slot_offsets,
                lhs_index,
            );
            let left_b = LeftBExpr::eval(
                lhs_b_slot0,
                lhs_b_slot1,
                lhs_b_slot2,
                lhs_b_slot3,
                lhs_b_slot_offsets,
                lhs_index,
            );
            let left_c = LeftCExpr::eval(
                lhs_c_slot0,
                lhs_c_slot1,
                lhs_c_slot2,
                lhs_c_slot3,
                lhs_c_slot_offsets,
                lhs_index,
            );
            if rhs_index >= rhs_logical_len {
                out_a[out] = left_a;
                out_b[out] = left_b;
                out_c[out] = left_c;
                source_sides[out] = 0u32;
                source_indices[out] = lhs_index as u32;
            } else {
                let right_a = RightAExpr::eval(
                    rhs_a_slot0,
                    rhs_a_slot1,
                    rhs_a_slot2,
                    rhs_a_slot3,
                    rhs_a_slot_offsets,
                    rhs_index,
                );
                let right_b = RightBExpr::eval(
                    rhs_b_slot0,
                    rhs_b_slot1,
                    rhs_b_slot2,
                    rhs_b_slot3,
                    rhs_b_slot_offsets,
                    rhs_index,
                );
                let right_c = RightCExpr::eval(
                    rhs_c_slot0,
                    rhs_c_slot1,
                    rhs_c_slot2,
                    rhs_c_slot3,
                    rhs_c_slot_offsets,
                    rhs_index,
                );
                if !Less::apply((right_a, right_b, right_c), (left_a, left_b, left_c)) {
                    out_a[out] = left_a;
                    out_b[out] = left_b;
                    out_c[out] = left_c;
                    source_sides[out] = 0u32;
                    source_indices[out] = lhs_index as u32;
                } else {
                    out_a[out] = right_a;
                    out_b[out] = right_b;
                    out_c[out] = right_c;
                    source_sides[out] = 1u32;
                    source_indices[out] = rhs_index as u32;
                }
            }
        } else {
            out_a[out] = RightAExpr::eval(
                rhs_a_slot0,
                rhs_a_slot1,
                rhs_a_slot2,
                rhs_a_slot3,
                rhs_a_slot_offsets,
                rhs_index,
            );
            out_b[out] = RightBExpr::eval(
                rhs_b_slot0,
                rhs_b_slot1,
                rhs_b_slot2,
                rhs_b_slot3,
                rhs_b_slot_offsets,
                rhs_index,
            );
            out_c[out] = RightCExpr::eval(
                rhs_c_slot0,
                rhs_c_slot1,
                rhs_c_slot2,
                rhs_c_slot3,
                rhs_c_slot_offsets,
                rhs_index,
            );
            source_sides[out] = 1u32;
            source_indices[out] = rhs_index as u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_by_key_values_from_control_device_expr_kernel<
    T: CubePrimitive,
    LeftExpr: DeviceGpuExpr<T>,
    RightExpr: DeviceGpuExpr<T>,
>(
    lhs_slot0: &[T],
    lhs_slot1: &[T],
    lhs_slot2: &[T],
    lhs_slot3: &[T],
    lhs_slot_offsets: &[u32],
    rhs_slot0: &[T],
    rhs_slot1: &[T],
    rhs_slot2: &[T],
    rhs_slot3: &[T],
    rhs_slot_offsets: &[u32],
    source_sides: &[u32],
    source_indices: &[u32],
    len: &[u32],
    output_offset: &[u32],
    out_values: &mut [T],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < len[0] as usize {
        let index = source_indices[out] as usize;
        if source_sides[out] == 0u32 {
            out_values[output_offset[0] as usize + out] = LeftExpr::eval(
                lhs_slot0,
                lhs_slot1,
                lhs_slot2,
                lhs_slot3,
                lhs_slot_offsets,
                index,
            );
        } else {
            out_values[output_offset[0] as usize + out] = RightExpr::eval(
                rhs_slot0,
                rhs_slot1,
                rhs_slot2,
                rhs_slot3,
                rhs_slot_offsets,
                index,
            );
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_pass_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &[T],
    width: &[u32],
    output: &mut [T],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < input.len() {
        let run = width[0] as usize;
        let pair_width = run * 2usize;
        let pair_start = (out / pair_width) * pair_width;
        let left_start = pair_start;
        let left_len = RuntimeCell::<usize>::new(run);
        if left_start + left_len.read() > input.len() {
            left_len.store(input.len() - left_start);
        }

        let right_start = left_start + left_len.read();
        let right_len = RuntimeCell::<usize>::new(0usize);
        if right_start < input.len() {
            right_len.store(run);
            if right_start + right_len.read() > input.len() {
                right_len.store(input.len() - right_start);
            }
        }

        if right_len.read() == 0usize {
            output[out] = input[out];
        } else {
            let local_out = out - pair_start;
            let low_init = RuntimeCell::<usize>::new(0usize);
            if local_out > right_len.read() {
                low_init.store(local_out - right_len.read());
            }

            let high_init = RuntimeCell::<usize>::new(local_out);
            if high_init.read() > left_len.read() {
                high_init.store(left_len.read());
            }

            let low = RuntimeCell::<usize>::new(low_init.read());
            let high = RuntimeCell::<usize>::new(high_init.read());
            while low.read() < high.read() {
                let mid = (low.read() + high.read()) / 2usize;
                let rhs_index = local_out - mid;
                if mid < left_len.read()
                    && rhs_index > 0usize
                    && !Less::apply(
                        input[right_start + rhs_index - 1usize],
                        input[left_start + mid],
                    )
                {
                    low.store(mid + 1usize);
                } else {
                    high.store(mid);
                }
            }

            let lhs_index = low.read();
            let rhs_index = local_out - lhs_index;
            if lhs_index < left_len.read()
                && (rhs_index >= right_len.read()
                    || !Less::apply(
                        input[right_start + rhs_index],
                        input[left_start + lhs_index],
                    ))
            {
                output[out] = input[left_start + lhs_index];
            } else {
                output[out] = input[right_start + rhs_index];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_expr_first_pass_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let logical_len = len[0] as usize;
    if out < logical_len {
        let pair_start = (out / 2usize) * 2usize;
        let right = pair_start + 1usize;
        if right >= logical_len {
            output[out] = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, out);
        } else {
            let left_value = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, pair_start);
            let right_value = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, right);
            let take_right_first = Less::apply(right_value, left_value);
            if out == pair_start {
                if take_right_first {
                    output[out] = right_value;
                } else {
                    output[out] = left_value;
                }
            } else if take_right_first {
                output[out] = left_value;
            } else {
                output[out] = right_value;
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_by_key_expr_first_pass_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    ValueExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<K>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_slot_offsets: &[u32],
    len: &[u32],
    output_keys: &mut [K],
    output_values: &mut [T],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let logical_len = len[0] as usize;
    if out < logical_len {
        let pair_start = (out / 2usize) * 2usize;
        let right = pair_start + 1usize;
        if right >= logical_len {
            output_keys[out] = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                out,
            );
            output_values[out] = ValueExpr::eval(
                value_slot0,
                value_slot1,
                value_slot2,
                value_slot3,
                value_slot_offsets,
                out,
            );
        } else {
            let left_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                pair_start,
            );
            let left_value = ValueExpr::eval(
                value_slot0,
                value_slot1,
                value_slot2,
                value_slot3,
                value_slot_offsets,
                pair_start,
            );
            let right_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                right,
            );
            let right_value = ValueExpr::eval(
                value_slot0,
                value_slot1,
                value_slot2,
                value_slot3,
                value_slot_offsets,
                right,
            );
            let take_right_first = Less::apply(right_key, left_key);
            if out == pair_start {
                if take_right_first {
                    output_keys[out] = right_key;
                    output_values[out] = right_value;
                } else {
                    output_keys[out] = left_key;
                    output_values[out] = left_value;
                }
            } else if take_right_first {
                output_keys[out] = left_key;
                output_values[out] = left_value;
            } else {
                output_keys[out] = right_key;
                output_values[out] = right_value;
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple2_expr_first_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Less: BinaryPredicateOp<(A, B)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_slot_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let logical_len = len[0] as usize;
    if out < logical_len {
        let pair_start = (out / 2usize) * 2usize;
        let right = pair_start + 1usize;
        if right >= logical_len {
            output_a[out] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, out);
            output_b[out] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, out);
        } else {
            let left_a = ExprA::eval(
                a_slot0,
                a_slot1,
                a_slot2,
                a_slot3,
                a_slot_offsets,
                pair_start,
            );
            let left_b = ExprB::eval(
                b_slot0,
                b_slot1,
                b_slot2,
                b_slot3,
                b_slot_offsets,
                pair_start,
            );
            let right_a = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, right);
            let right_b = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, right);
            let take_right_first = Less::apply((right_a, right_b), (left_a, left_b));
            if out == pair_start {
                if take_right_first {
                    output_a[out] = right_a;
                    output_b[out] = right_b;
                } else {
                    output_a[out] = left_a;
                    output_b[out] = left_b;
                }
            } else if take_right_first {
                output_a[out] = left_a;
                output_b[out] = left_b;
            } else {
                output_a[out] = right_a;
                output_b[out] = right_b;
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple3_expr_first_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    Less: BinaryPredicateOp<(A, B, C)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_slot_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_slot_offsets: &[u32],
    c_slot0: &[C],
    c_slot1: &[C],
    c_slot2: &[C],
    c_slot3: &[C],
    c_slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let logical_len = len[0] as usize;
    if out < logical_len {
        let pair_start = (out / 2usize) * 2usize;
        let right = pair_start + 1usize;
        if right >= logical_len {
            output_a[out] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, out);
            output_b[out] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, out);
            output_c[out] = ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, out);
        } else {
            let left_a = ExprA::eval(
                a_slot0,
                a_slot1,
                a_slot2,
                a_slot3,
                a_slot_offsets,
                pair_start,
            );
            let left_b = ExprB::eval(
                b_slot0,
                b_slot1,
                b_slot2,
                b_slot3,
                b_slot_offsets,
                pair_start,
            );
            let left_c = ExprC::eval(
                c_slot0,
                c_slot1,
                c_slot2,
                c_slot3,
                c_slot_offsets,
                pair_start,
            );
            let right_a = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, right);
            let right_b = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, right);
            let right_c = ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, right);
            let take_right_first =
                Less::apply((right_a, right_b, right_c), (left_a, left_b, left_c));
            if out == pair_start {
                if take_right_first {
                    output_a[out] = right_a;
                    output_b[out] = right_b;
                    output_c[out] = right_c;
                } else {
                    output_a[out] = left_a;
                    output_b[out] = left_b;
                    output_c[out] = left_c;
                }
            } else if take_right_first {
                output_a[out] = left_a;
                output_b[out] = left_b;
                output_c[out] = left_c;
            } else {
                output_a[out] = right_a;
                output_b[out] = right_b;
                output_c[out] = right_c;
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple2_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    Less: BinaryPredicateOp<(A, B)>,
>(
    input_a: &[A],
    input_b: &[B],
    width: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < input_a.len() {
        let run = width[0] as usize;
        let pair_width = run * 2usize;
        let pair_start = (out / pair_width) * pair_width;
        let left_start = pair_start;
        let left_len = RuntimeCell::<usize>::new(run);
        if left_start + left_len.read() > input_a.len() {
            left_len.store(input_a.len() - left_start);
        }

        let right_start = left_start + left_len.read();
        let right_len = RuntimeCell::<usize>::new(0usize);
        if right_start < input_a.len() {
            right_len.store(run);
            if right_start + right_len.read() > input_a.len() {
                right_len.store(input_a.len() - right_start);
            }
        }

        if right_len.read() == 0usize {
            output_a[out] = input_a[out];
            output_b[out] = input_b[out];
        } else {
            let local_out = out - pair_start;
            let low_init = RuntimeCell::<usize>::new(0usize);
            if local_out > right_len.read() {
                low_init.store(local_out - right_len.read());
            }

            let high_init = RuntimeCell::<usize>::new(local_out);
            if high_init.read() > left_len.read() {
                high_init.store(left_len.read());
            }

            let low = RuntimeCell::<usize>::new(low_init.read());
            let high = RuntimeCell::<usize>::new(high_init.read());
            while low.read() < high.read() {
                let mid = (low.read() + high.read()) / 2usize;
                let rhs_index = local_out - mid;
                if mid < left_len.read()
                    && rhs_index > 0usize
                    && !Less::apply(
                        (
                            input_a[right_start + rhs_index - 1usize],
                            input_b[right_start + rhs_index - 1usize],
                        ),
                        (input_a[left_start + mid], input_b[left_start + mid]),
                    )
                {
                    low.store(mid + 1usize);
                } else {
                    high.store(mid);
                }
            }

            let lhs_index = low.read();
            let rhs_index = local_out - lhs_index;
            let take_left = RuntimeCell::<bool>::new(false);
            if lhs_index < left_len.read()
                && (rhs_index >= right_len.read()
                    || !Less::apply(
                        (
                            input_a[right_start + rhs_index],
                            input_b[right_start + rhs_index],
                        ),
                        (
                            input_a[left_start + lhs_index],
                            input_b[left_start + lhs_index],
                        ),
                    ))
            {
                take_left.store(true);
            }

            if take_left.read() {
                output_a[out] = input_a[left_start + lhs_index];
                output_b[out] = input_b[left_start + lhs_index];
            } else {
                output_a[out] = input_a[right_start + rhs_index];
                output_b[out] = input_b[right_start + rhs_index];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple3_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Less: BinaryPredicateOp<(A, B, C)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_c: &[C],
    width: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < input_a.len() {
        let run = width[0] as usize;
        let pair_width = run * 2usize;
        let pair_start = (out / pair_width) * pair_width;
        let left_start = pair_start;
        let left_len = RuntimeCell::<usize>::new(run);
        if left_start + left_len.read() > input_a.len() {
            left_len.store(input_a.len() - left_start);
        }

        let right_start = left_start + left_len.read();
        let right_len = RuntimeCell::<usize>::new(0usize);
        if right_start < input_a.len() {
            right_len.store(run);
            if right_start + right_len.read() > input_a.len() {
                right_len.store(input_a.len() - right_start);
            }
        }

        if right_len.read() == 0usize {
            output_a[out] = input_a[out];
            output_b[out] = input_b[out];
            output_c[out] = input_c[out];
        } else {
            let local_out = out - pair_start;
            let low_init = RuntimeCell::<usize>::new(0usize);
            if local_out > right_len.read() {
                low_init.store(local_out - right_len.read());
            }

            let high_init = RuntimeCell::<usize>::new(local_out);
            if high_init.read() > left_len.read() {
                high_init.store(left_len.read());
            }

            let low = RuntimeCell::<usize>::new(low_init.read());
            let high = RuntimeCell::<usize>::new(high_init.read());
            while low.read() < high.read() {
                let mid = (low.read() + high.read()) / 2usize;
                let rhs_index = local_out - mid;
                if mid < left_len.read()
                    && rhs_index > 0usize
                    && !Less::apply(
                        (
                            input_a[right_start + rhs_index - 1usize],
                            input_b[right_start + rhs_index - 1usize],
                            input_c[right_start + rhs_index - 1usize],
                        ),
                        (
                            input_a[left_start + mid],
                            input_b[left_start + mid],
                            input_c[left_start + mid],
                        ),
                    )
                {
                    low.store(mid + 1usize);
                } else {
                    high.store(mid);
                }
            }

            let lhs_index = low.read();
            let rhs_index = local_out - lhs_index;
            let take_left = RuntimeCell::<bool>::new(false);
            if lhs_index < left_len.read()
                && (rhs_index >= right_len.read()
                    || !Less::apply(
                        (
                            input_a[right_start + rhs_index],
                            input_b[right_start + rhs_index],
                            input_c[right_start + rhs_index],
                        ),
                        (
                            input_a[left_start + lhs_index],
                            input_b[left_start + lhs_index],
                            input_c[left_start + lhs_index],
                        ),
                    ))
            {
                take_left.store(true);
            }

            if take_left.read() {
                output_a[out] = input_a[left_start + lhs_index];
                output_b[out] = input_b[left_start + lhs_index];
                output_c[out] = input_c[left_start + lhs_index];
            } else {
                output_a[out] = input_a[right_start + rhs_index];
                output_b[out] = input_b[right_start + rhs_index];
                output_c[out] = input_c[right_start + rhs_index];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple3_by_key_expr_first_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    V: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    ExprV: DeviceGpuExpr<V>,
    Less: BinaryPredicateOp<(A, B, C)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_slot_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_slot_offsets: &[u32],
    c_slot0: &[C],
    c_slot1: &[C],
    c_slot2: &[C],
    c_slot3: &[C],
    c_slot_offsets: &[u32],
    v_slot0: &[V],
    v_slot1: &[V],
    v_slot2: &[V],
    v_slot3: &[V],
    v_slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_v: &mut [V],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let logical_len = len[0] as usize;
    if out < logical_len {
        let pair_start = (out / 2usize) * 2usize;
        let right = pair_start + 1usize;
        if right >= logical_len {
            output_a[out] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, out);
            output_b[out] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, out);
            output_c[out] = ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, out);
            output_v[out] = ExprV::eval(v_slot0, v_slot1, v_slot2, v_slot3, v_slot_offsets, out);
        } else {
            let left_a = ExprA::eval(
                a_slot0,
                a_slot1,
                a_slot2,
                a_slot3,
                a_slot_offsets,
                pair_start,
            );
            let left_b = ExprB::eval(
                b_slot0,
                b_slot1,
                b_slot2,
                b_slot3,
                b_slot_offsets,
                pair_start,
            );
            let left_c = ExprC::eval(
                c_slot0,
                c_slot1,
                c_slot2,
                c_slot3,
                c_slot_offsets,
                pair_start,
            );
            let left_v = ExprV::eval(
                v_slot0,
                v_slot1,
                v_slot2,
                v_slot3,
                v_slot_offsets,
                pair_start,
            );
            let right_a = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, right);
            let right_b = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, right);
            let right_c = ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, right);
            let right_v = ExprV::eval(v_slot0, v_slot1, v_slot2, v_slot3, v_slot_offsets, right);
            let take_right_first =
                Less::apply((right_a, right_b, right_c), (left_a, left_b, left_c));
            if out == pair_start {
                if take_right_first {
                    output_a[out] = right_a;
                    output_b[out] = right_b;
                    output_c[out] = right_c;
                    output_v[out] = right_v;
                } else {
                    output_a[out] = left_a;
                    output_b[out] = left_b;
                    output_c[out] = left_c;
                    output_v[out] = left_v;
                }
            } else if take_right_first {
                output_a[out] = left_a;
                output_b[out] = left_b;
                output_c[out] = left_c;
                output_v[out] = left_v;
            } else {
                output_a[out] = right_a;
                output_b[out] = right_b;
                output_c[out] = right_c;
                output_v[out] = right_v;
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple3_by_key_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    V: CubePrimitive,
    Less: BinaryPredicateOp<(A, B, C)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_c: &[C],
    input_v: &[V],
    width: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_v: &mut [V],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < input_a.len() {
        let run = width[0] as usize;
        let pair_width = run * 2usize;
        let pair_start = (out / pair_width) * pair_width;
        let left_start = pair_start;
        let left_len = RuntimeCell::<usize>::new(run);
        if left_start + left_len.read() > input_a.len() {
            left_len.store(input_a.len() - left_start);
        }

        let right_start = left_start + left_len.read();
        let right_len = RuntimeCell::<usize>::new(0usize);
        if right_start < input_a.len() {
            right_len.store(run);
            if right_start + right_len.read() > input_a.len() {
                right_len.store(input_a.len() - right_start);
            }
        }

        if right_len.read() == 0usize {
            output_a[out] = input_a[out];
            output_b[out] = input_b[out];
            output_c[out] = input_c[out];
            output_v[out] = input_v[out];
        } else {
            let local_out = out - pair_start;
            let low_init = RuntimeCell::<usize>::new(0usize);
            if local_out > right_len.read() {
                low_init.store(local_out - right_len.read());
            }

            let high_init = RuntimeCell::<usize>::new(local_out);
            if high_init.read() > left_len.read() {
                high_init.store(left_len.read());
            }

            let low = RuntimeCell::<usize>::new(low_init.read());
            let high = RuntimeCell::<usize>::new(high_init.read());
            while low.read() < high.read() {
                let mid = (low.read() + high.read()) / 2usize;
                let rhs_index = local_out - mid;
                if mid < left_len.read()
                    && rhs_index > 0usize
                    && !Less::apply(
                        (
                            input_a[right_start + rhs_index - 1usize],
                            input_b[right_start + rhs_index - 1usize],
                            input_c[right_start + rhs_index - 1usize],
                        ),
                        (
                            input_a[left_start + mid],
                            input_b[left_start + mid],
                            input_c[left_start + mid],
                        ),
                    )
                {
                    low.store(mid + 1usize);
                } else {
                    high.store(mid);
                }
            }

            let lhs_index = low.read();
            let rhs_index = local_out - lhs_index;
            let take_left = RuntimeCell::<bool>::new(false);
            if lhs_index < left_len.read()
                && (rhs_index >= right_len.read()
                    || !Less::apply(
                        (
                            input_a[right_start + rhs_index],
                            input_b[right_start + rhs_index],
                            input_c[right_start + rhs_index],
                        ),
                        (
                            input_a[left_start + lhs_index],
                            input_b[left_start + lhs_index],
                            input_c[left_start + lhs_index],
                        ),
                    ))
            {
                take_left.store(true);
            }

            if take_left.read() {
                output_a[out] = input_a[left_start + lhs_index];
                output_b[out] = input_b[left_start + lhs_index];
                output_c[out] = input_c[left_start + lhs_index];
                output_v[out] = input_v[left_start + lhs_index];
            } else {
                output_a[out] = input_a[right_start + rhs_index];
                output_b[out] = input_b[right_start + rhs_index];
                output_c[out] = input_c[right_start + rhs_index];
                output_v[out] = input_v[right_start + rhs_index];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple7_indices_expr_first_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    ExprD: DeviceGpuExpr<D>,
    ExprE: DeviceGpuExpr<E>,
    ExprF: DeviceGpuExpr<F>,
    ExprG: DeviceGpuExpr<G>,
    Less: BinaryPredicateOp<(A, B, C, D, E, F, G)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_slot_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_slot_offsets: &[u32],
    c_slot0: &[C],
    c_slot1: &[C],
    c_slot2: &[C],
    c_slot3: &[C],
    c_slot_offsets: &[u32],
    d_slot0: &[D],
    d_slot1: &[D],
    d_slot2: &[D],
    d_slot3: &[D],
    d_slot_offsets: &[u32],
    e_slot0: &[E],
    e_slot1: &[E],
    e_slot2: &[E],
    e_slot3: &[E],
    e_slot_offsets: &[u32],
    f_slot0: &[F],
    f_slot1: &[F],
    f_slot2: &[F],
    f_slot3: &[F],
    f_slot_offsets: &[u32],
    g_slot0: &[G],
    g_slot1: &[G],
    g_slot2: &[G],
    g_slot3: &[G],
    g_slot_offsets: &[u32],
    len: &[u32],
    output_indices: &mut [u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    let logical_len = len[0] as usize;
    if out < logical_len {
        let pair_start = (out / 2usize) * 2usize;
        let right = pair_start + 1usize;
        if right >= logical_len {
            output_indices[out] = out as u32;
        } else {
            let left = (
                ExprA::eval(
                    a_slot0,
                    a_slot1,
                    a_slot2,
                    a_slot3,
                    a_slot_offsets,
                    pair_start,
                ),
                ExprB::eval(
                    b_slot0,
                    b_slot1,
                    b_slot2,
                    b_slot3,
                    b_slot_offsets,
                    pair_start,
                ),
                ExprC::eval(
                    c_slot0,
                    c_slot1,
                    c_slot2,
                    c_slot3,
                    c_slot_offsets,
                    pair_start,
                ),
                ExprD::eval(
                    d_slot0,
                    d_slot1,
                    d_slot2,
                    d_slot3,
                    d_slot_offsets,
                    pair_start,
                ),
                ExprE::eval(
                    e_slot0,
                    e_slot1,
                    e_slot2,
                    e_slot3,
                    e_slot_offsets,
                    pair_start,
                ),
                ExprF::eval(
                    f_slot0,
                    f_slot1,
                    f_slot2,
                    f_slot3,
                    f_slot_offsets,
                    pair_start,
                ),
                ExprG::eval(
                    g_slot0,
                    g_slot1,
                    g_slot2,
                    g_slot3,
                    g_slot_offsets,
                    pair_start,
                ),
            );
            let right_value = (
                ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, right),
                ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, right),
                ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, right),
                ExprD::eval(d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets, right),
                ExprE::eval(e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets, right),
                ExprF::eval(f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets, right),
                ExprG::eval(g_slot0, g_slot1, g_slot2, g_slot3, g_slot_offsets, right),
            );
            let take_right_first = Less::apply(right_value, left);
            if out == pair_start {
                output_indices[out] = if take_right_first {
                    right as u32
                } else {
                    pair_start as u32
                };
            } else if take_right_first {
                output_indices[out] = pair_start as u32;
            } else {
                output_indices[out] = right as u32;
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple7_indices_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    ExprD: DeviceGpuExpr<D>,
    ExprE: DeviceGpuExpr<E>,
    ExprF: DeviceGpuExpr<F>,
    ExprG: DeviceGpuExpr<G>,
    Less: BinaryPredicateOp<(A, B, C, D, E, F, G)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_slot_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_slot_offsets: &[u32],
    c_slot0: &[C],
    c_slot1: &[C],
    c_slot2: &[C],
    c_slot3: &[C],
    c_slot_offsets: &[u32],
    d_slot0: &[D],
    d_slot1: &[D],
    d_slot2: &[D],
    d_slot3: &[D],
    d_slot_offsets: &[u32],
    e_slot0: &[E],
    e_slot1: &[E],
    e_slot2: &[E],
    e_slot3: &[E],
    e_slot_offsets: &[u32],
    f_slot0: &[F],
    f_slot1: &[F],
    f_slot2: &[F],
    f_slot3: &[F],
    f_slot_offsets: &[u32],
    g_slot0: &[G],
    g_slot1: &[G],
    g_slot2: &[G],
    g_slot3: &[G],
    g_slot_offsets: &[u32],
    input_indices: &[u32],
    width: &[u32],
    output_indices: &mut [u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < input_indices.len() {
        let run = width[0] as usize;
        let pair_width = run * 2usize;
        let pair_start = (out / pair_width) * pair_width;
        let left_start = pair_start;
        let left_len = RuntimeCell::<usize>::new(run);
        if left_start + left_len.read() > input_indices.len() {
            left_len.store(input_indices.len() - left_start);
        }

        let right_start = left_start + left_len.read();
        let right_len = RuntimeCell::<usize>::new(0usize);
        if right_start < input_indices.len() {
            right_len.store(run);
            if right_start + right_len.read() > input_indices.len() {
                right_len.store(input_indices.len() - right_start);
            }
        }

        if right_len.read() == 0usize {
            output_indices[out] = input_indices[out];
        } else {
            let local_out = out - pair_start;
            let low_init = RuntimeCell::<usize>::new(0usize);
            if local_out > right_len.read() {
                low_init.store(local_out - right_len.read());
            }
            let high_init = RuntimeCell::<usize>::new(local_out);
            if high_init.read() > left_len.read() {
                high_init.store(left_len.read());
            }

            let low = RuntimeCell::<usize>::new(low_init.read());
            let high = RuntimeCell::<usize>::new(high_init.read());
            while low.read() < high.read() {
                let mid = (low.read() + high.read()) / 2usize;
                let rhs_index = local_out - mid;
                if mid < left_len.read() && rhs_index > 0usize {
                    let r = input_indices[right_start + rhs_index - 1usize] as usize;
                    let l = input_indices[left_start + mid] as usize;
                    let right_value = (
                        ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, r),
                        ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, r),
                        ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, r),
                        ExprD::eval(d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets, r),
                        ExprE::eval(e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets, r),
                        ExprF::eval(f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets, r),
                        ExprG::eval(g_slot0, g_slot1, g_slot2, g_slot3, g_slot_offsets, r),
                    );
                    let left_value = (
                        ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, l),
                        ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, l),
                        ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, l),
                        ExprD::eval(d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets, l),
                        ExprE::eval(e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets, l),
                        ExprF::eval(f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets, l),
                        ExprG::eval(g_slot0, g_slot1, g_slot2, g_slot3, g_slot_offsets, l),
                    );
                    if !Less::apply(right_value, left_value) {
                        low.store(mid + 1usize);
                    } else {
                        high.store(mid);
                    }
                } else {
                    high.store(mid);
                }
            }

            let lhs_index = low.read();
            let rhs_index = local_out - lhs_index;
            let take_left = RuntimeCell::<bool>::new(false);
            if lhs_index < left_len.read() {
                if rhs_index >= right_len.read() {
                    take_left.store(true);
                } else {
                    let l = input_indices[left_start + lhs_index] as usize;
                    let r = input_indices[right_start + rhs_index] as usize;
                    let left_value = (
                        ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, l),
                        ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, l),
                        ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, l),
                        ExprD::eval(d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets, l),
                        ExprE::eval(e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets, l),
                        ExprF::eval(f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets, l),
                        ExprG::eval(g_slot0, g_slot1, g_slot2, g_slot3, g_slot_offsets, l),
                    );
                    let right_value = (
                        ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, r),
                        ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, r),
                        ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, r),
                        ExprD::eval(d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets, r),
                        ExprE::eval(e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets, r),
                        ExprF::eval(f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets, r),
                        ExprG::eval(g_slot0, g_slot1, g_slot2, g_slot3, g_slot_offsets, r),
                    );
                    if !Less::apply(right_value, left_value) {
                        take_left.store(true);
                    }
                }
            }

            if take_left.read() {
                output_indices[out] = input_indices[left_start + lhs_index];
            } else {
                output_indices[out] = input_indices[right_start + rhs_index];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_by_key_pass_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    Less: BinaryPredicateOp<K>,
>(
    input_keys: &[K],
    input_values: &[T],
    width: &[u32],
    output_keys: &mut [K],
    output_values: &mut [T],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < input_keys.len() {
        let run = width[0] as usize;
        let pair_width = run * 2usize;
        let pair_start = (out / pair_width) * pair_width;
        let left_start = pair_start;
        let left_len = RuntimeCell::<usize>::new(run);
        if left_start + left_len.read() > input_keys.len() {
            left_len.store(input_keys.len() - left_start);
        }

        let right_start = left_start + left_len.read();
        let right_len = RuntimeCell::<usize>::new(0usize);
        if right_start < input_keys.len() {
            right_len.store(run);
            if right_start + right_len.read() > input_keys.len() {
                right_len.store(input_keys.len() - right_start);
            }
        }

        if right_len.read() == 0usize {
            output_keys[out] = input_keys[out];
            output_values[out] = input_values[out];
        } else {
            let local_out = out - pair_start;
            let low_init = RuntimeCell::<usize>::new(0usize);
            if local_out > right_len.read() {
                low_init.store(local_out - right_len.read());
            }

            let high_init = RuntimeCell::<usize>::new(local_out);
            if high_init.read() > left_len.read() {
                high_init.store(left_len.read());
            }

            let low = RuntimeCell::<usize>::new(low_init.read());
            let high = RuntimeCell::<usize>::new(high_init.read());
            while low.read() < high.read() {
                let mid = (low.read() + high.read()) / 2usize;
                let rhs_index = local_out - mid;
                if mid < left_len.read()
                    && rhs_index > 0usize
                    && !Less::apply(
                        input_keys[right_start + rhs_index - 1usize],
                        input_keys[left_start + mid],
                    )
                {
                    low.store(mid + 1usize);
                } else {
                    high.store(mid);
                }
            }

            let lhs_index = low.read();
            let rhs_index = local_out - lhs_index;
            if lhs_index < left_len.read()
                && (rhs_index >= right_len.read()
                    || !Less::apply(
                        input_keys[right_start + rhs_index],
                        input_keys[left_start + lhs_index],
                    ))
            {
                output_keys[out] = input_keys[left_start + lhs_index];
                output_values[out] = input_values[left_start + lhs_index];
            } else {
                output_keys[out] = input_keys[right_start + rhs_index];
                output_values[out] = input_values[right_start + rhs_index];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn radix_digit_histogram_u32_kernel(
    input: &[u32],
    shift: &[u32],
    histograms: &mut [u32],
) {
    let local = UNIT_POS as usize;
    if local < 16usize {
        let blocks = histograms.len() / 16usize;
        let block_start = (CUBE_POS as usize) * 256usize;
        let block_end = block_start + 256usize;
        let count = RuntimeCell::<u32>::new(0u32);
        let index = RuntimeCell::<usize>::new(block_start);

        while index.read() < block_end && index.read() < input.len() {
            if ((input[index.read()] >> shift[0]) & 15u32) as usize == local {
                count.store(count.read() + 1u32);
            }
            index.store(index.read() + 1usize);
        }

        histograms[local * blocks + (CUBE_POS as usize)] = count.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn radix_digit_scatter_u32_kernel(
    input: &[u32],
    shift: &[u32],
    histograms: &[u32],
    histogram_prefixes: &[u32],
    output: &mut [u32],
) {
    let local = UNIT_POS as usize;
    let cube_dim = 256usize;
    let unit = (CUBE_POS as usize) * cube_dim + local;
    let mut digit_flags = Shared::<[u32]>::new_slice(4096usize);
    let current_digit = RuntimeCell::<usize>::new(16usize);

    if unit < input.len() {
        current_digit.store(((input[unit] >> shift[0]) & 15u32) as usize);
    }

    let digit_index = RuntimeCell::<usize>::new(0usize);
    while digit_index.read() < 16usize {
        let flag_index = digit_index.read() * cube_dim + local;
        if current_digit.read() == digit_index.read() {
            digit_flags[flag_index] = 1u32;
        } else {
            digit_flags[flag_index] = 0u32;
        }
        digit_index.store(digit_index.read() + 1usize);
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        digit_index.store(0usize);
        while digit_index.read() < 16usize {
            let flag_index = digit_index.read() * cube_dim + local;
            let addend = RuntimeCell::<u32>::new(0u32);
            if local >= stride.read() {
                addend.store(digit_flags[flag_index - stride.read()]);
            }
            sync_cube();
            if local >= stride.read() {
                digit_flags[flag_index] = digit_flags[flag_index] + addend.read();
            }
            sync_cube();
            digit_index.store(digit_index.read() + 1usize);
        }
        stride.store(stride.read() * 2usize);
    }

    if unit < input.len() {
        let digit = current_digit.read();
        let blocks = histograms.len() / 16usize;
        let histogram_index = digit * blocks + (CUBE_POS as usize);
        let local_rank = digit_flags[digit * cube_dim + local] - 1u32;
        let block_digit_start = histogram_prefixes[histogram_index] - histograms[histogram_index];
        output[(block_digit_start + local_rank) as usize] = input[unit];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn radix_digit_scatter_by_key_u32_kernel<T: CubePrimitive>(
    input_keys: &[u32],
    input_values: &[T],
    shift: &[u32],
    histograms: &[u32],
    histogram_prefixes: &[u32],
    output_keys: &mut [u32],
    output_values: &mut [T],
) {
    let local = UNIT_POS as usize;
    let cube_dim = 256usize;
    let unit = (CUBE_POS as usize) * cube_dim + local;
    let mut digit_flags = Shared::<[u32]>::new_slice(4096usize);
    let current_digit = RuntimeCell::<usize>::new(16usize);

    if unit < input_keys.len() {
        current_digit.store(((input_keys[unit] >> shift[0]) & 15u32) as usize);
    }

    let digit_index = RuntimeCell::<usize>::new(0usize);
    while digit_index.read() < 16usize {
        let flag_index = digit_index.read() * cube_dim + local;
        if current_digit.read() == digit_index.read() {
            digit_flags[flag_index] = 1u32;
        } else {
            digit_flags[flag_index] = 0u32;
        }
        digit_index.store(digit_index.read() + 1usize);
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        digit_index.store(0usize);
        while digit_index.read() < 16usize {
            let flag_index = digit_index.read() * cube_dim + local;
            let addend = RuntimeCell::<u32>::new(0u32);
            if local >= stride.read() {
                addend.store(digit_flags[flag_index - stride.read()]);
            }
            sync_cube();
            if local >= stride.read() {
                digit_flags[flag_index] = digit_flags[flag_index] + addend.read();
            }
            sync_cube();
            digit_index.store(digit_index.read() + 1usize);
        }
        stride.store(stride.read() * 2usize);
    }

    if unit < input_keys.len() {
        let digit = current_digit.read();
        let blocks = histograms.len() / 16usize;
        let histogram_index = digit * blocks + (CUBE_POS as usize);
        let local_rank = digit_flags[digit * cube_dim + local] - 1u32;
        let block_digit_start = histogram_prefixes[histogram_index] - histograms[histogram_index];
        let out_index = (block_digit_start + local_rank) as usize;
        output_keys[out_index] = input_keys[unit];
        output_values[out_index] = input_values[unit];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reverse_kernel<T: CubePrimitive>(input: &[T], output: &mut [T]) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < input.len() {
        output[unit] = input[input.len() - 1usize - unit];
    }
}
