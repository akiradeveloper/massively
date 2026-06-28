#![allow(dead_code)]

use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp},
    expr::DeviceGpuExpr,
};
use cubecl::prelude::*;

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_head_flags_device_expr_kernel<
    K: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    if global < logical_len {
        if global == 0usize {
            flags[global] = 1u32;
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            flags[global] = if KeyEq::apply(previous_key, current_key) {
                0u32
            } else {
                1u32
            };
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn inclusive_scan_by_flags_device_expr_kernel<
    T: CubePrimitive,
    ValueExpr: DeviceGpuExpr<T>,
    Op: BinaryOp<(T,)>,
>(
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_slot_offsets: &[u32],
    head_flags: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(ValueExpr::eval(
            value_slot0,
            value_slot1,
            value_slot2,
            value_slot3,
            value_slot_offsets,
            start.read(),
        ));
        let index = RuntimeCell::<usize>::new(start.read() + 1usize);
        while index.read() <= global {
            let value = Op::apply(
                (acc.read(),),
                (ValueExpr::eval(
                    value_slot0,
                    value_slot1,
                    value_slot2,
                    value_slot3,
                    value_slot_offsets,
                    index.read(),
                ),),
            );
            acc.store(value.0);
            index.store(index.read() + 1usize);
        }
        output[global] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_by_flags_device_expr_kernel<
    T: CubePrimitive,
    ValueExpr: DeviceGpuExpr<T>,
    Op: BinaryOp<(T,)>,
>(
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_slot_offsets: &[u32],
    head_flags: &[u32],
    init: &[T],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(init[0]);
        let index = RuntimeCell::<usize>::new(start.read());
        while index.read() < global {
            let value = Op::apply(
                (acc.read(),),
                (ValueExpr::eval(
                    value_slot0,
                    value_slot1,
                    value_slot2,
                    value_slot3,
                    value_slot_offsets,
                    index.read(),
                ),),
            );
            acc.store(value.0);
            index.store(index.read() + 1usize);
        }
        output[global] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn inclusive_scan_tuple2_by_flags_device_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp<(A, B)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_offsets: &[u32],
    head_flags: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc_a = RuntimeCell::<A>::new(ExprA::eval(
            a_slot0,
            a_slot1,
            a_slot2,
            a_slot3,
            a_offsets,
            start.read(),
        ));
        let acc_b = RuntimeCell::<B>::new(ExprB::eval(
            b_slot0,
            b_slot1,
            b_slot2,
            b_slot3,
            b_offsets,
            start.read(),
        ));
        let index = RuntimeCell::<usize>::new(start.read() + 1usize);
        while index.read() <= global {
            let value = Op::apply(
                (acc_a.read(), acc_b.read()),
                (
                    ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_offsets, index.read()),
                    ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_offsets, index.read()),
                ),
            );
            acc_a.store(value.0);
            acc_b.store(value.1);
            index.store(index.read() + 1usize);
        }
        output_a[global] = acc_a.read();
        output_b[global] = acc_b.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_tuple2_by_flags_device_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp<(A, B)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_offsets: &[u32],
    head_flags: &[u32],
    init_a: &[A],
    init_b: &[B],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc_a = RuntimeCell::<A>::new(init_a[0]);
        let acc_b = RuntimeCell::<B>::new(init_b[0]);
        let index = RuntimeCell::<usize>::new(start.read());
        while index.read() < global {
            let value = Op::apply(
                (acc_a.read(), acc_b.read()),
                (
                    ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_offsets, index.read()),
                    ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_offsets, index.read()),
                ),
            );
            acc_a.store(value.0);
            acc_b.store(value.1);
            index.store(index.read() + 1usize);
        }
        output_a[global] = acc_a.read();
        output_b[global] = acc_b.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn inclusive_scan_tuple3_by_flags_device_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    Op: BinaryOp<(A, B, C)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_offsets: &[u32],
    c_slot0: &[C],
    c_slot1: &[C],
    c_slot2: &[C],
    c_slot3: &[C],
    c_offsets: &[u32],
    head_flags: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc_a = RuntimeCell::<A>::new(ExprA::eval(
            a_slot0,
            a_slot1,
            a_slot2,
            a_slot3,
            a_offsets,
            start.read(),
        ));
        let acc_b = RuntimeCell::<B>::new(ExprB::eval(
            b_slot0,
            b_slot1,
            b_slot2,
            b_slot3,
            b_offsets,
            start.read(),
        ));
        let acc_c = RuntimeCell::<C>::new(ExprC::eval(
            c_slot0,
            c_slot1,
            c_slot2,
            c_slot3,
            c_offsets,
            start.read(),
        ));
        let index = RuntimeCell::<usize>::new(start.read() + 1usize);
        while index.read() <= global {
            let value = Op::apply(
                (acc_a.read(), acc_b.read(), acc_c.read()),
                (
                    ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_offsets, index.read()),
                    ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_offsets, index.read()),
                    ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_offsets, index.read()),
                ),
            );
            acc_a.store(value.0);
            acc_b.store(value.1);
            acc_c.store(value.2);
            index.store(index.read() + 1usize);
        }
        output_a[global] = acc_a.read();
        output_b[global] = acc_b.read();
        output_c[global] = acc_c.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_tuple3_by_flags_device_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    Op: BinaryOp<(A, B, C)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_offsets: &[u32],
    b_slot0: &[B],
    b_slot1: &[B],
    b_slot2: &[B],
    b_slot3: &[B],
    b_offsets: &[u32],
    c_slot0: &[C],
    c_slot1: &[C],
    c_slot2: &[C],
    c_slot3: &[C],
    c_offsets: &[u32],
    head_flags: &[u32],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc_a = RuntimeCell::<A>::new(init_a[0]);
        let acc_b = RuntimeCell::<B>::new(init_b[0]);
        let acc_c = RuntimeCell::<C>::new(init_c[0]);
        let index = RuntimeCell::<usize>::new(start.read());
        while index.read() < global {
            let value = Op::apply(
                (acc_a.read(), acc_b.read(), acc_c.read()),
                (
                    ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_offsets, index.read()),
                    ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_offsets, index.read()),
                    ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_offsets, index.read()),
                ),
            );
            acc_a.store(value.0);
            acc_b.store(value.1);
            acc_c.store(value.2);
            index.store(index.read() + 1usize);
        }
        output_a[global] = acc_a.read();
        output_b[global] = acc_b.read();
        output_c[global] = acc_c.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn inclusive_scan_tuple7_by_flags_view_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_c: &[C],
    input_d: &[D],
    input_e: &[E],
    input_f: &[F],
    input_g: &[G],
    offsets: &[u32],
    head_flags: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_d: &mut [D],
    output_e: &mut [E],
    output_f: &mut [F],
    output_g: &mut [G],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc_a = RuntimeCell::<A>::new(input_a[offsets[0] as usize + start.read()]);
        let acc_b = RuntimeCell::<B>::new(input_b[offsets[1] as usize + start.read()]);
        let acc_c = RuntimeCell::<C>::new(input_c[offsets[2] as usize + start.read()]);
        let acc_d = RuntimeCell::<D>::new(input_d[offsets[3] as usize + start.read()]);
        let acc_e = RuntimeCell::<E>::new(input_e[offsets[4] as usize + start.read()]);
        let acc_f = RuntimeCell::<F>::new(input_f[offsets[5] as usize + start.read()]);
        let acc_g = RuntimeCell::<G>::new(input_g[offsets[6] as usize + start.read()]);
        let index = RuntimeCell::<usize>::new(start.read() + 1usize);
        while index.read() <= global {
            let value = Op::apply(
                (
                    acc_a.read(),
                    acc_b.read(),
                    acc_c.read(),
                    acc_d.read(),
                    acc_e.read(),
                    acc_f.read(),
                    acc_g.read(),
                ),
                (
                    input_a[offsets[0] as usize + index.read()],
                    input_b[offsets[1] as usize + index.read()],
                    input_c[offsets[2] as usize + index.read()],
                    input_d[offsets[3] as usize + index.read()],
                    input_e[offsets[4] as usize + index.read()],
                    input_f[offsets[5] as usize + index.read()],
                    input_g[offsets[6] as usize + index.read()],
                ),
            );
            acc_a.store(value.0);
            acc_b.store(value.1);
            acc_c.store(value.2);
            acc_d.store(value.3);
            acc_e.store(value.4);
            acc_f.store(value.5);
            acc_g.store(value.6);
            index.store(index.read() + 1usize);
        }
        output_a[global] = acc_a.read();
        output_b[global] = acc_b.read();
        output_c[global] = acc_c.read();
        output_d[global] = acc_d.read();
        output_e[global] = acc_e.read();
        output_f[global] = acc_f.read();
        output_g[global] = acc_g.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_tuple7_by_flags_view_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_c: &[C],
    input_d: &[D],
    input_e: &[E],
    input_f: &[F],
    input_g: &[G],
    offsets: &[u32],
    head_flags: &[u32],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    init_d: &[D],
    init_e: &[E],
    init_f: &[F],
    init_g: &[G],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_d: &mut [D],
    output_e: &mut [E],
    output_f: &mut [F],
    output_g: &mut [G],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && head_flags[start.read()] == 0u32 {
            start.store(start.read() - 1usize);
        }

        let acc_a = RuntimeCell::<A>::new(init_a[0]);
        let acc_b = RuntimeCell::<B>::new(init_b[0]);
        let acc_c = RuntimeCell::<C>::new(init_c[0]);
        let acc_d = RuntimeCell::<D>::new(init_d[0]);
        let acc_e = RuntimeCell::<E>::new(init_e[0]);
        let acc_f = RuntimeCell::<F>::new(init_f[0]);
        let acc_g = RuntimeCell::<G>::new(init_g[0]);
        let index = RuntimeCell::<usize>::new(start.read());
        while index.read() < global {
            let value = Op::apply(
                (
                    acc_a.read(),
                    acc_b.read(),
                    acc_c.read(),
                    acc_d.read(),
                    acc_e.read(),
                    acc_f.read(),
                    acc_g.read(),
                ),
                (
                    input_a[offsets[0] as usize + index.read()],
                    input_b[offsets[1] as usize + index.read()],
                    input_c[offsets[2] as usize + index.read()],
                    input_d[offsets[3] as usize + index.read()],
                    input_e[offsets[4] as usize + index.read()],
                    input_f[offsets[5] as usize + index.read()],
                    input_g[offsets[6] as usize + index.read()],
                ),
            );
            acc_a.store(value.0);
            acc_b.store(value.1);
            acc_c.store(value.2);
            acc_d.store(value.3);
            acc_e.store(value.4);
            acc_f.store(value.5);
            acc_g.store(value.6);
            index.store(index.read() + 1usize);
        }
        output_a[global] = acc_a.read();
        output_b[global] = acc_b.read();
        output_c[global] = acc_c.read();
        output_d[global] = acc_d.read();
        output_e[global] = acc_e.read();
        output_f[global] = acc_f.read();
        output_g[global] = acc_g.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_pass_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    input: &[T],
    offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        let step = offset[0] as usize;
        if global >= step && KeyEq::apply(keys[global - step], keys[global]) {
            output[global] = Op::apply(input[global - step], input[global]);
        } else {
            output[global] = input[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_block_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    input: &[T],
    len: &[u32],
    output: &mut [T],
    block_tail_keys: &mut [K],
    block_tail_values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = Shared::<[T]>::new_slice(cube_dim);
    let mut heads = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values[unit] = input[global];
        valid[unit] = 1u32;
        if unit == 0usize || !KeyEq::apply(keys[global - 1usize], keys[global]) {
            heads[unit] = 1u32;
        } else {
            heads[unit] = 0u32;
        }
    } else {
        valid[unit] = 0u32;
        heads[unit] = 1u32;
        if logical_len > 0usize {
            values[unit] = input[0];
        }
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend = RuntimeCell::<T>::new(values[unit]);
        let addend_head = RuntimeCell::<u32>::new(0u32);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() {
            addend.store(values[unit - stride.read()]);
            addend_head.store(heads[unit - stride.read()]);
            addend_valid.store(valid[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
            if heads[unit] == 0u32 {
                values[unit] = Op::apply(addend.read(), values[unit]);
            }
            heads[unit] = heads[unit] | addend_head.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output[global] = values[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_tail_keys[block] = keys[global];
            block_tail_values[block] = values[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_device_expr_block_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    ValueExpr: DeviceGpuExpr<T>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
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
    output: &mut [T],
    block_tail_keys: &mut [K],
    block_tail_values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = Shared::<[T]>::new_slice(cube_dim);
    let mut heads = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values[unit] = ValueExpr::eval(
            value_slot0,
            value_slot1,
            value_slot2,
            value_slot3,
            value_slot_offsets,
            global,
        );
        valid[unit] = 1u32;
        if unit == 0usize {
            heads[unit] = 1u32;
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                heads[unit] = 0u32;
            } else {
                heads[unit] = 1u32;
            }
        }
    } else {
        valid[unit] = 0u32;
        heads[unit] = 1u32;
        if logical_len > 0usize {
            values[unit] = ValueExpr::eval(
                value_slot0,
                value_slot1,
                value_slot2,
                value_slot3,
                value_slot_offsets,
                0usize,
            );
        }
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend = RuntimeCell::<T>::new(values[unit]);
        let addend_head = RuntimeCell::<u32>::new(0u32);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() {
            addend.store(values[unit - stride.read()]);
            addend_head.store(heads[unit - stride.read()]);
            addend_valid.store(valid[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
            if heads[unit] == 0u32 {
                values[unit] = Op::apply(addend.read(), values[unit]);
            }
            heads[unit] = heads[unit] | addend_head.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output[global] = values[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_tail_keys[block] = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            block_tail_values[block] = values[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_add_block_prefix_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    block_tail_keys: &[K],
    block_prefixes: &[T],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize || KeyEq::apply(keys[global - 1usize], keys[global]) {
            first_segment[unit] = 1u32;
        } else {
            first_segment[unit] = 0u32;
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if block > 0usize
        && global < logical_len
        && first_segment[unit] != 0u32
        && KeyEq::apply(block_tail_keys[block - 1usize], keys[block * cube_dim])
    {
        output[global] = Op::apply(block_prefixes[block - 1usize], output[global]);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_device_expr_add_block_prefix_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    block_tail_keys: &[K],
    block_prefixes: &[T],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize {
            first_segment[unit] = 1u32;
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                first_segment[unit] = 1u32;
            } else {
                first_segment[unit] = 0u32;
            }
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if block > 0usize && global < logical_len && first_segment[unit] != 0u32 {
        let block_first_key = KeyExpr::eval(
            key_slot0,
            key_slot1,
            key_slot2,
            key_slot3,
            key_slot_offsets,
            block * cube_dim,
        );
        if KeyEq::apply(block_tail_keys[block - 1usize], block_first_key) {
            output[global] = Op::apply(block_prefixes[block - 1usize], output[global]);
        }
    }
}

macro_rules! define_tuple_by_key_block_scan_kernels {
    (
        $block_name:ident,
        $add_prefix_name:ident,
        ( $( $ty:ident: $key:ident: $tail:ident ),+ )
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $block_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &[$ty], )+
            input: &[T],
            len: &[u32],
            output: &mut [T],
            $( $tail: &mut [$ty], )+
            block_tail_values: &mut [T],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let block = CUBE_POS as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            let mut values = Shared::<[T]>::new_slice(cube_dim);
            let mut heads = Shared::<[u32]>::new_slice(cube_dim);
            let mut valid = Shared::<[u32]>::new_slice(cube_dim);

            if global < logical_len {
                values[unit] = input[global];
                valid[unit] = 1u32;
                if unit == 0usize || !KeyEq::apply(
                    ($( $key[global - 1usize] ),+),
                    ($( $key[global] ),+),
                ) {
                    heads[unit] = 1u32;
                } else {
                    heads[unit] = 0u32;
                }
            } else {
                valid[unit] = 0u32;
                heads[unit] = 1u32;
                if logical_len > 0usize {
                    values[unit] = input[0];
                }
            }
            sync_cube();

            let stride = RuntimeCell::<usize>::new(1usize);
            while stride.read() < cube_dim {
                let addend = RuntimeCell::<T>::new(values[unit]);
                let addend_head = RuntimeCell::<u32>::new(0u32);
                let addend_valid = RuntimeCell::<u32>::new(0u32);
                if unit >= stride.read() {
                    addend.store(values[unit - stride.read()]);
                    addend_head.store(heads[unit - stride.read()]);
                    addend_valid.store(valid[unit - stride.read()]);
                }
                sync_cube();
                if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
                    if heads[unit] == 0u32 {
                        values[unit] = Op::apply(addend.read(), values[unit]);
                    }
                    heads[unit] = heads[unit] | addend_head.read();
                }
                sync_cube();
                stride.store(stride.read() * 2usize);
            }

            if global < logical_len {
                output[global] = values[unit];
                if unit == cube_dim - 1usize || global == logical_len - 1usize {
                    $(
                        $tail[block] = $key[global];
                    )+
                    block_tail_values[block] = values[unit];
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $add_prefix_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &[$ty], )+
            $( $tail: &[$ty], )+
            block_prefixes: &[T],
            len: &[u32],
            output: &mut [T],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let block = CUBE_POS as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

            if global < logical_len {
                if unit == 0usize || KeyEq::apply(
                    ($( $key[global - 1usize] ),+),
                    ($( $key[global] ),+),
                ) {
                    first_segment[unit] = 1u32;
                } else {
                    first_segment[unit] = 0u32;
                }
            } else {
                first_segment[unit] = 0u32;
            }
            sync_cube();

            let stride = RuntimeCell::<usize>::new(1usize);
            while stride.read() < cube_dim {
                let previous = RuntimeCell::<u32>::new(first_segment[unit]);
                if unit >= stride.read() {
                    previous.store(first_segment[unit - stride.read()]);
                }
                sync_cube();
                if unit >= stride.read() {
                    first_segment[unit] = first_segment[unit] & previous.read();
                }
                sync_cube();
                stride.store(stride.read() * 2usize);
            }

            if block > 0usize
                && global < logical_len
                && first_segment[unit] != 0u32
                && KeyEq::apply(
                    ($( $tail[block - 1usize] ),+),
                    ($( $key[block * cube_dim] ),+),
                )
            {
                output[global] = Op::apply(block_prefixes[block - 1usize], output[global]);
            }
        }
    };
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_make_exclusive_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    inclusive: &[T],
    init: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output.len() {
        if global == 0usize || !KeyEq::apply(keys[global - 1usize], keys[global]) {
            output[global] = init[0];
        } else {
            output[global] = Op::apply(init[0], inclusive[global - 1usize]);
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_device_expr_make_exclusive_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    inclusive: &[T],
    init: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output.len() {
        if global == 0usize {
            output[global] = init[0];
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                output[global] = Op::apply(init[0], inclusive[global - 1usize]);
            } else {
                output[global] = init[0];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_tuple2_device_expr_block_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
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
    block_tail_keys: &mut [K],
    tail_a: &mut [A],
    tail_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut heads = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, global);
        values_b[unit] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, global);
        valid[unit] = 1u32;
        if unit == 0usize {
            heads[unit] = 1u32;
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                heads[unit] = 0u32;
            } else {
                heads[unit] = 1u32;
            }
        }
    } else {
        valid[unit] = 0u32;
        heads[unit] = 1u32;
        if logical_len > 0usize {
            values_a[unit] =
                ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, 0usize);
            values_b[unit] =
                ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, 0usize);
        }
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_b = RuntimeCell::<B>::new(values_b[unit]);
        let addend_head = RuntimeCell::<u32>::new(0u32);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_head.store(heads[unit - stride.read()]);
            addend_valid.store(valid[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
            if heads[unit] == 0u32 {
                let value = Op::apply(
                    (addend_a.read(), addend_b.read()),
                    (values_a[unit], values_b[unit]),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
            }
            heads[unit] = heads[unit] | addend_head.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        output_b[global] = values_b[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_tail_keys[block] = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            tail_a[block] = values_a[unit];
            tail_b[block] = values_b[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_tuple2_device_expr_add_block_prefix_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    block_tail_keys: &[K],
    prefix_a: &[A],
    prefix_b: &[B],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize {
            first_segment[unit] = 1u32;
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                first_segment[unit] = 1u32;
            } else {
                first_segment[unit] = 0u32;
            }
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if block > 0usize && global < logical_len && first_segment[unit] != 0u32 {
        let block_first_key = KeyExpr::eval(
            key_slot0,
            key_slot1,
            key_slot2,
            key_slot3,
            key_slot_offsets,
            block * cube_dim,
        );
        if KeyEq::apply(block_tail_keys[block - 1usize], block_first_key) {
            let value = Op::apply(
                (prefix_a[block - 1usize], prefix_b[block - 1usize]),
                (output_a[global], output_b[global]),
            );
            output_a[global] = value.0;
            output_b[global] = value.1;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_tuple2_device_expr_make_exclusive_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    inclusive_a: &[A],
    inclusive_b: &[B],
    init_a: &[A],
    init_b: &[B],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output_a.len() {
        if global == 0usize {
            output_a[global] = init_a[0];
            output_b[global] = init_b[0];
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                let value = Op::apply(
                    (init_a[0], init_b[0]),
                    (inclusive_a[global - 1usize], inclusive_b[global - 1usize]),
                );
                output_a[global] = value.0;
                output_b[global] = value.1;
            } else {
                output_a[global] = init_a[0];
                output_b[global] = init_b[0];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_tuple3_device_expr_block_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
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
    block_tail_keys: &mut [K],
    tail_a: &mut [A],
    tail_b: &mut [B],
    tail_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut heads = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, global);
        values_b[unit] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, global);
        values_c[unit] = ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, global);
        valid[unit] = 1u32;
        if unit == 0usize {
            heads[unit] = 1u32;
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                heads[unit] = 0u32;
            } else {
                heads[unit] = 1u32;
            }
        }
    } else {
        valid[unit] = 0u32;
        heads[unit] = 1u32;
        if logical_len > 0usize {
            values_a[unit] =
                ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, 0usize);
            values_b[unit] =
                ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, 0usize);
            values_c[unit] =
                ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, 0usize);
        }
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_b = RuntimeCell::<B>::new(values_b[unit]);
        let addend_c = RuntimeCell::<C>::new(values_c[unit]);
        let addend_head = RuntimeCell::<u32>::new(0u32);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_c.store(values_c[unit - stride.read()]);
            addend_head.store(heads[unit - stride.read()]);
            addend_valid.store(valid[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
            if heads[unit] == 0u32 {
                let value = Op::apply(
                    (addend_a.read(), addend_b.read(), addend_c.read()),
                    (values_a[unit], values_b[unit], values_c[unit]),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
                values_c[unit] = value.2;
            }
            heads[unit] = heads[unit] | addend_head.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        output_b[global] = values_b[unit];
        output_c[global] = values_c[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_tail_keys[block] = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            tail_a[block] = values_a[unit];
            tail_b[block] = values_b[unit];
            tail_c[block] = values_c[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_tuple3_device_expr_add_block_prefix_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    block_tail_keys: &[K],
    prefix_a: &[A],
    prefix_b: &[B],
    prefix_c: &[C],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize {
            first_segment[unit] = 1u32;
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                first_segment[unit] = 1u32;
            } else {
                first_segment[unit] = 0u32;
            }
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if block > 0usize && global < logical_len && first_segment[unit] != 0u32 {
        let block_first_key = KeyExpr::eval(
            key_slot0,
            key_slot1,
            key_slot2,
            key_slot3,
            key_slot_offsets,
            block * cube_dim,
        );
        if KeyEq::apply(block_tail_keys[block - 1usize], block_first_key) {
            let value = Op::apply(
                (
                    prefix_a[block - 1usize],
                    prefix_b[block - 1usize],
                    prefix_c[block - 1usize],
                ),
                (output_a[global], output_b[global], output_c[global]),
            );
            output_a[global] = value.0;
            output_b[global] = value.1;
            output_c[global] = value.2;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_tuple3_device_expr_make_exclusive_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    inclusive_a: &[A],
    inclusive_b: &[B],
    inclusive_c: &[C],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output_a.len() {
        if global == 0usize {
            output_a[global] = init_a[0];
            output_b[global] = init_b[0];
            output_c[global] = init_c[0];
        } else {
            let previous_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global - 1usize,
            );
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            if KeyEq::apply(previous_key, current_key) {
                let value = Op::apply(
                    (init_a[0], init_b[0], init_c[0]),
                    (
                        inclusive_a[global - 1usize],
                        inclusive_b[global - 1usize],
                        inclusive_c[global - 1usize],
                    ),
                );
                output_a[global] = value.0;
                output_b[global] = value.1;
                output_c[global] = value.2;
            } else {
                output_a[global] = init_a[0];
                output_b[global] = init_b[0];
                output_c[global] = init_c[0];
            }
        }
    }
}

macro_rules! define_tuple_value_by_key_scan_kernels {
    (
        $block_name:ident,
        $add_prefix_name:ident,
        ( $( $ty:ident: $input:ident: $output:ident: $shared:ident: $tail:ident: $prefix:ident: $init:tt: $addend:ident ),+ )
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $block_name<
            K: CubePrimitive,
            $( $ty: CubePrimitive, )+
            KeyEq: BinaryPredicateOp<K>,
            Op: BinaryOp<($( $ty ),+)>,
        >(
            keys: &[K],
            $( $input: &[$ty], )+
            len: &[u32],
            $( $output: &mut [$ty], )+
            block_tail_keys: &mut [K],
            $( $tail: &mut [$ty], )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let block = CUBE_POS as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            $(
                let mut $shared = Shared::<[$ty]>::new_slice(cube_dim);
            )+
            let mut heads = Shared::<[u32]>::new_slice(cube_dim);
            let mut valid = Shared::<[u32]>::new_slice(cube_dim);

            if global < logical_len {
                $(
                    $shared[unit] = $input[global];
                )+
                valid[unit] = 1u32;
                if unit == 0usize || !KeyEq::apply(keys[global - 1usize], keys[global]) {
                    heads[unit] = 1u32;
                } else {
                    heads[unit] = 0u32;
                }
            } else {
                valid[unit] = 0u32;
                heads[unit] = 1u32;
                if logical_len > 0usize {
                    $(
                        $shared[unit] = $input[0];
                    )+
                }
            }
            sync_cube();

            let stride = RuntimeCell::<usize>::new(1usize);
            while stride.read() < cube_dim {
                $(
                    let $addend = RuntimeCell::<$ty>::new($shared[unit]);
                )+
                let addend_head = RuntimeCell::<u32>::new(0u32);
                let addend_valid = RuntimeCell::<u32>::new(0u32);
                if unit >= stride.read() {
                    $(
                        $addend.store($shared[unit - stride.read()]);
                    )+
                    addend_head.store(heads[unit - stride.read()]);
                    addend_valid.store(valid[unit - stride.read()]);
                }
                sync_cube();
                if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
                    if heads[unit] == 0u32 {
                        let reduced = Op::apply(
                            ($( $addend.read() ),+),
                            ($( $shared[unit] ),+),
                        );
                        $(
                            $shared[unit] = reduced.$init;
                        )+
                    }
                    heads[unit] = heads[unit] | addend_head.read();
                }
                sync_cube();
                stride.store(stride.read() * 2usize);
            }

            if global < logical_len {
                $(
                    $output[global] = $shared[unit];
                )+
                if unit == cube_dim - 1usize || global == logical_len - 1usize {
                    block_tail_keys[block] = keys[global];
                    $(
                        $tail[block] = $shared[unit];
                    )+
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $add_prefix_name<
            K: CubePrimitive,
            $( $ty: CubePrimitive, )+
            KeyEq: BinaryPredicateOp<K>,
            Op: BinaryOp<($( $ty ),+)>,
        >(
            keys: &[K],
            block_tail_keys: &[K],
            $( $prefix: &[$ty], )+
            len: &[u32],
            $( $output: &mut [$ty], )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let block = CUBE_POS as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

            if global < logical_len {
                if unit == 0usize || KeyEq::apply(keys[global - 1usize], keys[global]) {
                    first_segment[unit] = 1u32;
                } else {
                    first_segment[unit] = 0u32;
                }
            } else {
                first_segment[unit] = 0u32;
            }
            sync_cube();

            let stride = RuntimeCell::<usize>::new(1usize);
            while stride.read() < cube_dim {
                let previous = RuntimeCell::<u32>::new(first_segment[unit]);
                if unit >= stride.read() {
                    previous.store(first_segment[unit - stride.read()]);
                }
                sync_cube();
                if unit >= stride.read() {
                    first_segment[unit] = first_segment[unit] & previous.read();
                }
                sync_cube();
                stride.store(stride.read() * 2usize);
            }

            if block > 0usize
                && global < logical_len
                && first_segment[unit] != 0u32
                && KeyEq::apply(block_tail_keys[block - 1usize], keys[block * cube_dim])
            {
                let reduced = Op::apply(
                    ($( $prefix[block - 1usize] ),+),
                    ($( $output[global] ),+),
                );
                $(
                    $output[global] = reduced.$init;
                )+
            }
        }

    };
}

define_tuple_value_by_key_scan_kernels!(
    scan_by_key_tuple2_block_kernel,
    scan_by_key_tuple2_add_block_prefix_kernel,
    (A: input_a: output_a: shared_a: block_tail_a: prefix_a: 0: addend_a, B: input_b: output_b: shared_b: block_tail_b: prefix_b: 1: addend_b)
);
define_tuple_value_by_key_scan_kernels!(
    scan_by_key_tuple3_block_kernel,
    scan_by_key_tuple3_add_block_prefix_kernel,
    (A: input_a: output_a: shared_a: block_tail_a: prefix_a: 0: addend_a, B: input_b: output_b: shared_b: block_tail_b: prefix_b: 1: addend_b, C: input_c: output_c: shared_c: block_tail_c: prefix_c: 2: addend_c)
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_end_flags_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    inclusive: &[T],
    init: &[T],
    flags: &mut [u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < keys.len() {
        if global + 1usize == keys.len() || !KeyEq::apply(keys[global], keys[global + 1usize]) {
            flags[global] = 1u32;
            values[global] = Op::apply(init[0], inclusive[global]);
        } else {
            flags[global] = 0u32;
            values[global] = inclusive[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_device_expr_key_end_flags_kernel<
    K: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    if global < logical_len {
        let is_end = RuntimeCell::<bool>::new(false);
        if global + 1usize == logical_len {
            is_end.store(true);
        } else {
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            let next_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global + 1usize,
            );
            if !KeyEq::apply(current_key, next_key) {
                is_end.store(true);
            }
        }

        flags[global] = if is_end.read() { 1u32 } else { 0u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn head_flags_to_end_flags_kernel(head_flags: &[u32], len: &[u32], flags: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    if global < logical_len {
        flags[global] = if global + 1usize == logical_len || head_flags[global + 1usize] != 0u32 {
            1u32
        } else {
            0u32
        };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_apply_init_kernel<T: CubePrimitive, Op: BinaryOp<(T,)>>(
    inclusive: &[T],
    init: &[T],
    len: &[u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        values[global] = Op::apply((init[0],), (inclusive[global],)).0;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_tuple2_apply_init_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    Op: BinaryOp<(A, B)>,
>(
    inclusive_a: &[A],
    inclusive_b: &[B],
    init_a: &[A],
    init_b: &[B],
    len: &[u32],
    values_a: &mut [A],
    values_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let value = Op::apply(
            (init_a[0], init_b[0]),
            (inclusive_a[global], inclusive_b[global]),
        );
        values_a[global] = value.0;
        values_b[global] = value.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_tuple3_apply_init_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: BinaryOp<(A, B, C)>,
>(
    inclusive_a: &[A],
    inclusive_b: &[B],
    inclusive_c: &[C],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    len: &[u32],
    values_a: &mut [A],
    values_b: &mut [B],
    values_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let value = Op::apply(
            (init_a[0], init_b[0], init_c[0]),
            (
                inclusive_a[global],
                inclusive_b[global],
                inclusive_c[global],
            ),
        );
        values_a[global] = value.0;
        values_b[global] = value.1;
        values_c[global] = value.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_tuple7_apply_init_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
>(
    inclusive_a: &[A],
    inclusive_b: &[B],
    inclusive_c: &[C],
    inclusive_d: &[D],
    inclusive_e: &[E],
    inclusive_f: &[F],
    inclusive_g: &[G],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    init_d: &[D],
    init_e: &[E],
    init_f: &[F],
    init_g: &[G],
    len: &[u32],
    values_a: &mut [A],
    values_b: &mut [B],
    values_c: &mut [C],
    values_d: &mut [D],
    values_e: &mut [E],
    values_f: &mut [F],
    values_g: &mut [G],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let value = Op::apply(
            (
                init_a[0], init_b[0], init_c[0], init_d[0], init_e[0], init_f[0], init_g[0],
            ),
            (
                inclusive_a[global],
                inclusive_b[global],
                inclusive_c[global],
                inclusive_d[global],
                inclusive_e[global],
                inclusive_f[global],
                inclusive_g[global],
            ),
        );
        values_a[global] = value.0;
        values_b[global] = value.1;
        values_c[global] = value.2;
        values_d[global] = value.3;
        values_e[global] = value.4;
        values_f[global] = value.5;
        values_g[global] = value.6;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_device_expr_end_flags_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    inclusive: &[T],
    init: &[T],
    len: &[u32],
    flags: &mut [u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    if global < logical_len {
        let is_end = RuntimeCell::<bool>::new(false);
        if global + 1usize == logical_len {
            is_end.store(true);
        } else {
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            let next_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global + 1usize,
            );
            if !KeyEq::apply(current_key, next_key) {
                is_end.store(true);
            }
        }

        if is_end.read() {
            flags[global] = 1u32;
            values[global] = Op::apply(init[0], inclusive[global]);
        } else {
            flags[global] = 0u32;
            values[global] = inclusive[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_tuple2_device_expr_end_flags_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    inclusive_a: &[A],
    inclusive_b: &[B],
    init_a: &[A],
    init_b: &[B],
    len: &[u32],
    flags: &mut [u32],
    values_a: &mut [A],
    values_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    if global < logical_len {
        let is_end = RuntimeCell::<bool>::new(false);
        if global + 1usize == logical_len {
            is_end.store(true);
        } else {
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            let next_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global + 1usize,
            );
            if !KeyEq::apply(current_key, next_key) {
                is_end.store(true);
            }
        }

        if is_end.read() {
            flags[global] = 1u32;
            let value = Op::apply(
                (init_a[0], init_b[0]),
                (inclusive_a[global], inclusive_b[global]),
            );
            values_a[global] = value.0;
            values_b[global] = value.1;
        } else {
            flags[global] = 0u32;
            values_a[global] = inclusive_a[global];
            values_b[global] = inclusive_b[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_tuple3_device_expr_end_flags_kernel<
    K: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    KeyExpr: DeviceGpuExpr<K>,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<(A, B, C)>,
>(
    key_slot0: &[K],
    key_slot1: &[K],
    key_slot2: &[K],
    key_slot3: &[K],
    key_slot_offsets: &[u32],
    inclusive_a: &[A],
    inclusive_b: &[B],
    inclusive_c: &[C],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    len: &[u32],
    flags: &mut [u32],
    values_a: &mut [A],
    values_b: &mut [B],
    values_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    if global < logical_len {
        let is_end = RuntimeCell::<bool>::new(false);
        if global + 1usize == logical_len {
            is_end.store(true);
        } else {
            let current_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global,
            );
            let next_key = KeyExpr::eval(
                key_slot0,
                key_slot1,
                key_slot2,
                key_slot3,
                key_slot_offsets,
                global + 1usize,
            );
            if !KeyEq::apply(current_key, next_key) {
                is_end.store(true);
            }
        }

        if is_end.read() {
            flags[global] = 1u32;
            let value = Op::apply(
                (init_a[0], init_b[0], init_c[0]),
                (
                    inclusive_a[global],
                    inclusive_b[global],
                    inclusive_c[global],
                ),
            );
            values_a[global] = value.0;
            values_b[global] = value.1;
            values_c[global] = value.2;
        } else {
            flags[global] = 0u32;
            values_a[global] = inclusive_a[global];
            values_b[global] = inclusive_b[global];
            values_c[global] = inclusive_c[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_end_flags_with_block_prefix_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    local_inclusive: &[T],
    block_tail_keys: &[K],
    block_prefixes: &[T],
    init: &[T],
    len: &[u32],
    flags: &mut [u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize || KeyEq::apply(keys[global - 1usize], keys[global]) {
            first_segment[unit] = 1u32;
        } else {
            first_segment[unit] = 0u32;
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        let is_end =
            global + 1usize == logical_len || !KeyEq::apply(keys[global], keys[global + 1usize]);
        if is_end {
            flags[global] = 1u32;
            let carry = block > 0usize
                && first_segment[unit] != 0u32
                && KeyEq::apply(block_tail_keys[block - 1usize], keys[block * cube_dim]);
            if carry {
                values[global] = Op::apply(
                    init[0],
                    Op::apply(block_prefixes[block - 1usize], local_inclusive[global]),
                );
            } else {
                values[global] = Op::apply(init[0], local_inclusive[global]);
            }
        } else {
            flags[global] = 0u32;
            values[global] = local_inclusive[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_values_at_ends_with_block_prefix_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    local_inclusive: &[T],
    block_tail_keys: &[K],
    block_prefixes: &[T],
    init: &[T],
    len: &[u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize || KeyEq::apply(keys[global - 1usize], keys[global]) {
            first_segment[unit] = 1u32;
        } else {
            first_segment[unit] = 0u32;
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        let is_end =
            global + 1usize == logical_len || !KeyEq::apply(keys[global], keys[global + 1usize]);
        if is_end {
            let carry = block > 0usize
                && first_segment[unit] != 0u32
                && KeyEq::apply(block_tail_keys[block - 1usize], keys[block * cube_dim]);
            if carry {
                values[global] = Op::apply(
                    init[0],
                    Op::apply(block_prefixes[block - 1usize], local_inclusive[global]),
                );
            } else {
                values[global] = Op::apply(init[0], local_inclusive[global]);
            }
        } else {
            values[global] = local_inclusive[global];
        }
    }
}

macro_rules! define_tuple_by_key_scan_kernels {
    (
        $block_name:ident,
        $add_prefix_name:ident,
        $exclusive_name:ident,
        ( $( $ty:ident: $key:ident: $tail:ident ),+ )
    ) => {
        define_tuple_by_key_block_scan_kernels!(
            $block_name,
            $add_prefix_name,
            ($( $ty: $key: $tail ),+)
        );

        #[allow(dead_code)]
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $exclusive_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &[$ty], )+
            inclusive: &[T],
            init: &[T],
            output: &mut [T],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < output.len() {
                if global == 0usize
                    || !KeyEq::apply(
                        ($( $key[global - 1usize] ),+),
                        ($( $key[global] ),+),
                    )
                {
                    output[global] = init[0];
                } else {
                    output[global] = Op::apply(init[0], inclusive[global - 1usize]);
                }
            }
        }
    };
}

define_tuple_by_key_scan_kernels!(
    scan_tuple2_by_key_block_kernel,
    scan_tuple2_by_key_add_block_prefix_kernel,
    scan_tuple2_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a, B: key_b: block_tail_b)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple3_by_key_block_kernel,
    scan_tuple3_by_key_add_block_prefix_kernel,
    scan_tuple3_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a, B: key_b: block_tail_b, C: key_c: block_tail_c)
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_values_at_ends_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    inclusive: &[T],
    init: &[T],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < keys.len() {
        if global + 1usize == keys.len() || !KeyEq::apply(keys[global], keys[global + 1usize]) {
            values[global] = Op::apply(init[0], inclusive[global]);
        } else {
            values[global] = inclusive[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_apply_init_kernel<A: CubePrimitive, B: CubePrimitive, Op: BinaryOp<(A, B)>>(
    a: &[A],
    b: &[B],
    init_a: &[A],
    init_b: &[B],
    out_a: &mut [A],
    out_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < a.len() {
        let reduced = Op::apply((init_a[0], init_b[0]), (a[global], b[global]));
        out_a[global] = reduced.0;
        out_b[global] = reduced.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_apply_init_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: BinaryOp<(A, B, C)>,
>(
    a: &[A],
    b: &[B],
    c: &[C],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    out_a: &mut [A],
    out_b: &mut [B],
    out_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < a.len() {
        let reduced = Op::apply(
            (init_a[0], init_b[0], init_c[0]),
            (a[global], b[global], c[global]),
        );
        out_a[global] = reduced.0;
        out_b[global] = reduced.1;
        out_c[global] = reduced.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn adjacent_difference_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
    input: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < input.len() {
        if unit == 0 {
            output[unit] = input[unit];
        } else {
            output[unit] = Op::apply(input[unit], input[unit - 1usize]);
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_if_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    Op: UnaryOp<T, Output = T>,
    Pred: PredicateOp<S>,
>(
    input: &[T],
    stencil: &[S],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < output.len() && Pred::apply(stencil[unit]) {
        output[unit] = Op::apply(input[unit]);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_binary_if_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    Op: BinaryOp<T>,
    Pred: PredicateOp<S>,
>(
    lhs: &[T],
    rhs: &[T],
    stencil: &[S],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < output.len() && Pred::apply(stencil[unit]) {
        output[unit] = Op::apply(lhs[unit], rhs[unit]);
    }
}
