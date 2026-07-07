use crate::{
    detail::op::kernel::{BinaryPredicateOp, PredicateOp},
    expr::DeviceGpuExpr,
};
use cubecl::prelude::*;

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn gather_if_flags_kernel<
    T: CubePrimitive,
    InputExpr: crate::expr::DeviceGpuExpr<T>,
    IndexExpr: crate::expr::DeviceGpuExpr<u32>,
>(
    input_slot0: &[T],
    input_slot1: &[T],
    input_slot2: &[T],
    input_slot3: &[T],
    input_slot_offsets: &[u32],
    index_slot0: &[u32],
    index_slot1: &[u32],
    index_slot2: &[u32],
    index_slot3: &[u32],
    index_slot_offsets: &[u32],
    flags: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let index = IndexExpr::eval(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot_offsets,
            global,
        ) as usize;
        output[global] = InputExpr::eval(
            input_slot0,
            input_slot1,
            input_slot2,
            input_slot3,
            input_slot_offsets,
            index,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn gather_if_flags_into_kernel<
    T: CubePrimitive,
    InputExpr: crate::expr::DeviceGpuExpr<T>,
    IndexExpr: crate::expr::DeviceGpuExpr<u32>,
>(
    input_slot0: &[T],
    input_slot1: &[T],
    input_slot2: &[T],
    input_slot3: &[T],
    input_slot_offsets: &[u32],
    index_slot0: &[u32],
    index_slot1: &[u32],
    index_slot2: &[u32],
    index_slot3: &[u32],
    index_slot_offsets: &[u32],
    flags: &[u32],
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let index = IndexExpr::eval(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot_offsets,
            global,
        ) as usize;
        output[output_offset[0] as usize + global] = InputExpr::eval(
            input_slot0,
            input_slot1,
            input_slot2,
            input_slot3,
            input_slot_offsets,
            index,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn gather_if_flags_index7_into_kernel<
    T: CubePrimitive,
    InputExpr: crate::expr::DeviceGpuExpr<T>,
    IndexExpr: crate::expr::LogicalDeviceExpr7<u32, I0, I1, I2, I3, I4, I5, I6>,
    I0: CubePrimitive,
    I1: CubePrimitive,
    I2: CubePrimitive,
    I3: CubePrimitive,
    I4: CubePrimitive,
    I5: CubePrimitive,
    I6: CubePrimitive,
>(
    input_slot0: &[T],
    input_slot1: &[T],
    input_slot2: &[T],
    input_slot3: &[T],
    input_slot_offsets: &[u32],
    index_slot0: &[I0],
    index_slot1: &[I1],
    index_slot2: &[I2],
    index_slot3: &[I3],
    index_slot4: &[I4],
    index_slot5: &[I5],
    index_slot6: &[I6],
    index_slot_offsets: &[u32],
    flags: &[u32],
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let index = IndexExpr::eval7(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot4,
            index_slot5,
            index_slot6,
            index_slot_offsets,
            global,
        ) as usize;
        output[output_offset[0] as usize + global] = InputExpr::eval(
            input_slot0,
            input_slot1,
            input_slot2,
            input_slot3,
            input_slot_offsets,
            index,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_flags_into_kernel<
    T: CubePrimitive,
    InputExpr: crate::expr::DeviceGpuExpr<T>,
>(
    input_slot0: &[T],
    input_slot1: &[T],
    input_slot2: &[T],
    input_slot3: &[T],
    input_slot_offsets: &[u32],
    flags: &[u32],
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        output[output_offset[0] as usize + global] = InputExpr::eval(
            input_slot0,
            input_slot1,
            input_slot2,
            input_slot3,
            input_slot_offsets,
            global,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scatter_if_flags_kernel<
    T: CubePrimitive,
    ValueExpr: crate::expr::DeviceGpuExpr<T>,
    IndexExpr: crate::expr::DeviceGpuExpr<u32>,
>(
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_slot_offsets: &[u32],
    index_slot0: &[u32],
    index_slot1: &[u32],
    index_slot2: &[u32],
    index_slot3: &[u32],
    index_slot_offsets: &[u32],
    flags: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let index = IndexExpr::eval(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot_offsets,
            global,
        ) as usize;
        output[index] = ValueExpr::eval(
            value_slot0,
            value_slot1,
            value_slot2,
            value_slot3,
            value_slot_offsets,
            global,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scatter_if_flags_into_kernel<
    T: CubePrimitive,
    ValueExpr: crate::expr::DeviceGpuExpr<T>,
    IndexExpr: crate::expr::DeviceGpuExpr<u32>,
>(
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_slot_offsets: &[u32],
    index_slot0: &[u32],
    index_slot1: &[u32],
    index_slot2: &[u32],
    index_slot3: &[u32],
    index_slot_offsets: &[u32],
    flags: &[u32],
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let index = IndexExpr::eval(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot_offsets,
            global,
        ) as usize;
        output[output_offset[0] as usize + index] = ValueExpr::eval(
            value_slot0,
            value_slot1,
            value_slot2,
            value_slot3,
            value_slot_offsets,
            global,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scatter_if_flags_index7_into_kernel<
    T: CubePrimitive,
    ValueExpr: crate::expr::DeviceGpuExpr<T>,
    IndexExpr: crate::expr::LogicalDeviceExpr7<u32, I0, I1, I2, I3, I4, I5, I6>,
    I0: CubePrimitive,
    I1: CubePrimitive,
    I2: CubePrimitive,
    I3: CubePrimitive,
    I4: CubePrimitive,
    I5: CubePrimitive,
    I6: CubePrimitive,
>(
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_slot_offsets: &[u32],
    index_slot0: &[I0],
    index_slot1: &[I1],
    index_slot2: &[I2],
    index_slot3: &[I3],
    index_slot4: &[I4],
    index_slot5: &[I5],
    index_slot6: &[I6],
    index_slot_offsets: &[u32],
    flags: &[u32],
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let index = IndexExpr::eval7(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot4,
            index_slot5,
            index_slot6,
            index_slot_offsets,
            global,
        ) as usize;
        output[output_offset[0] as usize + index] = ValueExpr::eval(
            value_slot0,
            value_slot1,
            value_slot2,
            value_slot3,
            value_slot_offsets,
            global,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn first_flag_partials_kernel(flags: &[u32], logical_len: &[u32], partials: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let sentinel = logical_len[0];
    let mut candidates = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = cube_dim * partial_count;
    let best = RuntimeCell::<u32>::new(sentinel);

    while i.read() < (logical_len[0] as usize) {
        if flags[i.read()] != 0u32 && (i.read() as u32) < best.read() {
            best.store(i.read() as u32);
        }
        i.store(i.read() + step);
    }

    candidates[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() {
            let rhs = candidates[unit + stride.read()];
            if rhs != sentinel && (candidates[unit] == sentinel || rhs < candidates[unit]) {
                candidates[unit] = rhs;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize {
        partials[CUBE_POS as usize] = candidates[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn first_unset_flag_partials_kernel(
    flags: &[u32],
    logical_len: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let sentinel = logical_len[0];
    let mut candidates = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = cube_dim * partial_count;
    let best = RuntimeCell::<u32>::new(sentinel);

    while i.read() < (logical_len[0] as usize) {
        if flags[i.read()] == 0u32 && (i.read() as u32) < best.read() {
            best.store(i.read() as u32);
        }
        i.store(i.read() + step);
    }

    candidates[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() {
            let rhs = candidates[unit + stride.read()];
            if rhs != sentinel && (candidates[unit] == sentinel || rhs < candidates[unit]) {
                candidates[unit] = rhs;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize {
        partials[CUBE_POS as usize] = candidates[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn last_flag_partials_kernel(flags: &[u32], logical_len: &[u32], partials: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let sentinel = logical_len[0];
    let mut candidates = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = cube_dim * partial_count;
    let best = RuntimeCell::<u32>::new(sentinel);

    while i.read() < (logical_len[0] as usize) {
        if flags[i.read()] != 0u32 {
            best.store(i.read() as u32);
        }
        i.store(i.read() + step);
    }

    candidates[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() {
            let rhs = candidates[unit + stride.read()];
            if rhs != sentinel && (candidates[unit] == sentinel || candidates[unit] < rhs) {
                candidates[unit] = rhs;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize {
        partials[CUBE_POS as usize] = candidates[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn first_index_partials_kernel(
    candidates_in: &[u32],
    candidate_len: &[u32],
    sentinel: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let mut candidates = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = cube_dim * partial_count;
    let best = RuntimeCell::<u32>::new(sentinel[0]);

    while i.read() < (candidate_len[0] as usize) {
        let value = candidates_in[i.read()];
        if value != sentinel[0] && (best.read() == sentinel[0] || value < best.read()) {
            best.store(value);
        }
        i.store(i.read() + step);
    }

    candidates[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() {
            let rhs = candidates[unit + stride.read()];
            if rhs != sentinel[0] && (candidates[unit] == sentinel[0] || rhs < candidates[unit]) {
                candidates[unit] = rhs;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize {
        partials[CUBE_POS as usize] = candidates[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn last_index_partials_kernel(
    candidates_in: &[u32],
    candidate_len: &[u32],
    sentinel: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let mut candidates = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = cube_dim * partial_count;
    let best = RuntimeCell::<u32>::new(sentinel[0]);

    while i.read() < (candidate_len[0] as usize) {
        let value = candidates_in[i.read()];
        if value != sentinel[0] && (best.read() == sentinel[0] || best.read() < value) {
            best.store(value);
        }
        i.store(i.read() + step);
    }

    candidates[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() {
            let rhs = candidates[unit + stride.read()];
            if rhs != sentinel[0] && (candidates[unit] == sentinel[0] || candidates[unit] < rhs) {
                candidates[unit] = rhs;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize {
        partials[CUBE_POS as usize] = candidates[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn invert_flags_kernel(flags: &[u32], output: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if flags[global] == 0u32 {
            output[global] = 1u32;
        } else {
            output[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn mismatch_flags_kernel<T: CubePrimitive, Eq: BinaryPredicateOp<T>>(
    left: &[T],
    right: &[T],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if Eq::apply(left[global], right[global]) {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn mismatch_device_expr_flags_kernel<
    T: CubePrimitive,
    LeftExpr: DeviceGpuExpr<T>,
    RightExpr: DeviceGpuExpr<T>,
    Eq: BinaryPredicateOp<T>,
>(
    left_slot0: &[T],
    left_slot1: &[T],
    left_slot2: &[T],
    left_slot3: &[T],
    left_slot_offsets: &[u32],
    right_slot0: &[T],
    right_slot1: &[T],
    right_slot2: &[T],
    right_slot3: &[T],
    right_slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let left = LeftExpr::eval(
            left_slot0,
            left_slot1,
            left_slot2,
            left_slot3,
            left_slot_offsets,
            global,
        );
        let right = RightExpr::eval(
            right_slot0,
            right_slot1,
            right_slot2,
            right_slot3,
            right_slot_offsets,
            global,
        );
        if Eq::apply(left, right) {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn adjacent_find_device_expr_flags_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Pred: BinaryPredicateOp<T>,
>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if global + 1usize < flags.len()
            && Pred::apply(
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global),
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global + 1usize),
            )
        {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn sorted_break_device_expr_flags_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if global > 0usize
            && Less::apply(
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global),
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global - 1usize),
            )
        {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn find_first_of_device_expr_flags_kernel<
    T: CubePrimitive,
    InputExpr: DeviceGpuExpr<T>,
    NeedleExpr: DeviceGpuExpr<T>,
    Eq: BinaryPredicateOp<T>,
>(
    input_slot0: &[T],
    input_slot1: &[T],
    input_slot2: &[T],
    input_slot3: &[T],
    input_slot_offsets: &[u32],
    needle_slot0: &[T],
    needle_slot1: &[T],
    needle_slot2: &[T],
    needle_slot3: &[T],
    needle_slot_offsets: &[u32],
    needle_len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let value = InputExpr::eval(
            input_slot0,
            input_slot1,
            input_slot2,
            input_slot3,
            input_slot_offsets,
            global,
        );
        let needle = RuntimeCell::<usize>::new(0usize);
        let found = RuntimeCell::<u32>::new(0u32);
        while needle.read() < needle_len[0] as usize {
            if Eq::apply(
                value,
                NeedleExpr::eval(
                    needle_slot0,
                    needle_slot1,
                    needle_slot2,
                    needle_slot3,
                    needle_slot_offsets,
                    needle.read(),
                ),
            ) {
                found.store(1u32);
                needle.store(needle_len[0] as usize);
            } else {
                needle.store(needle.read() + 1usize);
            }
        }
        flags[global] = found.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn subrange_match_flags_kernel<T: CubePrimitive, Eq: BinaryPredicateOp<T>>(
    input: &[T],
    pattern: &[T],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let start = (CUBE_POS as usize) * cube_dim + unit;
    if start < flags.len() {
        let offset = RuntimeCell::<usize>::new(0usize);
        let matched = RuntimeCell::<u32>::new(1u32);
        while offset.read() < pattern.len() {
            if !Eq::apply(input[start + offset.read()], pattern[offset.read()]) {
                matched.store(0u32);
                offset.store(pattern.len());
            } else {
                offset.store(offset.read() + 1usize);
            }
        }
        flags[start] = matched.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn lexicographical_diff_device_expr_flags_kernel<
    T: CubePrimitive,
    LeftExpr: DeviceGpuExpr<T>,
    RightExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    left_slot0: &[T],
    left_slot1: &[T],
    left_slot2: &[T],
    left_slot3: &[T],
    left_slot_offsets: &[u32],
    right_slot0: &[T],
    right_slot1: &[T],
    right_slot2: &[T],
    right_slot3: &[T],
    right_slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let left = LeftExpr::eval(
            left_slot0,
            left_slot1,
            left_slot2,
            left_slot3,
            left_slot_offsets,
            global,
        );
        let right = RightExpr::eval(
            right_slot0,
            right_slot1,
            right_slot2,
            right_slot3,
            right_slot_offsets,
            global,
        );
        if Less::apply(left, right) || Less::apply(right, left) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn lexicographical_compare_at_device_expr_kernel<
    T: CubePrimitive,
    LeftExpr: DeviceGpuExpr<T>,
    RightExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    left_slot0: &[T],
    left_slot1: &[T],
    left_slot2: &[T],
    left_slot3: &[T],
    left_slot_offsets: &[u32],
    right_slot0: &[T],
    right_slot1: &[T],
    right_slot2: &[T],
    right_slot3: &[T],
    right_slot_offsets: &[u32],
    index: &[u32],
    output: &mut [u32],
) {
    if UNIT_POS == 0 {
        let i = index[0] as usize;
        let left = LeftExpr::eval(
            left_slot0,
            left_slot1,
            left_slot2,
            left_slot3,
            left_slot_offsets,
            i,
        );
        let right = RightExpr::eval(
            right_slot0,
            right_slot1,
            right_slot2,
            right_slot3,
            right_slot_offsets,
            i,
        );
        if Less::apply(left, right) {
            output[0] = 1u32;
        } else {
            output[0] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn partition_tail_selected_flags_kernel(
    input_flags: &[u32],
    point: &[u32],
    output_flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input_flags.len() {
        if global > (point[0] as usize) && input_flags[global] != 0u32 {
            output_flags[global] = 1u32;
        } else {
            output_flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn lower_bound_device_expr_flags_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    value: &[T],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if Less::apply(
            Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global),
            value[0],
        ) {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn upper_bound_device_expr_flags_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    value: &[T],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if Less::apply(
            value[0],
            Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global),
        ) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn lower_bound_device_expr_many_kernel<
    T: CubePrimitive,
    SourceExpr: DeviceGpuExpr<T>,
    ValueExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    source_slot0: &[T],
    source_slot1: &[T],
    source_slot2: &[T],
    source_slot3: &[T],
    source_offsets: &[u32],
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let value = ValueExpr::eval(
            value_slot0,
            value_slot1,
            value_slot2,
            value_slot3,
            value_offsets,
            global,
        );
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            if Less::apply(
                SourceExpr::eval(
                    source_slot0,
                    source_slot1,
                    source_slot2,
                    source_slot3,
                    source_offsets,
                    mid,
                ),
                value,
            ) {
                first = mid + 1usize;
                count = count - step - 1usize;
            } else {
                count = step;
            }
        }
        output[global] = first as u32;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn upper_bound_device_expr_many_kernel<
    T: CubePrimitive,
    SourceExpr: DeviceGpuExpr<T>,
    ValueExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    source_slot0: &[T],
    source_slot1: &[T],
    source_slot2: &[T],
    source_slot3: &[T],
    source_offsets: &[u32],
    value_slot0: &[T],
    value_slot1: &[T],
    value_slot2: &[T],
    value_slot3: &[T],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let value = ValueExpr::eval(
            value_slot0,
            value_slot1,
            value_slot2,
            value_slot3,
            value_offsets,
            global,
        );
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            if !Less::apply(
                value,
                SourceExpr::eval(
                    source_slot0,
                    source_slot1,
                    source_slot2,
                    source_slot3,
                    source_offsets,
                    mid,
                ),
            ) {
                first = mid + 1usize;
                count = count - step - 1usize;
            } else {
                count = step;
            }
        }
        output[global] = first as u32;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn binary_search_at_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &[T],
    value: &[T],
    index: &[u32],
    output: &mut [u32],
) {
    if UNIT_POS == 0 {
        let i = index[0] as usize;
        if i < input.len() && !Less::apply(input[i], value[0]) && !Less::apply(value[0], input[i]) {
            output[0] = 1u32;
        } else {
            output[0] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn remove_value_flags_kernel<T: CubePrimitive, Pred: BinaryPredicateOp<T>>(
    input: &[T],
    value: &[T],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    if unit < input.len() {
        if Pred::apply(input[unit], value[0]) {
            flags[unit] = 0u32;
        } else {
            flags[unit] = 1u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn replace_if_value_kernel<T: CubePrimitive, Pred: PredicateOp<T>>(
    input: &[T],
    replacement: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if Pred::apply(input[global]) {
            output[global] = replacement[0];
        } else {
            output[global] = input[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn replace_device_expr_with_flags_kernel<T: CubePrimitive, Expr: DeviceGpuExpr<T>>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    replacement: &[T],
    flags: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if flags[global] != 0u32 {
            output[global] = replacement[0];
        } else {
            output[global] = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn replace_with_flags_into_kernel<T: CubePrimitive>(
    replacement: &[T],
    flags: &[u32],
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        output[output_offset[0] as usize + global] = replacement[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn sorted_membership_device_expr_flags_kernel<
    T: CubePrimitive,
    CandidateExpr: DeviceGpuExpr<T>,
    SortedExpr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    candidate_slot0: &[T],
    candidate_slot1: &[T],
    candidate_slot2: &[T],
    candidate_slot3: &[T],
    candidate_slot_offsets: &[u32],
    candidate_len: &[u32],
    sorted_slot0: &[T],
    sorted_slot1: &[T],
    sorted_slot2: &[T],
    sorted_slot3: &[T],
    sorted_slot_offsets: &[u32],
    sorted_len: &[u32],
    keep_present: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let candidate_logical_len = candidate_len[0] as usize;
    let sorted_logical_len = sorted_len[0] as usize;
    if global < candidate_logical_len {
        let value = RuntimeCell::<T>::new(CandidateExpr::eval(
            candidate_slot0,
            candidate_slot1,
            candidate_slot2,
            candidate_slot3,
            candidate_slot_offsets,
            global,
        ));
        let candidate_first = RuntimeCell::<usize>::new(0usize);
        let candidate_count = RuntimeCell::<usize>::new(candidate_logical_len);

        while candidate_count.read() > 0usize {
            let step = candidate_count.read() / 2usize;
            let mid = candidate_first.read() + step;
            if Less::apply(
                CandidateExpr::eval(
                    candidate_slot0,
                    candidate_slot1,
                    candidate_slot2,
                    candidate_slot3,
                    candidate_slot_offsets,
                    mid,
                ),
                value.read(),
            ) {
                candidate_first.store(mid + 1usize);
                candidate_count.store(candidate_count.read() - step - 1usize);
            } else {
                candidate_count.store(step);
            }
        }

        let sorted_first = RuntimeCell::<usize>::new(0usize);
        let sorted_count = RuntimeCell::<usize>::new(sorted_logical_len);

        while sorted_count.read() > 0usize {
            let step = sorted_count.read() / 2usize;
            let mid = sorted_first.read() + step;
            if Less::apply(
                SortedExpr::eval(
                    sorted_slot0,
                    sorted_slot1,
                    sorted_slot2,
                    sorted_slot3,
                    sorted_slot_offsets,
                    mid,
                ),
                value.read(),
            ) {
                sorted_first.store(mid + 1usize);
                sorted_count.store(sorted_count.read() - step - 1usize);
            } else {
                sorted_count.store(step);
            }
        }

        let sorted_after = RuntimeCell::<usize>::new(0usize);
        let sorted_after_count = RuntimeCell::<usize>::new(sorted_logical_len);

        while sorted_after_count.read() > 0usize {
            let step = sorted_after_count.read() / 2usize;
            let mid = sorted_after.read() + step;
            if !Less::apply(
                value.read(),
                SortedExpr::eval(
                    sorted_slot0,
                    sorted_slot1,
                    sorted_slot2,
                    sorted_slot3,
                    sorted_slot_offsets,
                    mid,
                ),
            ) {
                sorted_after.store(mid + 1usize);
                sorted_after_count.store(sorted_after_count.read() - step - 1usize);
            } else {
                sorted_after_count.store(step);
            }
        }

        let rank = global - candidate_first.read();
        let other_count = sorted_after.read() - sorted_first.read();
        if (keep_present[0] != 0u32 && rank < other_count)
            || (keep_present[0] == 0u32 && rank >= other_count)
        {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn gather_if_kernel<T: CubePrimitive, Pred: PredicateOp<T>>(
    input: &[T],
    indices: &[u32],
    output: &mut [T],
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < indices.len() {
        let value = input[indices[unit] as usize];
        if Pred::apply(value) {
            output[unit] = value;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn minmax_element_partials_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &[T],
    len: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let partial_count = partials.len() / 2usize;
    let mut min_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut max_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_count;
    let has_value = RuntimeCell::<u32>::new(0u32);
    let min_index = RuntimeCell::<usize>::new(0usize);
    let max_index = RuntimeCell::<usize>::new(0usize);

    while i.read() < logical_len {
        if has_value.read() == 0u32 {
            min_index.store(i.read());
            max_index.store(i.read());
            has_value.store(1u32);
        } else {
            if Less::apply(input[i.read()], input[min_index.read()]) {
                min_index.store(i.read());
            }
            if Less::apply(input[max_index.read()], input[i.read()]) {
                max_index.store(i.read());
            }
        }
        i.store(i.read() + step);
    }

    min_indices[unit] = min_index.read() as u32;
    max_indices[unit] = max_index.read() as u32;
    valid[unit] = has_value.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() && valid[unit + stride.read()] != 0u32 {
            if valid[unit] == 0u32 {
                min_indices[unit] = min_indices[unit + stride.read()];
                max_indices[unit] = max_indices[unit + stride.read()];
                valid[unit] = 1u32;
            } else {
                let other_min = min_indices[unit + stride.read()] as usize;
                let current_min = min_indices[unit] as usize;
                if Less::apply(input[other_min], input[current_min]) {
                    min_indices[unit] = other_min as u32;
                }

                let other_max = max_indices[unit + stride.read()] as usize;
                let current_max = max_indices[unit] as usize;
                if Less::apply(input[current_max], input[other_max]) {
                    max_indices[unit] = other_max as u32;
                }
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize && valid[0] != 0u32 {
        let out = (CUBE_POS as usize) * 2usize;
        partials[out] = min_indices[0];
        partials[out + 1usize] = max_indices[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn minmax_index_partials_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &[T],
    candidates: &[u32],
    candidate_len: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = candidate_len[0] as usize;
    let partial_count = partials.len() / 2usize;
    let mut min_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut max_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_count;
    let has_value = RuntimeCell::<u32>::new(0u32);
    let min_index = RuntimeCell::<usize>::new(0usize);
    let max_index = RuntimeCell::<usize>::new(0usize);

    while i.read() < logical_len {
        let candidate_min = candidates[i.read() * 2usize] as usize;
        let candidate_max = candidates[i.read() * 2usize + 1usize] as usize;
        if has_value.read() == 0u32 {
            min_index.store(candidate_min);
            max_index.store(candidate_max);
            has_value.store(1u32);
        } else {
            if Less::apply(input[candidate_min], input[min_index.read()]) {
                min_index.store(candidate_min);
            }
            if Less::apply(input[max_index.read()], input[candidate_max]) {
                max_index.store(candidate_max);
            }
        }
        i.store(i.read() + step);
    }

    min_indices[unit] = min_index.read() as u32;
    max_indices[unit] = max_index.read() as u32;
    valid[unit] = has_value.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() && valid[unit + stride.read()] != 0u32 {
            if valid[unit] == 0u32 {
                min_indices[unit] = min_indices[unit + stride.read()];
                max_indices[unit] = max_indices[unit + stride.read()];
                valid[unit] = 1u32;
            } else {
                let other_min = min_indices[unit + stride.read()] as usize;
                let current_min = min_indices[unit] as usize;
                if Less::apply(input[other_min], input[current_min]) {
                    min_indices[unit] = other_min as u32;
                }

                let other_max = max_indices[unit + stride.read()] as usize;
                let current_max = max_indices[unit] as usize;
                if Less::apply(input[current_max], input[other_max]) {
                    max_indices[unit] = other_max as u32;
                }
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize && valid[0] != 0u32 {
        let out = (CUBE_POS as usize) * 2usize;
        partials[out] = min_indices[0];
        partials[out + 1usize] = max_indices[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn minmax_element_device_expr_partials_kernel<
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
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let partial_count = partials.len() / 2usize;
    let mut min_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut max_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_count;
    let has_value = RuntimeCell::<u32>::new(0u32);
    let min_index = RuntimeCell::<usize>::new(0usize);
    let max_index = RuntimeCell::<usize>::new(0usize);

    while i.read() < logical_len {
        if has_value.read() == 0u32 {
            min_index.store(i.read());
            max_index.store(i.read());
            has_value.store(1u32);
        } else {
            if Less::apply(
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, i.read()),
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, min_index.read()),
            ) {
                min_index.store(i.read());
            }
            if Less::apply(
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, max_index.read()),
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, i.read()),
            ) {
                max_index.store(i.read());
            }
        }
        i.store(i.read() + step);
    }

    min_indices[unit] = min_index.read() as u32;
    max_indices[unit] = max_index.read() as u32;
    valid[unit] = has_value.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() && valid[unit + stride.read()] != 0u32 {
            if valid[unit] == 0u32 {
                min_indices[unit] = min_indices[unit + stride.read()];
                max_indices[unit] = max_indices[unit + stride.read()];
                valid[unit] = 1u32;
            } else {
                let other_min = min_indices[unit + stride.read()] as usize;
                let current_min = min_indices[unit] as usize;
                if Less::apply(
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, other_min),
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, current_min),
                ) {
                    min_indices[unit] = other_min as u32;
                }

                let other_max = max_indices[unit + stride.read()] as usize;
                let current_max = max_indices[unit] as usize;
                if Less::apply(
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, current_max),
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, other_max),
                ) {
                    max_indices[unit] = other_max as u32;
                }
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize && valid[0] != 0u32 {
        let out = (CUBE_POS as usize) * 2usize;
        partials[out] = min_indices[0];
        partials[out + 1usize] = max_indices[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn minmax_index_device_expr_partials_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    candidates: &[u32],
    candidate_len: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = candidate_len[0] as usize;
    let partial_count = partials.len() / 2usize;
    let mut min_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut max_indices = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_count;
    let has_value = RuntimeCell::<u32>::new(0u32);
    let min_index = RuntimeCell::<usize>::new(0usize);
    let max_index = RuntimeCell::<usize>::new(0usize);

    while i.read() < logical_len {
        let candidate_min = candidates[i.read() * 2usize] as usize;
        let candidate_max = candidates[i.read() * 2usize + 1usize] as usize;
        if has_value.read() == 0u32 {
            min_index.store(candidate_min);
            max_index.store(candidate_max);
            has_value.store(1u32);
        } else {
            if Less::apply(
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, candidate_min),
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, min_index.read()),
            ) {
                min_index.store(candidate_min);
            }
            if Less::apply(
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, max_index.read()),
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, candidate_max),
            ) {
                max_index.store(candidate_max);
            }
        }
        i.store(i.read() + step);
    }

    min_indices[unit] = min_index.read() as u32;
    max_indices[unit] = max_index.read() as u32;
    valid[unit] = has_value.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2usize);
    while stride.read() > 0usize {
        if unit < stride.read() && valid[unit + stride.read()] != 0u32 {
            if valid[unit] == 0u32 {
                min_indices[unit] = min_indices[unit + stride.read()];
                max_indices[unit] = max_indices[unit + stride.read()];
                valid[unit] = 1u32;
            } else {
                let other_min = min_indices[unit + stride.read()] as usize;
                let current_min = min_indices[unit] as usize;
                if Less::apply(
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, other_min),
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, current_min),
                ) {
                    min_indices[unit] = other_min as u32;
                }

                let other_max = max_indices[unit + stride.read()] as usize;
                let current_max = max_indices[unit] as usize;
                if Less::apply(
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, current_max),
                    Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, other_max),
                ) {
                    max_indices[unit] = other_max as u32;
                }
            }
        }
        sync_cube();
        stride.store(stride.read() / 2usize);
    }

    if unit == 0usize && valid[0] != 0u32 {
        let out = (CUBE_POS as usize) * 2usize;
        partials[out] = min_indices[0];
        partials[out + 1usize] = max_indices[0];
    }
}
