use crate::op::BinaryPredicateOp;
use cubecl::prelude::*;

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_path_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    output: &mut [T],
    lhs: &[T],
    rhs: &[T],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < output.len() {
        let low_init = RuntimeCell::<usize>::new(0usize);
        if out > rhs.len() {
            low_init.store(out - rhs.len());
        }

        let high_init = RuntimeCell::<usize>::new(out);
        if high_init.read() > lhs.len() {
            high_init.store(lhs.len());
        }

        let low = RuntimeCell::<usize>::new(low_init.read());
        let high = RuntimeCell::<usize>::new(high_init.read());
        while low.read() < high.read() {
            let mid = (low.read() + high.read()) / 2usize;
            let rhs_index = out - mid;
            if mid < lhs.len()
                && rhs_index > 0usize
                && !Less::apply(rhs[rhs_index - 1usize], lhs[mid])
            {
                low.store(mid + 1usize);
            } else {
                high.store(mid);
            }
        }

        let lhs_index = low.read();
        let rhs_index = out - lhs_index;
        if lhs_index < lhs.len() {
            if rhs_index >= rhs.len() {
                output[out] = lhs[lhs_index];
            } else if !Less::apply(rhs[rhs_index], lhs[lhs_index]) {
                output[out] = lhs[lhs_index];
            } else {
                output[out] = rhs[rhs_index];
            }
        } else {
            output[out] = rhs[rhs_index];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_by_key_control_path_kernel<K: CubePrimitive, Less: BinaryPredicateOp<K>>(
    lhs_keys: &[K],
    rhs_keys: &[K],
    out_keys: &mut [K],
    source_sides: &mut [u32],
    source_indices: &mut [u32],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < out_keys.len() {
        let low_init = RuntimeCell::<usize>::new(0usize);
        if out > rhs_keys.len() {
            low_init.store(out - rhs_keys.len());
        }

        let high_init = RuntimeCell::<usize>::new(out);
        if high_init.read() > lhs_keys.len() {
            high_init.store(lhs_keys.len());
        }

        let low = RuntimeCell::<usize>::new(low_init.read());
        let high = RuntimeCell::<usize>::new(high_init.read());
        while low.read() < high.read() {
            let mid = (low.read() + high.read()) / 2usize;
            let rhs_index = out - mid;
            if mid < lhs_keys.len()
                && rhs_index > 0usize
                && !Less::apply(rhs_keys[rhs_index - 1usize], lhs_keys[mid])
            {
                low.store(mid + 1usize);
            } else {
                high.store(mid);
            }
        }

        let lhs_index = low.read();
        let rhs_index = out - lhs_index;
        if lhs_index < lhs_keys.len() {
            if rhs_index >= rhs_keys.len() {
                out_keys[out] = lhs_keys[lhs_index];
                source_sides[out] = 0u32;
                source_indices[out] = lhs_index as u32;
            } else if !Less::apply(rhs_keys[rhs_index], lhs_keys[lhs_index]) {
                out_keys[out] = lhs_keys[lhs_index];
                source_sides[out] = 0u32;
                source_indices[out] = lhs_index as u32;
            } else {
                out_keys[out] = rhs_keys[rhs_index];
                source_sides[out] = 1u32;
                source_indices[out] = rhs_index as u32;
            }
        } else {
            out_keys[out] = rhs_keys[rhs_index];
            source_sides[out] = 1u32;
            source_indices[out] = rhs_index as u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_by_key_values_from_control_kernel<T: CubePrimitive>(
    lhs_values: &[T],
    rhs_values: &[T],
    source_sides: &[u32],
    source_indices: &[u32],
    out_values: &mut [T],
) {
    let out = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if out < out_values.len() {
        let index = source_indices[out] as usize;
        if source_sides[out] == 0u32 {
            out_values[out] = lhs_values[index];
        } else {
            out_values[out] = rhs_values[index];
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
pub(crate) fn merge_sort_tuple2_by_key_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    T: CubePrimitive,
    Less: BinaryPredicateOp<(A, B)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_values: &[T],
    width: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_values: &mut [T],
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
                output_values[out] = input_values[left_start + lhs_index];
            } else {
                output_a[out] = input_a[right_start + rhs_index];
                output_b[out] = input_b[right_start + rhs_index];
                output_values[out] = input_values[right_start + rhs_index];
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn merge_sort_tuple3_by_key_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    T: CubePrimitive,
    Less: BinaryPredicateOp<(A, B, C)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_c: &[C],
    input_values: &[T],
    width: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_values: &mut [T],
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
                output_values[out] = input_values[left_start + lhs_index];
            } else {
                output_a[out] = input_a[right_start + rhs_index];
                output_b[out] = input_b[right_start + rhs_index];
                output_c[out] = input_c[right_start + rhs_index];
                output_values[out] = input_values[right_start + rhs_index];
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
