use crate::op::{BinaryPredicateOp, PredicateOp};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn copy_if_flags_kernel<T: CubePrimitive, S: CubePrimitive, Pred: PredicateOp<S>>(
    input: &Array<T>,
    stencil: &Array<S>,
    invert: &Array<u32>,
    flags: &mut Array<u32>,
    values: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        values[global] = input[global];
        let matched = Pred::apply(stencil[global]);
        if (matched && invert[0] == 0u32) || (!matched && invert[0] != 0u32) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn copy_if_flag_only_kernel<S: CubePrimitive, Pred: PredicateOp<S>>(
    stencil: &Array<S>,
    invert: &Array<u32>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < stencil.len() {
        let matched = Pred::apply(stencil[global]);
        if (matched && invert[0] == 0u32) || (!matched && invert[0] != 0u32) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn gather_if_flags_kernel<T: CubePrimitive>(
    input: &Array<T>,
    indices: &Array<u32>,
    flags: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < indices.len() && flags[global] != 0u32 {
        let index = indices[global] as usize;
        output[global] = input[index];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scatter_if_flags_kernel<T: CubePrimitive>(
    input: &Array<T>,
    indices: &Array<u32>,
    flags: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() && flags[global] != 0u32 {
        let index = indices[global] as usize;
        output[index] = input[global];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn adjacent_find_flags_kernel<T: CubePrimitive, Pred: BinaryPredicateOp<T>>(
    input: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global + 1usize < input.len() {
        if Pred::apply(input[global], input[global + 1usize]) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn first_flag_partials_kernel(
    flags: &Array<u32>,
    logical_len: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let sentinel = logical_len[0];
    let mut candidates = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn first_unset_flag_partials_kernel(
    flags: &Array<u32>,
    logical_len: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let sentinel = logical_len[0];
    let mut candidates = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn last_flag_partials_kernel(
    flags: &Array<u32>,
    logical_len: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let sentinel = logical_len[0];
    let mut candidates = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn first_index_partials_kernel(
    candidates_in: &Array<u32>,
    candidate_len: &Array<u32>,
    sentinel: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let mut candidates = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn last_index_partials_kernel(
    candidates_in: &Array<u32>,
    candidate_len: &Array<u32>,
    sentinel: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let partial_count = partials.len();
    let mut candidates = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn invert_flags_kernel(flags: &Array<u32>, output: &mut Array<u32>) {
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

#[cube(launch_unchecked)]
pub(crate) fn mismatch_flags_kernel<T: CubePrimitive, Eq: BinaryPredicateOp<T>>(
    left: &Array<T>,
    right: &Array<T>,
    flags: &mut Array<u32>,
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

#[cube(launch_unchecked)]
pub(crate) fn sorted_break_flags_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if global > 0usize && Less::apply(input[global], input[global - 1usize]) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn find_first_of_flags_kernel<T: CubePrimitive, Eq: BinaryPredicateOp<T>>(
    input: &Array<T>,
    needles: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        let needle = RuntimeCell::<usize>::new(0usize);
        let found = RuntimeCell::<u32>::new(0u32);
        while needle.read() < needles.len() {
            if Eq::apply(input[global], needles[needle.read()]) {
                found.store(1u32);
                needle.store(needles.len());
            } else {
                needle.store(needle.read() + 1usize);
            }
        }
        flags[global] = found.read();
    }
}

#[cube(launch_unchecked)]
pub(crate) fn subrange_match_flags_kernel<T: CubePrimitive, Eq: BinaryPredicateOp<T>>(
    input: &Array<T>,
    pattern: &Array<T>,
    flags: &mut Array<u32>,
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

#[cube(launch_unchecked)]
pub(crate) fn lexicographical_diff_flags_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    left: &Array<T>,
    right: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if Less::apply(left[global], right[global]) || Less::apply(right[global], left[global]) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn lexicographical_compare_at_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    left: &Array<T>,
    right: &Array<T>,
    index: &Array<u32>,
    output: &mut Array<u32>,
) {
    if UNIT_POS == 0 {
        let i = index[0] as usize;
        if Less::apply(left[i], right[i]) {
            output[0] = 1u32;
        } else {
            output[0] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn partition_point_flags_kernel<T: CubePrimitive, Pred: PredicateOp<T>>(
    input: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if Pred::apply(input[global]) {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn partition_tail_true_flags_kernel<T: CubePrimitive, Pred: PredicateOp<T>>(
    input: &Array<T>,
    point: &Array<u32>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if global > (point[0] as usize) && Pred::apply(input[global]) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn lower_bound_flags_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &Array<T>,
    value: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if Less::apply(input[global], value[0]) {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn upper_bound_flags_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &Array<T>,
    value: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if Less::apply(value[0], input[global]) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn binary_search_at_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &Array<T>,
    value: &Array<T>,
    index: &Array<u32>,
    output: &mut Array<u32>,
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

#[cube(launch_unchecked)]
pub(crate) fn sorted_membership_flags_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    candidates: &Array<T>,
    sorted: &Array<T>,
    keep_present: &Array<u32>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < candidates.len() {
        let value = RuntimeCell::<T>::new(candidates[global]);
        let candidate_first = RuntimeCell::<usize>::new(0usize);
        let candidate_count = RuntimeCell::<usize>::new(candidates.len());

        while candidate_count.read() > 0usize {
            let step = candidate_count.read() / 2usize;
            let mid = candidate_first.read() + step;
            if Less::apply(candidates[mid], value.read()) {
                candidate_first.store(mid + 1usize);
                candidate_count.store(candidate_count.read() - step - 1usize);
            } else {
                candidate_count.store(step);
            }
        }

        let sorted_first = RuntimeCell::<usize>::new(0usize);
        let sorted_count = RuntimeCell::<usize>::new(sorted.len());

        while sorted_count.read() > 0usize {
            let step = sorted_count.read() / 2usize;
            let mid = sorted_first.read() + step;
            if Less::apply(sorted[mid], value.read()) {
                sorted_first.store(mid + 1usize);
                sorted_count.store(sorted_count.read() - step - 1usize);
            } else {
                sorted_count.store(step);
            }
        }

        let sorted_after = RuntimeCell::<usize>::new(0usize);
        let sorted_after_count = RuntimeCell::<usize>::new(sorted.len());

        while sorted_after_count.read() > 0usize {
            let step = sorted_after_count.read() / 2usize;
            let mid = sorted_after.read() + step;
            if !Less::apply(value.read(), sorted[mid]) {
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

#[cube(launch_unchecked)]
pub(crate) fn remove_value_flags_kernel<T: CubePrimitive, Pred: BinaryPredicateOp<T>>(
    input: &Array<T>,
    value: &Array<T>,
    flags: &mut Array<u32>,
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

#[cube(launch_unchecked)]
pub(crate) fn replace_if_value_kernel<T: CubePrimitive, Pred: PredicateOp<T>>(
    input: &Array<T>,
    replacement: &Array<T>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn replace_with_flags_kernel<T: CubePrimitive>(
    input: &Array<T>,
    replacement: &Array<T>,
    flags: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if flags[global] != 0u32 {
            output[global] = replacement[0];
        } else {
            output[global] = input[global];
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn unique_flags_kernel<T: CubePrimitive, Pred: BinaryPredicateOp<T>>(
    input: &Array<T>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        if global == 0 {
            flags[global] = 1u32;
        } else if Pred::apply(input[global - 1usize], input[global]) {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn unique_by_key_flags_kernel<K: CubePrimitive, Pred: BinaryPredicateOp<K>>(
    keys: &Array<K>,
    flags: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < keys.len() {
        if global == 0 {
            flags[global] = 1u32;
        } else if Pred::apply(keys[global - 1usize], keys[global]) {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn gather_if_kernel<T: CubePrimitive, Pred: PredicateOp<T>>(
    input: &Array<T>,
    indices: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < indices.len() {
        let value = input[indices[unit] as usize];
        if Pred::apply(value) {
            output[unit] = value;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn minmax_element_partials_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &Array<T>,
    len: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let partial_count = partials.len() / 2usize;
    let mut min_indices = SharedMemory::<u32>::new(cube_dim);
    let mut max_indices = SharedMemory::<u32>::new(cube_dim);
    let mut valid = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn minmax_index_partials_kernel<T: CubePrimitive, Less: BinaryPredicateOp<T>>(
    input: &Array<T>,
    candidates: &Array<u32>,
    candidate_len: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = candidate_len[0] as usize;
    let partial_count = partials.len() / 2usize;
    let mut min_indices = SharedMemory::<u32>::new(cube_dim);
    let mut max_indices = SharedMemory::<u32>::new(cube_dim);
    let mut valid = SharedMemory::<u32>::new(cube_dim);

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
