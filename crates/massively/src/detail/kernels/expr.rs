use crate::{
    detail::op::kernel::{BinaryOp2, PredicateOp1, PredicateOp2, UnaryOp},
    expr::{DeviceGpuExpr, GpuExpr},
};
use cubecl::prelude::*;

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn collect_expr_block_kernel<T: CubePrimitive, Expr: GpuExpr<T>>(
    output: &mut [T],
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        output[global] = Expr::eval(input, indices, rhs, rhs_indices, global);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn device_collect_expr_block_kernel<T: CubePrimitive, Expr: DeviceGpuExpr<T>>(
    output: &mut [T],
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    len: &[u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        output[global] = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn device_collect_expr_reverse_block_kernel<T: CubePrimitive, Expr: DeviceGpuExpr<T>>(
    output: &mut [T],
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    len: &[u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let source_index = (len[0] as usize) - 1usize - global;
        output[global] = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, source_index);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_unary_tuple1_kernel<
    T: CubePrimitive,
    A: CubePrimitive,
    Op: UnaryOp<(T,), Output = (A,)>,
>(
    input: &[T],
    input_offset: &[u32],
    len: &[u32],
    output_a: &mut [A],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply((input[input_offset[0] as usize + global],));
        output_a[global] = output.0;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_unary_tuple2_kernel<
    T: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    Op: UnaryOp<(T,), Output = (A, B)>,
>(
    input: &[T],
    input_offset: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply((input[input_offset[0] as usize + global],));
        output_a[global] = output.0;
        output_b[global] = output.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_unary_tuple3_kernel<
    T: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: UnaryOp<(T,), Output = (A, B, C)>,
>(
    input: &[T],
    input_offset: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply((input[input_offset[0] as usize + global],));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
    }
}

macro_rules! define_transform_tuple_to_tuple_kernel {
    (
        $fn_name:ident,
        ($( $in_ty:ident : $input:ident : $input_offset:ident ),+),
        ($( $out_ty:ident : $output:ident : $field:tt ),+)
    ) => {
        #[allow(dead_code)]
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $( $in_ty: CubePrimitive, )+
            $( $out_ty: CubePrimitive, )+
            Op: UnaryOp<($( $in_ty, )+), Output = ($( $out_ty, )+)>,
        >(
            $( $input: &[$in_ty], )+
            $( $input_offset: &[u32], )+
            len: &[u32],
            $( $output: &mut [$out_ty], )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (len[0] as usize) {
                let output = Op::apply((
                    $( $input[$input_offset[0] as usize + global], )+
                ));
                $(
                    $output[global] = output.$field;
                )+
            }
        }
    };
}

define_transform_tuple_to_tuple_kernel!(
    transform_tuple2_to_tuple1_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset),
    (OutA: output_a: 0)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple2_to_tuple2_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset),
    (OutA: output_a: 0, OutB: output_b: 1)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple2_to_tuple3_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple3_to_tuple1_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset),
    (OutA: output_a: 0)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple3_to_tuple2_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset),
    (OutA: output_a: 0, OutB: output_b: 1)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple3_to_tuple3_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2)
);

macro_rules! define_tuple_predicate_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Pred: PredicateOp1<($( $ty, )+)>>(
            $( $input: &[$ty], )+
            len: &[u32],
            invert: &[u32],
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (len[0] as usize) {
                let selected = Pred::apply((
                    $( $input[global], )+
                ));
                if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }
    };
}

define_tuple_predicate_flags_kernel!(tuple2_predicate_flags_kernel, (TyA: input_a, TyB: input_b));
define_tuple_predicate_flags_kernel!(
    tuple3_predicate_flags_kernel,
    (TyA: input_a, TyB: input_b, TyC: input_c)
);

macro_rules! define_tuple_adjacent_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Pred: PredicateOp2<($( $ty, )+)>>(
            $( $input: &[$ty], )+
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global + 1usize < flags.len() {
                if Pred::apply(($( $input[global], )+), ($( $input[global + 1usize], )+)) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }
    };
}

define_tuple_adjacent_flags_kernel!(tuple2_adjacent_flags_kernel, (TyA: input_a, TyB: input_b));
define_tuple_adjacent_flags_kernel!(tuple3_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c));

macro_rules! define_tuple_unique_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Pred: PredicateOp2<($( $ty, )+)>>(
            $( $input: &[$ty], )+
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if global == 0usize {
                    flags[global] = 1u32;
                } else if Pred::apply(($( $input[global - 1usize], )+), ($( $input[global], )+)) {
                    flags[global] = 0u32;
                } else {
                    flags[global] = 1u32;
                }
            }
        }
    };
}

define_tuple_unique_flags_kernel!(tuple2_unique_flags_kernel, (TyA: input_a, TyB: input_b));
define_tuple_unique_flags_kernel!(tuple3_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c));

macro_rules! define_tuple_mismatch_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $left:ident / $right:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Eq: PredicateOp2<($( $ty, )+)>>(
            $( $left: &[$ty], )+
            $( $right: &[$ty], )+
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if Eq::apply(($( $left[global], )+), ($( $right[global], )+)) {
                    flags[global] = 0u32;
                } else {
                    flags[global] = 1u32;
                }
            }
        }
    };
}

define_tuple_mismatch_flags_kernel!(tuple2_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b));
define_tuple_mismatch_flags_kernel!(tuple3_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c));

macro_rules! define_tuple_sorted_break_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Less: PredicateOp2<($( $ty, )+)>>(
            $( $input: &[$ty], )+
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if global > 0usize
                    && Less::apply(($( $input[global], )+), ($( $input[global - 1usize], )+))
                {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }
    };
}

define_tuple_sorted_break_flags_kernel!(tuple2_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b));
define_tuple_sorted_break_flags_kernel!(tuple3_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c));

macro_rules! define_tuple_bound_flags_kernel {
    (
        $lower_fn:ident,
        $upper_fn:ident,
        ($first_ty:ident : $first_input:ident / $first_value:ident $(, $ty:ident : $input:ident / $value:ident )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $lower_fn<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: PredicateOp2<($first_ty, $( $ty, )*)>>(
            $first_input: &[$first_ty],
            $( $input: &[$ty], )*
            $first_value: &[$first_ty],
            $( $value: &[$ty], )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if Less::apply(
                    ($first_input[global], $( $input[global], )*),
                    ($first_value[0], $( $value[0], )*),
                ) {
                    flags[global] = 0u32;
                } else {
                    flags[global] = 1u32;
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $upper_fn<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: PredicateOp2<($first_ty, $( $ty, )*)>>(
            $first_input: &[$first_ty],
            $( $input: &[$ty], )*
            $first_value: &[$first_ty],
            $( $value: &[$ty], )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if Less::apply(
                    ($first_value[0], $( $value[0], )*),
                    ($first_input[global], $( $input[global], )*),
                ) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }
    };
}

define_tuple_bound_flags_kernel!(tuple2_lower_bound_flags_kernel, tuple2_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b));
define_tuple_bound_flags_kernel!(tuple3_lower_bound_flags_kernel, tuple3_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c));

macro_rules! define_tuple_membership_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_candidate:ident / $first_sorted:ident $(, $ty:ident : $candidate:ident / $sorted:ident )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: PredicateOp2<($first_ty, $( $ty, )*)>>(
            $first_candidate: &[$first_ty],
            $( $candidate: &[$ty], )*
            $first_sorted: &[$first_ty],
            $( $sorted: &[$ty], )*
            keep_present: &[u32],
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let candidate_first = RuntimeCell::<usize>::new(0usize);
                let candidate_count = RuntimeCell::<usize>::new($first_candidate.len());
                while candidate_count.read() > 0usize {
                    let step = candidate_count.read() / 2usize;
                    let mid = candidate_first.read() + step;
                    if Less::apply(
                        ($first_candidate[mid], $( $candidate[mid], )*),
                        ($first_candidate[global], $( $candidate[global], )*),
                    ) {
                        candidate_first.store(mid + 1usize);
                        candidate_count.store(candidate_count.read() - step - 1usize);
                    } else {
                        candidate_count.store(step);
                    }
                }

                let sorted_first = RuntimeCell::<usize>::new(0usize);
                let sorted_count = RuntimeCell::<usize>::new($first_sorted.len());
                while sorted_count.read() > 0usize {
                    let step = sorted_count.read() / 2usize;
                    let mid = sorted_first.read() + step;
                    if Less::apply(
                        ($first_sorted[mid], $( $sorted[mid], )*),
                        ($first_candidate[global], $( $candidate[global], )*),
                    ) {
                        sorted_first.store(mid + 1usize);
                        sorted_count.store(sorted_count.read() - step - 1usize);
                    } else {
                        sorted_count.store(step);
                    }
                }

                let sorted_after = RuntimeCell::<usize>::new(0usize);
                let sorted_after_count = RuntimeCell::<usize>::new($first_sorted.len());
                while sorted_after_count.read() > 0usize {
                    let step = sorted_after_count.read() / 2usize;
                    let mid = sorted_after.read() + step;
                    if !Less::apply(
                        ($first_candidate[global], $( $candidate[global], )*),
                        ($first_sorted[mid], $( $sorted[mid], )*),
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
    };
}

define_tuple_membership_flags_kernel!(tuple2_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b));
define_tuple_membership_flags_kernel!(tuple3_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c));

macro_rules! define_tuple_minmax_kernels {
    (
        $element_fn:ident,
        $index_fn:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $element_fn<$( $ty: CubePrimitive, )+ Less: PredicateOp2<($( $ty, )+)>>(
            $( $input: &[$ty], )+
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
                    if Less::apply(($( $input[i.read()], )+), ($( $input[min_index.read()], )+)) {
                        min_index.store(i.read());
                    }
                    if Less::apply(($( $input[max_index.read()], )+), ($( $input[i.read()], )+)) {
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
                        if Less::apply(($( $input[other_min], )+), ($( $input[current_min], )+)) {
                            min_indices[unit] = other_min as u32;
                        }

                        let other_max = max_indices[unit + stride.read()] as usize;
                        let current_max = max_indices[unit] as usize;
                        if Less::apply(($( $input[current_max], )+), ($( $input[other_max], )+)) {
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
        pub(crate) fn $index_fn<$( $ty: CubePrimitive, )+ Less: PredicateOp2<($( $ty, )+)>>(
            $( $input: &[$ty], )+
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
                    if Less::apply(($( $input[candidate_min], )+), ($( $input[min_index.read()], )+)) {
                        min_index.store(candidate_min);
                    }
                    if Less::apply(($( $input[max_index.read()], )+), ($( $input[candidate_max], )+)) {
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
                        if Less::apply(($( $input[other_min], )+), ($( $input[current_min], )+)) {
                            min_indices[unit] = other_min as u32;
                        }

                        let other_max = max_indices[unit + stride.read()] as usize;
                        let current_max = max_indices[unit] as usize;
                        if Less::apply(($( $input[current_max], )+), ($( $input[other_max], )+)) {
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
    };
}

define_tuple_minmax_kernels!(tuple2_minmax_element_partials_kernel, tuple2_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b));
define_tuple_minmax_kernels!(tuple3_minmax_element_partials_kernel, tuple3_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c));

macro_rules! define_tuple_find_first_of_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_input:ident / $first_needle:ident $(, $ty:ident : $input:ident / $needle:ident )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Eq: PredicateOp2<($first_ty, $( $ty, )*)>>(
            $first_input: &[$first_ty],
            $( $input: &[$ty], )*
            $first_needle: &[$first_ty],
            $( $needle: &[$ty], )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let needle_index = RuntimeCell::<usize>::new(0usize);
                let found = RuntimeCell::<u32>::new(0u32);
                while needle_index.read() < $first_needle.len() {
                    if Eq::apply(
                        ($first_input[global], $( $input[global], )*),
                        ($first_needle[needle_index.read()], $( $needle[needle_index.read()], )*),
                    ) {
                        found.store(1u32);
                        needle_index.store($first_needle.len());
                    } else {
                        needle_index.store(needle_index.read() + 1usize);
                    }
                }
                flags[global] = found.read();
            }
        }
    };
}

define_tuple_find_first_of_flags_kernel!(tuple2_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b));
define_tuple_find_first_of_flags_kernel!(tuple3_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c));

macro_rules! define_tuple_lexicographical_diff_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $left:ident / $right:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Less: PredicateOp2<($( $ty, )+)>>(
            $( $left: &[$ty], )+
            $( $right: &[$ty], )+
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let lhs = ($( $left[global], )+);
                let rhs = ($( $right[global], )+);
                if Less::apply(lhs, rhs) || Less::apply(rhs, lhs) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }
    };
}

macro_rules! define_tuple_lexicographical_compare_at_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $left:ident / $right:ident ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Less: PredicateOp2<($( $ty, )+)>>(
            $( $left: &[$ty], )+
            $( $right: &[$ty], )+
            index: &[u32],
            output: &mut [u32],
        ) {
            if UNIT_POS == 0 {
                let i = index[0] as usize;
                if Less::apply(($( $left[i], )+), ($( $right[i], )+)) {
                    output[0] = 1u32;
                } else {
                    output[0] = 0u32;
                }
            }
        }
    };
}

define_tuple_lexicographical_diff_flags_kernel!(tuple2_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b));
define_tuple_lexicographical_diff_flags_kernel!(tuple3_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c));

define_tuple_lexicographical_compare_at_kernel!(tuple2_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b));
define_tuple_lexicographical_compare_at_kernel!(tuple3_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c));

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_expr_flags_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp1<T>,
>(
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let value = Expr::eval(input, indices, rhs, rhs_indices, unit);
        values[unit] = value;
        let selected = Pred::apply(value);
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[unit] = 1u32;
        } else {
            flags[unit] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_expr_flag_only_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp1<T>,
>(
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let selected = Pred::apply(Expr::eval(input, indices, rhs, rhs_indices, unit));
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[unit] = 1u32;
        } else {
            flags[unit] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_stencil_expr_flags_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    StencilExpr: GpuExpr<S>,
    Pred: PredicateOp1<S>,
>(
    value_input: &[T],
    value_indices: &[u32],
    value_rhs: &[T],
    value_rhs_indices: &[u32],
    stencil_input: &[S],
    stencil_indices: &[u32],
    stencil_rhs: &[S],
    stencil_rhs_indices: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let value = ValueExpr::eval(
            value_input,
            value_indices,
            value_rhs,
            value_rhs_indices,
            unit,
        );
        let stencil = StencilExpr::eval(
            stencil_input,
            stencil_indices,
            stencil_rhs,
            stencil_rhs_indices,
            unit,
        );
        values[unit] = value;
        let selected = Pred::apply(stencil);
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[unit] = 1u32;
        } else {
            flags[unit] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_if_stencil_expr_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    StencilExpr: GpuExpr<S>,
    Op: UnaryOp<T, Output = T>,
    Pred: PredicateOp1<S>,
>(
    value_input: &[T],
    value_indices: &[u32],
    value_rhs: &[T],
    value_rhs_indices: &[u32],
    stencil_input: &[S],
    stencil_indices: &[u32],
    stencil_rhs: &[S],
    stencil_rhs_indices: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let stencil = StencilExpr::eval(
            stencil_input,
            stencil_indices,
            stencil_rhs,
            stencil_rhs_indices,
            unit,
        );
        if Pred::apply(stencil) {
            let value = ValueExpr::eval(
                value_input,
                value_indices,
                value_rhs,
                value_rhs_indices,
                unit,
            );
            output[unit] = Op::apply(value);
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn gather_expr_kernel<T: CubePrimitive, IndexExpr: GpuExpr<u32>>(
    output: &mut [T],
    index_input: &[u32],
    index_indices: &[u32],
    index_rhs: &[u32],
    index_rhs_indices: &[u32],
    input: &[T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output.len() {
        let index = IndexExpr::eval(
            index_input,
            index_indices,
            index_rhs,
            index_rhs_indices,
            global,
        );
        output[global] = input[index as usize];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn gather_device_expr_kernel<
    T: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
>(
    output: &mut [T],
    value_input: &[T],
    value_indices: &[u32],
    value_rhs: &[T],
    value_rhs_indices: &[u32],
    index_input: &[u32],
    index_indices: &[u32],
    index_rhs: &[u32],
    index_rhs_indices: &[u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output.len() {
        let index = IndexExpr::eval(
            index_input,
            index_indices,
            index_rhs,
            index_rhs_indices,
            global,
        );
        output[global] = ValueExpr::eval(
            value_input,
            value_indices,
            value_rhs,
            value_rhs_indices,
            index as usize,
        );
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scatter_expr_kernel<
    T: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
>(
    value_input: &[T],
    value_indices: &[u32],
    value_rhs: &[T],
    value_rhs_indices: &[u32],
    index_input: &[u32],
    index_indices: &[u32],
    index_rhs: &[u32],
    index_rhs_indices: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let value = ValueExpr::eval(
            value_input,
            value_indices,
            value_rhs,
            value_rhs_indices,
            global,
        );
        let index = IndexExpr::eval(
            index_input,
            index_indices,
            index_rhs,
            index_rhs_indices,
            global,
        );
        output[index as usize] = value;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scatter_if_expr_kernel<
    T: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
    Pred: PredicateOp1<T>,
>(
    value_input: &[T],
    value_indices: &[u32],
    value_rhs: &[T],
    value_rhs_indices: &[u32],
    index_input: &[u32],
    index_indices: &[u32],
    index_rhs: &[u32],
    index_rhs_indices: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let value = ValueExpr::eval(
            value_input,
            value_indices,
            value_rhs,
            value_rhs_indices,
            global,
        );
        if Pred::apply(value) {
            let index = IndexExpr::eval(
                index_input,
                index_indices,
                index_rhs,
                index_rhs_indices,
                global,
            );
            output[index as usize] = value;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn compact_count_kernel(positions: &[u32], count: &mut [u32]) {
    if UNIT_POS == 0 {
        count[0] = positions[positions.len() - 1usize];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn u32_block_inclusive_scan_kernel(
    input: &[u32],
    len: &[u32],
    output: &mut [u32],
    block_sums: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values[unit] = input[global];
    } else {
        values[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() {
            addend.store(values[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            values[unit] = values[unit] + addend.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output[global] = values[unit];
    }
    if unit == cube_dim - 1usize {
        block_sums[CUBE_POS as usize] = values[unit];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn u32_add_block_prefix_kernel(block_prefixes: &[u32], len: &[u32], output: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;

    if block > 0usize && global < (len[0] as usize) {
        output[global] = output[global] + block_prefixes[block - 1usize];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scalar_inclusive_scan_block_kernel<T: CubePrimitive, Op: BinaryOp2<T>>(
    input: &[T],
    len: &[u32],
    output: &mut [T],
    block_sums: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = Shared::<[T]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values[unit] = input[global];
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend = RuntimeCell::<T>::new(values[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend.store(values[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                values[unit] = Op::apply(addend.read(), values[unit]);
            } else {
                values[unit] = addend.read();
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output[global] = values[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums[block] = values[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scalar_scan_add_block_prefix_kernel<T: CubePrimitive, Op: BinaryOp2<T>>(
    block_prefixes: &[T],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    if block > 0usize && global < len[0] as usize {
        output[global] = Op::apply(block_prefixes[block - 1usize], output[global]);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scalar_reduce_last_finalize_kernel<T: CubePrimitive, Op: BinaryOp2<T>>(
    partial: &[T],
    len: &[u32],
    init: &[T],
    output: &mut [T],
) {
    if UNIT_POS == 0 {
        output[0] = Op::apply(init[0], partial[len[0] as usize - 1usize]);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple1_device_inclusive_scan_expr_block_kernel<
    A: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    Op: BinaryOp2<(A,)>,
>(
    a_slot0: &[A],
    a_slot1: &[A],
    a_slot2: &[A],
    a_slot3: &[A],
    a_slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [A],
    block_sums_a: &mut [A],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, global);
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply((addend_a.read(),), (values_a[unit],));
                values_a[unit] = value.0;
            } else {
                values_a[unit] = addend_a.read();
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple1_inclusive_scan_block_kernel<A: CubePrimitive, Op: BinaryOp2<(A,)>>(
    input_a: &[A],
    len: &[u32],
    output_a: &mut [A],
    block_sums_a: &mut [A],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = input_a[global];
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply((addend_a.read(),), (values_a[unit],));
                values_a[unit] = value.0;
            } else {
                values_a[unit] = addend_a.read();
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple1_scan_add_block_prefix_kernel<A: CubePrimitive, Op: BinaryOp2<(A,)>>(
    block_prefixes_a: &[A],
    len: &[u32],
    output_a: &mut [A],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;

    if block > 0usize && global < (len[0] as usize) {
        let value = Op::apply((block_prefixes_a[block - 1usize],), (output_a[global],));
        output_a[global] = value.0;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple1_scan_make_exclusive_kernel<A: CubePrimitive, Op: BinaryOp2<(A,)>>(
    inclusive_a: &[A],
    init_a: &[A],
    output_a: &mut [A],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output_a.len() {
        if global == 0usize {
            output_a[global] = init_a[0];
        } else {
            let value = Op::apply((init_a[0],), (inclusive_a[global - 1usize],));
            output_a[global] = value.0;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_device_inclusive_scan_expr_block_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp2<(A, B)>,
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
    block_sums_a: &mut [A],
    block_sums_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, global);
        values_b[unit] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, global);
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_b = RuntimeCell::<B>::new(values_b[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply(
                    (addend_a.read(), addend_b.read()),
                    (values_a[unit], values_b[unit]),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
            } else {
                values_a[unit] = addend_a.read();
                values_b[unit] = addend_b.read();
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        output_b[global] = values_b[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
            block_sums_b[block] = values_b[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_inclusive_scan_block_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    Op: BinaryOp2<(A, B)>,
>(
    input_a: &[A],
    input_b: &[B],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    block_sums_a: &mut [A],
    block_sums_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = input_a[global];
        values_b[unit] = input_b[global];
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_b = RuntimeCell::<B>::new(values_b[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply(
                    (addend_a.read(), addend_b.read()),
                    (values_a[unit], values_b[unit]),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
            } else {
                values_a[unit] = addend_a.read();
                values_b[unit] = addend_b.read();
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        output_b[global] = values_b[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
            block_sums_b[block] = values_b[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_scan_add_block_prefix_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    Op: BinaryOp2<(A, B)>,
>(
    block_prefixes_a: &[A],
    block_prefixes_b: &[B],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;

    if block > 0usize && global < (len[0] as usize) {
        let value = Op::apply(
            (
                block_prefixes_a[block - 1usize],
                block_prefixes_b[block - 1usize],
            ),
            (output_a[global], output_b[global]),
        );
        output_a[global] = value.0;
        output_b[global] = value.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_scan_make_exclusive_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    Op: BinaryOp2<(A, B)>,
>(
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
            let value = Op::apply(
                (init_a[0], init_b[0]),
                (inclusive_a[global - 1usize], inclusive_b[global - 1usize]),
            );
            output_a[global] = value.0;
            output_b[global] = value.1;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_device_inclusive_scan_expr_block_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    Op: BinaryOp2<(A, B, C)>,
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
    block_sums_a: &mut [A],
    block_sums_b: &mut [B],
    block_sums_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, global);
        values_b[unit] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, global);
        values_c[unit] = ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, global);
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_b = RuntimeCell::<B>::new(values_b[unit]);
        let addend_c = RuntimeCell::<C>::new(values_c[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_c.store(values_c[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply(
                    (addend_a.read(), addend_b.read(), addend_c.read()),
                    (values_a[unit], values_b[unit], values_c[unit]),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
                values_c[unit] = value.2;
            } else {
                values_a[unit] = addend_a.read();
                values_b[unit] = addend_b.read();
                values_c[unit] = addend_c.read();
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        output_b[global] = values_b[unit];
        output_c[global] = values_c[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
            block_sums_b[block] = values_b[unit];
            block_sums_c[block] = values_c[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_inclusive_scan_block_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: BinaryOp2<(A, B, C)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_c: &[C],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    block_sums_a: &mut [A],
    block_sums_b: &mut [B],
    block_sums_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = input_a[global];
        values_b[unit] = input_b[global];
        values_c[unit] = input_c[global];
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend_a = RuntimeCell::<A>::new(values_a[unit]);
        let addend_b = RuntimeCell::<B>::new(values_b[unit]);
        let addend_c = RuntimeCell::<C>::new(values_c[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_c.store(values_c[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply(
                    (addend_a.read(), addend_b.read(), addend_c.read()),
                    (values_a[unit], values_b[unit], values_c[unit]),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
                values_c[unit] = value.2;
            } else {
                values_a[unit] = addend_a.read();
                values_b[unit] = addend_b.read();
                values_c[unit] = addend_c.read();
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output_a[global] = values_a[unit];
        output_b[global] = values_b[unit];
        output_c[global] = values_c[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
            block_sums_b[block] = values_b[unit];
            block_sums_c[block] = values_c[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_scan_add_block_prefix_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: BinaryOp2<(A, B, C)>,
>(
    block_prefixes_a: &[A],
    block_prefixes_b: &[B],
    block_prefixes_c: &[C],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;

    if block > 0usize && global < (len[0] as usize) {
        let value = Op::apply(
            (
                block_prefixes_a[block - 1usize],
                block_prefixes_b[block - 1usize],
                block_prefixes_c[block - 1usize],
            ),
            (output_a[global], output_b[global], output_c[global]),
        );
        output_a[global] = value.0;
        output_b[global] = value.1;
        output_c[global] = value.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_scan_make_exclusive_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: BinaryOp2<(A, B, C)>,
>(
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
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn compact_scatter_kernel<T: CubePrimitive>(
    flags: &[u32],
    positions: &[u32],
    values: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let position = positions[global];
        output[(position - 1u32) as usize] = values[global];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn compact_scatter_device_expr_kernel<T: CubePrimitive, Expr: DeviceGpuExpr<T>>(
    flags: &[u32],
    positions: &[u32],
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let position = positions[global];
        output[(position - 1u32) as usize] =
            Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn unique_by_key_device_expr_flags_kernel<
    K: CubePrimitive,
    Expr: DeviceGpuExpr<K>,
    Pred: PredicateOp2<K>,
>(
    slot0: &[K],
    slot1: &[K],
    slot2: &[K],
    slot3: &[K],
    slot_offsets: &[u32],
    len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    if global < logical_len {
        if global == 0 {
            flags[global] = 1u32;
        } else {
            let previous = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global - 1usize);
            let current = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
            if Pred::apply(previous, current) {
                flags[global] = 0u32;
            } else {
                flags[global] = 1u32;
            }
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn compact_rejected_scatter_kernel<T: CubePrimitive>(
    flags: &[u32],
    positions: &[u32],
    values: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] == 0u32 {
        let selected_before_or_at = positions[global];
        let rejected_before = (global as u32) - selected_before_or_at;
        output[rejected_before as usize] = values[global];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn compact_scatter_pair_kernel<A: CubePrimitive, B: CubePrimitive>(
    flags: &[u32],
    positions: &[u32],
    values_a: &[A],
    values_b: &[B],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let position = (positions[global] - 1u32) as usize;
        output_a[position] = values_a[global];
        output_b[position] = values_b[global];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn adjacent_difference_expr_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Op: BinaryOp2<T>,
>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        if global == 0usize {
            output[global] = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
        } else {
            output[global] = Op::apply(
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global),
                Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global - 1usize),
            );
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_adjacent_difference_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp2<(A, B)>,
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
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let current = (
            ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, global),
            ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, global),
        );
        let output = if global == 0usize {
            current
        } else {
            Op::apply(
                current,
                (
                    ExprA::eval(
                        a_slot0,
                        a_slot1,
                        a_slot2,
                        a_slot3,
                        a_slot_offsets,
                        global - 1usize,
                    ),
                    ExprB::eval(
                        b_slot0,
                        b_slot1,
                        b_slot2,
                        b_slot3,
                        b_slot_offsets,
                        global - 1usize,
                    ),
                ),
            )
        };
        output_a[global] = output.0;
        output_b[global] = output.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_adjacent_difference_expr_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    Op: BinaryOp2<(A, B, C)>,
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
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let current = (
            ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, global),
            ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, global),
            ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, global),
        );
        let output = if global == 0usize {
            current
        } else {
            Op::apply(
                current,
                (
                    ExprA::eval(
                        a_slot0,
                        a_slot1,
                        a_slot2,
                        a_slot3,
                        a_slot_offsets,
                        global - 1usize,
                    ),
                    ExprB::eval(
                        b_slot0,
                        b_slot1,
                        b_slot2,
                        b_slot3,
                        b_slot_offsets,
                        global - 1usize,
                    ),
                    ExprC::eval(
                        c_slot0,
                        c_slot1,
                        c_slot2,
                        c_slot3,
                        c_slot_offsets,
                        global - 1usize,
                    ),
                ),
            )
        };
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn inclusive_scan_by_key_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    KeyEq: crate::detail::op::kernel::PredicateOp2<K>,
    Op: BinaryOp2<T>,
>(
    keys: &[K],
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(unit);
        while start.read() > 0usize && KeyEq::apply(keys[start.read() - 1usize], keys[unit]) {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(Expr::eval(input, indices, rhs, rhs_indices, start.read()));
        let index = RuntimeCell::<usize>::new(start.read() + 1usize);
        while index.read() <= unit {
            acc.store(Op::apply(
                acc.read(),
                Expr::eval(input, indices, rhs, rhs_indices, index.read()),
            ));
            index.store(index.read() + 1usize);
        }

        output[unit] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn inclusive_scan_by_key_expr_keys_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: GpuExpr<K>,
    ValueExpr: GpuExpr<T>,
    KeyEq: crate::detail::op::kernel::PredicateOp2<K>,
    Op: BinaryOp2<T>,
>(
    key_input: &[K],
    key_indices: &[u32],
    key_rhs: &[K],
    key_rhs_indices: &[u32],
    value_input: &[T],
    value_indices: &[u32],
    value_rhs: &[T],
    value_rhs_indices: &[u32],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let current_key = KeyExpr::eval(key_input, key_indices, key_rhs, key_rhs_indices, unit);
        let start = RuntimeCell::<usize>::new(unit);
        while start.read() > 0usize
            && KeyEq::apply(
                KeyExpr::eval(
                    key_input,
                    key_indices,
                    key_rhs,
                    key_rhs_indices,
                    start.read() - 1usize,
                ),
                current_key,
            )
        {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(ValueExpr::eval(
            value_input,
            value_indices,
            value_rhs,
            value_rhs_indices,
            start.read(),
        ));
        let index = RuntimeCell::<usize>::new(start.read() + 1usize);
        while index.read() <= unit {
            acc.store(Op::apply(
                acc.read(),
                ValueExpr::eval(
                    value_input,
                    value_indices,
                    value_rhs,
                    value_rhs_indices,
                    index.read(),
                ),
            ));
            index.store(index.read() + 1usize);
        }

        output[unit] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_by_key_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    KeyEq: crate::detail::op::kernel::PredicateOp2<K>,
    Op: BinaryOp2<T>,
>(
    keys: &[K],
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    init: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(unit);
        while start.read() > 0usize && KeyEq::apply(keys[start.read() - 1usize], keys[unit]) {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(init[0]);
        let index = RuntimeCell::<usize>::new(start.read());
        while index.read() < unit {
            acc.store(Op::apply(
                acc.read(),
                Expr::eval(input, indices, rhs, rhs_indices, index.read()),
            ));
            index.store(index.read() + 1usize);
        }

        output[unit] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_by_key_expr_keys_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: GpuExpr<K>,
    ValueExpr: GpuExpr<T>,
    KeyEq: crate::detail::op::kernel::PredicateOp2<K>,
    Op: BinaryOp2<T>,
>(
    key_input: &[K],
    key_indices: &[u32],
    key_rhs: &[K],
    key_rhs_indices: &[u32],
    value_input: &[T],
    value_indices: &[u32],
    value_rhs: &[T],
    value_rhs_indices: &[u32],
    len: &[u32],
    init: &[T],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < (len[0] as usize) {
        let current_key = KeyExpr::eval(key_input, key_indices, key_rhs, key_rhs_indices, unit);
        let start = RuntimeCell::<usize>::new(unit);
        while start.read() > 0usize
            && KeyEq::apply(
                KeyExpr::eval(
                    key_input,
                    key_indices,
                    key_rhs,
                    key_rhs_indices,
                    start.read() - 1usize,
                ),
                current_key,
            )
        {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(init[0]);
        let index = RuntimeCell::<usize>::new(start.read());
        while index.read() < unit {
            acc.store(Op::apply(
                acc.read(),
                ValueExpr::eval(
                    value_input,
                    value_indices,
                    value_rhs,
                    value_rhs_indices,
                    index.read(),
                ),
            ));
            index.store(index.read() + 1usize);
        }

        output[unit] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_expr_partials_kernel<T: CubePrimitive, Expr: GpuExpr<T>, Op: BinaryOp2<T>>(
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    partials: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values = Shared::<[T]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partials.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc = RuntimeCell::<T>::new(Expr::eval(input, indices, rhs, rhs_indices, 0));

    while i.read() < logical_len {
        let value = Expr::eval(input, indices, rhs, rhs_indices, i.read());
        if has_value.read() != 0 {
            acc.store(Op::apply(acc.read(), value));
        } else {
            acc.store(value);
            has_value.store(1u32);
        }
        i.store(i.read() + step);
    }

    values[unit] = acc.read();
    if has_value.read() != 0 {
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                values[unit] = Op::apply(values[unit], values[unit + stride.read()]);
            } else {
                values[unit] = values[unit + stride.read()];
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 && valid[0] != 0 {
        partials[CUBE_POS as usize] = values[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple1_reduce_last_finalize_kernel<A: CubePrimitive, Op: BinaryOp2<(A,)>>(
    partial_a: &[A],
    len: &[u32],
    init_a: &[A],
    output_a: &mut [A],
) {
    if UNIT_POS == 0 {
        let last = len[0] as usize - 1usize;
        let output = Op::apply((init_a[0],), (partial_a[last],));
        output_a[0] = output.0;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_device_reduce_expr_partials_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    Op: BinaryOp2<(A, B)>,
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
    partial_a: &mut [A],
    partial_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_a.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc_a = RuntimeCell::<A>::new(ExprA::eval(
        a_slot0,
        a_slot1,
        a_slot2,
        a_slot3,
        a_slot_offsets,
        0,
    ));
    let acc_b = RuntimeCell::<B>::new(ExprB::eval(
        b_slot0,
        b_slot1,
        b_slot2,
        b_slot3,
        b_slot_offsets,
        0,
    ));

    while i.read() < logical_len {
        let value = (
            ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, i.read()),
            ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, i.read()),
        );
        if has_value.read() != 0 {
            let acc = Op::apply((acc_a.read(), acc_b.read()), value);
            acc_a.store(acc.0);
            acc_b.store(acc.1);
        } else {
            acc_a.store(value.0);
            acc_b.store(value.1);
            has_value.store(1u32);
        }
        i.store(i.read() + step);
    }

    values_a[unit] = acc_a.read();
    values_b[unit] = acc_b.read();
    if has_value.read() != 0 {
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let acc = Op::apply(
                    (values_a[unit], values_b[unit]),
                    (
                        values_a[unit + stride.read()],
                        values_b[unit + stride.read()],
                    ),
                );
                values_a[unit] = acc.0;
                values_b[unit] = acc.1;
            } else {
                values_a[unit] = values_a[unit + stride.read()];
                values_b[unit] = values_b[unit + stride.read()];
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 && valid[0] != 0 {
        partial_a[CUBE_POS as usize] = values_a[0];
        partial_b[CUBE_POS as usize] = values_b[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_reduce_partials_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    Op: BinaryOp2<(A, B)>,
>(
    input_a: &[A],
    input_b: &[B],
    len: &[u32],
    partial_a: &mut [A],
    partial_b: &mut [B],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_a.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc_a = RuntimeCell::<A>::new(input_a[0]);
    let acc_b = RuntimeCell::<B>::new(input_b[0]);

    while i.read() < logical_len {
        let value = (input_a[i.read()], input_b[i.read()]);
        if has_value.read() != 0 {
            let acc = Op::apply((acc_a.read(), acc_b.read()), value);
            acc_a.store(acc.0);
            acc_b.store(acc.1);
        } else {
            acc_a.store(value.0);
            acc_b.store(value.1);
            has_value.store(1u32);
        }
        i.store(i.read() + step);
    }

    values_a[unit] = acc_a.read();
    values_b[unit] = acc_b.read();
    if has_value.read() != 0 {
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let acc = Op::apply(
                    (values_a[unit], values_b[unit]),
                    (
                        values_a[unit + stride.read()],
                        values_b[unit + stride.read()],
                    ),
                );
                values_a[unit] = acc.0;
                values_b[unit] = acc.1;
            } else {
                values_a[unit] = values_a[unit + stride.read()];
                values_b[unit] = values_b[unit + stride.read()];
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 && valid[0] != 0 {
        partial_a[CUBE_POS as usize] = values_a[0];
        partial_b[CUBE_POS as usize] = values_b[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_reduce_finalize_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    Op: BinaryOp2<(A, B)>,
>(
    partial_a: &[A],
    partial_b: &[B],
    init_a: &[A],
    init_b: &[B],
    output_a: &mut [A],
    output_b: &mut [B],
) {
    if UNIT_POS == 0 {
        let output = Op::apply((init_a[0], init_b[0]), (partial_a[0], partial_b[0]));
        output_a[0] = output.0;
        output_b[0] = output.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_device_reduce_expr_partials_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    ExprA: DeviceGpuExpr<A>,
    ExprB: DeviceGpuExpr<B>,
    ExprC: DeviceGpuExpr<C>,
    Op: BinaryOp2<(A, B, C)>,
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
    partial_a: &mut [A],
    partial_b: &mut [B],
    partial_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_a.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc_a = RuntimeCell::<A>::new(ExprA::eval(
        a_slot0,
        a_slot1,
        a_slot2,
        a_slot3,
        a_slot_offsets,
        0,
    ));
    let acc_b = RuntimeCell::<B>::new(ExprB::eval(
        b_slot0,
        b_slot1,
        b_slot2,
        b_slot3,
        b_slot_offsets,
        0,
    ));
    let acc_c = RuntimeCell::<C>::new(ExprC::eval(
        c_slot0,
        c_slot1,
        c_slot2,
        c_slot3,
        c_slot_offsets,
        0,
    ));

    while i.read() < logical_len {
        let value = (
            ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets, i.read()),
            ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets, i.read()),
            ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets, i.read()),
        );
        if has_value.read() != 0 {
            let acc = Op::apply((acc_a.read(), acc_b.read(), acc_c.read()), value);
            acc_a.store(acc.0);
            acc_b.store(acc.1);
            acc_c.store(acc.2);
        } else {
            acc_a.store(value.0);
            acc_b.store(value.1);
            acc_c.store(value.2);
            has_value.store(1u32);
        }
        i.store(i.read() + step);
    }

    values_a[unit] = acc_a.read();
    values_b[unit] = acc_b.read();
    values_c[unit] = acc_c.read();
    if has_value.read() != 0 {
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let acc = Op::apply(
                    (values_a[unit], values_b[unit], values_c[unit]),
                    (
                        values_a[unit + stride.read()],
                        values_b[unit + stride.read()],
                        values_c[unit + stride.read()],
                    ),
                );
                values_a[unit] = acc.0;
                values_b[unit] = acc.1;
                values_c[unit] = acc.2;
            } else {
                values_a[unit] = values_a[unit + stride.read()];
                values_b[unit] = values_b[unit + stride.read()];
                values_c[unit] = values_c[unit + stride.read()];
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 && valid[0] != 0 {
        partial_a[CUBE_POS as usize] = values_a[0];
        partial_b[CUBE_POS as usize] = values_b[0];
        partial_c[CUBE_POS as usize] = values_c[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_reduce_partials_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: BinaryOp2<(A, B, C)>,
>(
    input_a: &[A],
    input_b: &[B],
    input_c: &[C],
    len: &[u32],
    partial_a: &mut [A],
    partial_b: &mut [B],
    partial_c: &mut [C],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_a.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc_a = RuntimeCell::<A>::new(input_a[0]);
    let acc_b = RuntimeCell::<B>::new(input_b[0]);
    let acc_c = RuntimeCell::<C>::new(input_c[0]);

    while i.read() < logical_len {
        let value = (input_a[i.read()], input_b[i.read()], input_c[i.read()]);
        if has_value.read() != 0 {
            let acc = Op::apply((acc_a.read(), acc_b.read(), acc_c.read()), value);
            acc_a.store(acc.0);
            acc_b.store(acc.1);
            acc_c.store(acc.2);
        } else {
            acc_a.store(value.0);
            acc_b.store(value.1);
            acc_c.store(value.2);
            has_value.store(1u32);
        }
        i.store(i.read() + step);
    }

    values_a[unit] = acc_a.read();
    values_b[unit] = acc_b.read();
    values_c[unit] = acc_c.read();
    if has_value.read() != 0 {
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let acc = Op::apply(
                    (values_a[unit], values_b[unit], values_c[unit]),
                    (
                        values_a[unit + stride.read()],
                        values_b[unit + stride.read()],
                        values_c[unit + stride.read()],
                    ),
                );
                values_a[unit] = acc.0;
                values_b[unit] = acc.1;
                values_c[unit] = acc.2;
            } else {
                values_a[unit] = values_a[unit + stride.read()];
                values_b[unit] = values_b[unit + stride.read()];
                values_c[unit] = values_c[unit + stride.read()];
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 && valid[0] != 0 {
        partial_a[CUBE_POS as usize] = values_a[0];
        partial_b[CUBE_POS as usize] = values_b[0];
        partial_c[CUBE_POS as usize] = values_c[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_reduce_finalize_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: BinaryOp2<(A, B, C)>,
>(
    partial_a: &[A],
    partial_b: &[B],
    partial_c: &[C],
    init_a: &[A],
    init_b: &[B],
    init_c: &[C],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
) {
    if UNIT_POS == 0 {
        let output = Op::apply(
            (init_a[0], init_b[0], init_c[0]),
            (partial_a[0], partial_b[0], partial_c[0]),
        );
        output_a[0] = output.0;
        output_b[0] = output.1;
        output_c[0] = output.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn count_if_expr_partials_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp1<T>,
>(
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut counts = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partials.len();
    let count = RuntimeCell::<u32>::new(0u32);

    while i.read() < logical_len {
        if Pred::apply(Expr::eval(input, indices, rhs, rhs_indices, i.read())) {
            count.store(count.read() + 1u32);
        }
        i.store(i.read() + step);
    }

    counts[unit] = count.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() {
            counts[unit] = counts[unit] + counts[unit + stride.read()];
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 {
        partials[CUBE_POS as usize] = counts[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn sum_u32_partials_kernel(input: &[u32], partials: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let mut counts = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partials.len();
    let count = RuntimeCell::<u32>::new(0u32);

    while i.read() < input.len() {
        count.store(count.read() + input[i.read()]);
        i.store(i.read() + step);
    }

    counts[unit] = count.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() {
            counts[unit] = counts[unit] + counts[unit + stride.read()];
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 {
        partials[CUBE_POS as usize] = counts[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn find_if_expr_partials_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp1<T>,
>(
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    invert: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut best_indices = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partials.len();
    let best = RuntimeCell::<u32>::new(len[0]);

    while i.read() < logical_len {
        let matches = Pred::apply(Expr::eval(input, indices, rhs, rhs_indices, i.read()));
        if (invert[0] == 0u32 && matches) || (invert[0] != 0u32 && !matches) {
            if (i.read() as u32) < best.read() {
                best.store(i.read() as u32);
            }
        }
        i.store(i.read() + step);
    }

    best_indices[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && best_indices[unit + stride.read()] < best_indices[unit] {
            best_indices[unit] = best_indices[unit + stride.read()];
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 {
        partials[CUBE_POS as usize] = best_indices[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn min_u32_partials_kernel(input: &[u32], partials: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let mut best_indices = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partials.len();
    let best = RuntimeCell::<u32>::new(input[0]);

    while i.read() < input.len() {
        if input[i.read()] < best.read() {
            best.store(input[i.read()]);
        }
        i.store(i.read() + step);
    }

    best_indices[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && best_indices[unit + stride.read()] < best_indices[unit] {
            best_indices[unit] = best_indices[unit + stride.read()];
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 {
        partials[CUBE_POS as usize] = best_indices[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn adjacent_find_expr_partials_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp2<T>,
>(
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    partials: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut best_indices = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partials.len();
    let best = RuntimeCell::<u32>::new(len[0]);

    while i.read() + 1usize < logical_len {
        if Pred::apply(
            Expr::eval(input, indices, rhs, rhs_indices, i.read()),
            Expr::eval(input, indices, rhs, rhs_indices, i.read() + 1usize),
        ) {
            if (i.read() as u32) < best.read() {
                best.store(i.read() as u32);
            }
        }
        i.store(i.read() + step);
    }

    best_indices[unit] = best.read();
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && best_indices[unit + stride.read()] < best_indices[unit] {
            best_indices[unit] = best_indices[unit + stride.read()];
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 {
        partials[CUBE_POS as usize] = best_indices[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn minmax_element_expr_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Less: PredicateOp2<T>,
>(
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    output: &mut [u32],
) {
    if UNIT_POS == 0 {
        let min_index = RuntimeCell::<usize>::new(0usize);
        let max_index = RuntimeCell::<usize>::new(0usize);
        let index = RuntimeCell::<usize>::new(1usize);

        while index.read() < (len[0] as usize) {
            if Less::apply(
                Expr::eval(input, indices, rhs, rhs_indices, index.read()),
                Expr::eval(input, indices, rhs, rhs_indices, min_index.read()),
            ) {
                min_index.store(index.read());
            }
            if Less::apply(
                Expr::eval(input, indices, rhs, rhs_indices, max_index.read()),
                Expr::eval(input, indices, rhs, rhs_indices, index.read()),
            ) {
                max_index.store(index.read());
            }
            index.store(index.read() + 1usize);
        }

        output[0] = min_index.read() as u32;
        output[1] = max_index.read() as u32;
    }
}
