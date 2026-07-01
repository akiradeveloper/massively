use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp},
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
pub(crate) fn device_collect_expr_into_block_kernel<T: CubePrimitive, Expr: DeviceGpuExpr<T>>(
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    len: &[u32],
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        output[output_offset[0] as usize + global] =
            Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
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
    env: Op::Env,
    input: &[T],
    input_offset: &[u32],
    len: &[u32],
    output_a: &mut [A],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(env, (input[input_offset[0] as usize + global],));
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
    env: Op::Env,
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
        let output = Op::apply(env, (input[input_offset[0] as usize + global],));
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
    env: Op::Env,
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
        let output = Op::apply(env, (input[input_offset[0] as usize + global],));
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
            env: Op::Env,
            $( $input: &[$in_ty], )+
            $( $input_offset: &[u32], )+
            len: &[u32],
            $( $output: &mut [$out_ty], )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (len[0] as usize) {
                let output = Op::apply(env, (
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
    transform_tuple1_to_tuple4_kernel,
    (TyA: input_a: input_a_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple1_to_tuple5_kernel,
    (TyA: input_a: input_a_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple1_to_tuple6_kernel,
    (TyA: input_a: input_a_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple1_to_tuple7_kernel,
    (TyA: input_a: input_a_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
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
define_transform_tuple_to_tuple_kernel!(
    transform_tuple4_to_tuple1_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset
    ),
    (OutA: output_a: 0)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple5_to_tuple1_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset
    ),
    (OutA: output_a: 0)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple6_to_tuple1_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset, TyF: input_f: input_f_offset
    ),
    (OutA: output_a: 0)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple7_to_tuple1_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset,
        TyD: input_d: input_d_offset, TyE: input_e: input_e_offset, TyF: input_f: input_f_offset,
        TyG: input_g: input_g_offset
    ),
    (OutA: output_a: 0)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple7_to_tuple2_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset,
        TyD: input_d: input_d_offset, TyE: input_e: input_e_offset, TyF: input_f: input_f_offset,
        TyG: input_g: input_g_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple7_to_tuple3_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset,
        TyD: input_d: input_d_offset, TyE: input_e: input_e_offset, TyF: input_f: input_f_offset,
        TyG: input_g: input_g_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple7_to_tuple4_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset,
        TyD: input_d: input_d_offset, TyE: input_e: input_e_offset, TyF: input_f: input_f_offset,
        TyG: input_g: input_g_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple7_to_tuple5_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset,
        TyD: input_d: input_d_offset, TyE: input_e: input_e_offset, TyF: input_f: input_f_offset,
        TyG: input_g: input_g_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple7_to_tuple6_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset,
        TyD: input_d: input_d_offset, TyE: input_e: input_e_offset, TyF: input_f: input_f_offset,
        TyG: input_g: input_g_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple7_to_tuple7_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset,
        TyD: input_d: input_d_offset, TyE: input_e: input_e_offset, TyF: input_f: input_f_offset,
        TyG: input_g: input_g_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple2_predicate_device_expr_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    ExprA: DeviceGpuExpr<TyA>,
    ExprB: DeviceGpuExpr<TyB>,
    Pred: PredicateOp<(TyA, TyB)>,
>(
    env: Pred::Env,
    input_a_slot0: &[TyA],
    input_a_slot1: &[TyA],
    input_a_slot2: &[TyA],
    input_a_slot3: &[TyA],
    input_a_slot_offsets: &[u32],
    input_b_slot0: &[TyB],
    input_b_slot1: &[TyB],
    input_b_slot2: &[TyB],
    input_b_slot3: &[TyB],
    input_b_slot_offsets: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply(
            env,
            (
                ExprA::eval(
                    input_a_slot0,
                    input_a_slot1,
                    input_a_slot2,
                    input_a_slot3,
                    input_a_slot_offsets,
                    global,
                ),
                ExprB::eval(
                    input_b_slot0,
                    input_b_slot1,
                    input_b_slot2,
                    input_b_slot3,
                    input_b_slot_offsets,
                    global,
                ),
            ),
        );
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple3_predicate_device_expr_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    ExprA: DeviceGpuExpr<TyA>,
    ExprB: DeviceGpuExpr<TyB>,
    ExprC: DeviceGpuExpr<TyC>,
    Pred: PredicateOp<(TyA, TyB, TyC)>,
>(
    env: Pred::Env,
    input_a_slot0: &[TyA],
    input_a_slot1: &[TyA],
    input_a_slot2: &[TyA],
    input_a_slot3: &[TyA],
    input_a_slot_offsets: &[u32],
    input_b_slot0: &[TyB],
    input_b_slot1: &[TyB],
    input_b_slot2: &[TyB],
    input_b_slot3: &[TyB],
    input_b_slot_offsets: &[u32],
    input_c_slot0: &[TyC],
    input_c_slot1: &[TyC],
    input_c_slot2: &[TyC],
    input_c_slot3: &[TyC],
    input_c_slot_offsets: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply(
            env,
            (
                ExprA::eval(
                    input_a_slot0,
                    input_a_slot1,
                    input_a_slot2,
                    input_a_slot3,
                    input_a_slot_offsets,
                    global,
                ),
                ExprB::eval(
                    input_b_slot0,
                    input_b_slot1,
                    input_b_slot2,
                    input_b_slot3,
                    input_b_slot_offsets,
                    global,
                ),
                ExprC::eval(
                    input_c_slot0,
                    input_c_slot1,
                    input_c_slot2,
                    input_c_slot3,
                    input_c_slot_offsets,
                    global,
                ),
            ),
        );
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

macro_rules! define_tuple_unique_device_expr_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_expr:ident :
            $first_slot0:ident, $first_slot1:ident, $first_slot2:ident, $first_slot3:ident, $first_offsets:ident
        $(, $ty:ident : $expr:ident :
            $slot0:ident, $slot1:ident, $slot2:ident, $slot3:ident, $offsets:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            Pred: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_slot0: &[$first_ty],
            $first_slot1: &[$first_ty],
            $first_slot2: &[$first_ty],
            $first_slot3: &[$first_ty],
            $first_offsets: &[u32],
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if global == 0usize {
                    flags[global] = 1u32;
                } else if Pred::apply(
                    (
                        $first_expr::eval(
                            $first_slot0,
                            $first_slot1,
                            $first_slot2,
                            $first_slot3,
                            $first_offsets,
                            global - 1usize,
                        ),
                        $(
                            $expr::eval(
                                $slot0,
                                $slot1,
                                $slot2,
                                $slot3,
                                $offsets,
                                global - 1usize,
                            ),
                        )*
                    ),
                    (
                        $first_expr::eval(
                            $first_slot0,
                            $first_slot1,
                            $first_slot2,
                            $first_slot3,
                            $first_offsets,
                            global,
                        ),
                        $(
                            $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, global),
                        )*
                    ),
                ) {
                    flags[global] = 0u32;
                } else {
                    flags[global] = 1u32;
                }
            }
        }
    };
}

define_tuple_unique_device_expr_flags_kernel!(
    tuple2_unique_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets)
);
define_tuple_unique_device_expr_flags_kernel!(
    tuple3_unique_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets)
);
define_tuple_unique_device_expr_flags_kernel!(
    tuple7_unique_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets,
     TyD: ExprD:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets,
     TyE: ExprE:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets,
     TyF: ExprF:
        input_f_slot0, input_f_slot1, input_f_slot2, input_f_slot3, input_f_offsets,
     TyG: ExprG:
        input_g_slot0, input_g_slot1, input_g_slot2, input_g_slot3, input_g_offsets)
);

macro_rules! define_tuple_mismatch_device_expr_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_left_expr:ident / $first_right_expr:ident :
            $first_left_slot0:ident, $first_left_slot1:ident, $first_left_slot2:ident, $first_left_slot3:ident, $first_left_offsets:ident /
            $first_right_slot0:ident, $first_right_slot1:ident, $first_right_slot2:ident, $first_right_slot3:ident, $first_right_offsets:ident
        $(, $ty:ident : $left_expr:ident / $right_expr:ident :
            $left_slot0:ident, $left_slot1:ident, $left_slot2:ident, $left_slot3:ident, $left_offsets:ident /
            $right_slot0:ident, $right_slot1:ident, $right_slot2:ident, $right_slot3:ident, $right_offsets:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_left_expr: DeviceGpuExpr<$first_ty>,
            $first_right_expr: DeviceGpuExpr<$first_ty>,
            $( $left_expr: DeviceGpuExpr<$ty>, $right_expr: DeviceGpuExpr<$ty>, )*
            Eq: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_left_slot0: &[$first_ty],
            $first_left_slot1: &[$first_ty],
            $first_left_slot2: &[$first_ty],
            $first_left_slot3: &[$first_ty],
            $first_left_offsets: &[u32],
            $(
                $left_slot0: &[$ty],
                $left_slot1: &[$ty],
                $left_slot2: &[$ty],
                $left_slot3: &[$ty],
                $left_offsets: &[u32],
            )*
            $first_right_slot0: &[$first_ty],
            $first_right_slot1: &[$first_ty],
            $first_right_slot2: &[$first_ty],
            $first_right_slot3: &[$first_ty],
            $first_right_offsets: &[u32],
            $(
                $right_slot0: &[$ty],
                $right_slot1: &[$ty],
                $right_slot2: &[$ty],
                $right_slot3: &[$ty],
                $right_offsets: &[u32],
            )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if Eq::apply(
                    (
                        $first_left_expr::eval(
                            $first_left_slot0,
                            $first_left_slot1,
                            $first_left_slot2,
                            $first_left_slot3,
                            $first_left_offsets,
                            global,
                        ),
                        $(
                            $left_expr::eval(
                                $left_slot0,
                                $left_slot1,
                                $left_slot2,
                                $left_slot3,
                                $left_offsets,
                                global,
                            ),
                        )*
                    ),
                    (
                        $first_right_expr::eval(
                            $first_right_slot0,
                            $first_right_slot1,
                            $first_right_slot2,
                            $first_right_slot3,
                            $first_right_offsets,
                            global,
                        ),
                        $(
                            $right_expr::eval(
                                $right_slot0,
                                $right_slot1,
                                $right_slot2,
                                $right_slot3,
                                $right_offsets,
                                global,
                            ),
                        )*
                    ),
                ) {
                    flags[global] = 0u32;
                } else {
                    flags[global] = 1u32;
                }
            }
        }
    };
}

define_tuple_mismatch_device_expr_flags_kernel!(
    tuple2_mismatch_device_expr_flags_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets)
);
define_tuple_mismatch_device_expr_flags_kernel!(
    tuple3_mismatch_device_expr_flags_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets)
);

macro_rules! define_tuple_find_first_of_device_expr_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_input_expr:ident / $first_needle_expr:ident :
            $first_input_slot0:ident, $first_input_slot1:ident, $first_input_slot2:ident, $first_input_slot3:ident, $first_input_offsets:ident /
            $first_needle_slot0:ident, $first_needle_slot1:ident, $first_needle_slot2:ident, $first_needle_slot3:ident, $first_needle_offsets:ident
        $(, $ty:ident : $input_expr:ident / $needle_expr:ident :
            $input_slot0:ident, $input_slot1:ident, $input_slot2:ident, $input_slot3:ident, $input_offsets:ident /
            $needle_slot0:ident, $needle_slot1:ident, $needle_slot2:ident, $needle_slot3:ident, $needle_offsets:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_input_expr: DeviceGpuExpr<$first_ty>,
            $first_needle_expr: DeviceGpuExpr<$first_ty>,
            $( $input_expr: DeviceGpuExpr<$ty>, $needle_expr: DeviceGpuExpr<$ty>, )*
            Eq: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_input_slot0: &[$first_ty],
            $first_input_slot1: &[$first_ty],
            $first_input_slot2: &[$first_ty],
            $first_input_slot3: &[$first_ty],
            $first_input_offsets: &[u32],
            $(
                $input_slot0: &[$ty],
                $input_slot1: &[$ty],
                $input_slot2: &[$ty],
                $input_slot3: &[$ty],
                $input_offsets: &[u32],
            )*
            $first_needle_slot0: &[$first_ty],
            $first_needle_slot1: &[$first_ty],
            $first_needle_slot2: &[$first_ty],
            $first_needle_slot3: &[$first_ty],
            $first_needle_offsets: &[u32],
            $(
                $needle_slot0: &[$ty],
                $needle_slot1: &[$ty],
                $needle_slot2: &[$ty],
                $needle_slot3: &[$ty],
                $needle_offsets: &[u32],
            )*
            needle_len: &[u32],
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let needle = RuntimeCell::<usize>::new(0usize);
                let found = RuntimeCell::<u32>::new(0u32);
                while needle.read() < needle_len[0] as usize {
                    if Eq::apply(
                        (
                            $first_input_expr::eval(
                                $first_input_slot0,
                                $first_input_slot1,
                                $first_input_slot2,
                                $first_input_slot3,
                                $first_input_offsets,
                                global,
                            ),
                            $(
                                $input_expr::eval(
                                    $input_slot0,
                                    $input_slot1,
                                    $input_slot2,
                                    $input_slot3,
                                    $input_offsets,
                                    global,
                                ),
                            )*
                        ),
                        (
                            $first_needle_expr::eval(
                                $first_needle_slot0,
                                $first_needle_slot1,
                                $first_needle_slot2,
                                $first_needle_slot3,
                                $first_needle_offsets,
                                needle.read(),
                            ),
                            $(
                                $needle_expr::eval(
                                    $needle_slot0,
                                    $needle_slot1,
                                    $needle_slot2,
                                    $needle_slot3,
                                    $needle_offsets,
                                    needle.read(),
                                ),
                            )*
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
    };
}

define_tuple_find_first_of_device_expr_flags_kernel!(
    tuple2_find_first_of_device_expr_flags_kernel,
    (TyA: InputAExpr / NeedleAExpr:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets /
        needle_a_slot0, needle_a_slot1, needle_a_slot2, needle_a_slot3, needle_a_offsets,
     TyB: InputBExpr / NeedleBExpr:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets /
        needle_b_slot0, needle_b_slot1, needle_b_slot2, needle_b_slot3, needle_b_offsets)
);
define_tuple_find_first_of_device_expr_flags_kernel!(
    tuple3_find_first_of_device_expr_flags_kernel,
    (TyA: InputAExpr / NeedleAExpr:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets /
        needle_a_slot0, needle_a_slot1, needle_a_slot2, needle_a_slot3, needle_a_offsets,
     TyB: InputBExpr / NeedleBExpr:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets /
        needle_b_slot0, needle_b_slot1, needle_b_slot2, needle_b_slot3, needle_b_offsets,
     TyC: InputCExpr / NeedleCExpr:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets /
        needle_c_slot0, needle_c_slot1, needle_c_slot2, needle_c_slot3, needle_c_offsets)
);

macro_rules! define_tuple_search_device_expr_kernels {
    (
        $adjacent_fn:ident,
        $sorted_break_fn:ident,
        $lower_fn:ident,
        $upper_fn:ident,
        ($first_ty:ident : $first_expr:ident :
            $first_slot0:ident, $first_slot1:ident, $first_slot2:ident, $first_slot3:ident, $first_offsets:ident / $first_value:ident
        $(, $ty:ident : $expr:ident :
            $slot0:ident, $slot1:ident, $slot2:ident, $slot3:ident, $offsets:ident / $value:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $adjacent_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            Pred: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_slot0: &[$first_ty],
            $first_slot1: &[$first_ty],
            $first_slot2: &[$first_ty],
            $first_slot3: &[$first_ty],
            $first_offsets: &[u32],
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global + 1usize < flags.len() {
                if Pred::apply(
                    (
                        $first_expr::eval(
                            $first_slot0,
                            $first_slot1,
                            $first_slot2,
                            $first_slot3,
                            $first_offsets,
                            global,
                        ),
                        $(
                            $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, global),
                        )*
                    ),
                    (
                        $first_expr::eval(
                            $first_slot0,
                            $first_slot1,
                            $first_slot2,
                            $first_slot3,
                            $first_offsets,
                            global + 1usize,
                        ),
                        $(
                            $expr::eval(
                                $slot0,
                                $slot1,
                                $slot2,
                                $slot3,
                                $offsets,
                                global + 1usize,
                            ),
                        )*
                    ),
                ) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $sorted_break_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_slot0: &[$first_ty],
            $first_slot1: &[$first_ty],
            $first_slot2: &[$first_ty],
            $first_slot3: &[$first_ty],
            $first_offsets: &[u32],
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if global > 0usize
                    && Less::apply(
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                global,
                            ),
                            $(
                                $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, global),
                            )*
                        ),
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                global - 1usize,
                            ),
                            $(
                                $expr::eval(
                                    $slot0,
                                    $slot1,
                                    $slot2,
                                    $slot3,
                                    $offsets,
                                    global - 1usize,
                                ),
                            )*
                        ),
                    )
                {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $lower_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_slot0: &[$first_ty],
            $first_slot1: &[$first_ty],
            $first_slot2: &[$first_ty],
            $first_slot3: &[$first_ty],
            $first_offsets: &[u32],
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )*
            $first_value: &[$first_ty],
            $( $value: &[$ty], )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                if Less::apply(
                    (
                        $first_expr::eval(
                            $first_slot0,
                            $first_slot1,
                            $first_slot2,
                            $first_slot3,
                            $first_offsets,
                            global,
                        ),
                        $(
                            $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, global),
                        )*
                    ),
                    ($first_value[0], $( $value[0], )*),
                ) {
                    flags[global] = 0u32;
                } else {
                    flags[global] = 1u32;
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $upper_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_slot0: &[$first_ty],
            $first_slot1: &[$first_ty],
            $first_slot2: &[$first_ty],
            $first_slot3: &[$first_ty],
            $first_offsets: &[u32],
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )*
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
                    (
                        $first_expr::eval(
                            $first_slot0,
                            $first_slot1,
                            $first_slot2,
                            $first_slot3,
                            $first_offsets,
                            global,
                        ),
                        $(
                            $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, global),
                        )*
                    ),
                ) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }
    };
}

define_tuple_search_device_expr_kernels!(
    tuple2_adjacent_device_expr_flags_kernel,
    tuple2_sorted_break_device_expr_flags_kernel,
    tuple2_lower_bound_device_expr_flags_kernel,
    tuple2_upper_bound_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets / value_a,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets / value_b)
);
define_tuple_search_device_expr_kernels!(
    tuple3_adjacent_device_expr_flags_kernel,
    tuple3_sorted_break_device_expr_flags_kernel,
    tuple3_lower_bound_device_expr_flags_kernel,
    tuple3_upper_bound_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets / value_a,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets / value_b,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets / value_c)
);

macro_rules! define_tuple_bound_many_device_expr_kernels {
    (
        $lower_fn:ident,
        $upper_fn:ident,
        ($first_ty:ident : $first_source_expr:ident / $first_value_expr:ident :
            $first_source_slot0:ident, $first_source_slot1:ident, $first_source_slot2:ident, $first_source_slot3:ident, $first_source_offsets:ident /
            $first_value_slot0:ident, $first_value_slot1:ident, $first_value_slot2:ident, $first_value_slot3:ident, $first_value_offsets:ident
        $(, $ty:ident : $source_expr:ident / $value_expr:ident :
            $source_slot0:ident, $source_slot1:ident, $source_slot2:ident, $source_slot3:ident, $source_offsets:ident /
            $value_slot0:ident, $value_slot1:ident, $value_slot2:ident, $value_slot3:ident, $value_offsets:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $lower_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_source_expr: DeviceGpuExpr<$first_ty>,
            $first_value_expr: DeviceGpuExpr<$first_ty>,
            $( $source_expr: DeviceGpuExpr<$ty>, )*
            $( $value_expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_source_slot0: &[$first_ty],
            $first_source_slot1: &[$first_ty],
            $first_source_slot2: &[$first_ty],
            $first_source_slot3: &[$first_ty],
            $first_source_offsets: &[u32],
            $first_value_slot0: &[$first_ty],
            $first_value_slot1: &[$first_ty],
            $first_value_slot2: &[$first_ty],
            $first_value_slot3: &[$first_ty],
            $first_value_offsets: &[u32],
            $(
                $source_slot0: &[$ty],
                $source_slot1: &[$ty],
                $source_slot2: &[$ty],
                $source_slot3: &[$ty],
                $source_offsets: &[u32],
                $value_slot0: &[$ty],
                $value_slot1: &[$ty],
                $value_slot2: &[$ty],
                $value_slot3: &[$ty],
                $value_offsets: &[u32],
            )*
            source_len: &[u32],
            value_len: &[u32],
            output: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (value_len[0] as usize) {
                let value = (
                    $first_value_expr::eval(
                        $first_value_slot0,
                        $first_value_slot1,
                        $first_value_slot2,
                        $first_value_slot3,
                        $first_value_offsets,
                        global,
                    ),
                    $(
                        $value_expr::eval(
                            $value_slot0,
                            $value_slot1,
                            $value_slot2,
                            $value_slot3,
                            $value_offsets,
                            global,
                        ),
                    )*
                );
                let mut first = 0usize;
                let mut count = source_len[0] as usize;
                while count > 0usize {
                    let step = count / 2usize;
                    let mid = first + step;
                    let candidate = (
                        $first_source_expr::eval(
                            $first_source_slot0,
                            $first_source_slot1,
                            $first_source_slot2,
                            $first_source_slot3,
                            $first_source_offsets,
                            mid,
                        ),
                        $(
                            $source_expr::eval(
                                $source_slot0,
                                $source_slot1,
                                $source_slot2,
                                $source_slot3,
                                $source_offsets,
                                mid,
                            ),
                        )*
                    );
                    if Less::apply(candidate, value) {
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
        pub(crate) fn $upper_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_source_expr: DeviceGpuExpr<$first_ty>,
            $first_value_expr: DeviceGpuExpr<$first_ty>,
            $( $source_expr: DeviceGpuExpr<$ty>, )*
            $( $value_expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_source_slot0: &[$first_ty],
            $first_source_slot1: &[$first_ty],
            $first_source_slot2: &[$first_ty],
            $first_source_slot3: &[$first_ty],
            $first_source_offsets: &[u32],
            $first_value_slot0: &[$first_ty],
            $first_value_slot1: &[$first_ty],
            $first_value_slot2: &[$first_ty],
            $first_value_slot3: &[$first_ty],
            $first_value_offsets: &[u32],
            $(
                $source_slot0: &[$ty],
                $source_slot1: &[$ty],
                $source_slot2: &[$ty],
                $source_slot3: &[$ty],
                $source_offsets: &[u32],
                $value_slot0: &[$ty],
                $value_slot1: &[$ty],
                $value_slot2: &[$ty],
                $value_slot3: &[$ty],
                $value_offsets: &[u32],
            )*
            source_len: &[u32],
            value_len: &[u32],
            output: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (value_len[0] as usize) {
                let value = (
                    $first_value_expr::eval(
                        $first_value_slot0,
                        $first_value_slot1,
                        $first_value_slot2,
                        $first_value_slot3,
                        $first_value_offsets,
                        global,
                    ),
                    $(
                        $value_expr::eval(
                            $value_slot0,
                            $value_slot1,
                            $value_slot2,
                            $value_slot3,
                            $value_offsets,
                            global,
                        ),
                    )*
                );
                let mut first = 0usize;
                let mut count = source_len[0] as usize;
                while count > 0usize {
                    let step = count / 2usize;
                    let mid = first + step;
                    let candidate = (
                        $first_source_expr::eval(
                            $first_source_slot0,
                            $first_source_slot1,
                            $first_source_slot2,
                            $first_source_slot3,
                            $first_source_offsets,
                            mid,
                        ),
                        $(
                            $source_expr::eval(
                                $source_slot0,
                                $source_slot1,
                                $source_slot2,
                                $source_slot3,
                                $source_offsets,
                                mid,
                            ),
                        )*
                    );
                    if !Less::apply(value, candidate) {
                        first = mid + 1usize;
                        count = count - step - 1usize;
                    } else {
                        count = step;
                    }
                }
                output[global] = first as u32;
            }
        }
    };
}

define_tuple_bound_many_device_expr_kernels!(
    tuple2_lower_bound_device_expr_many_kernel,
    tuple2_upper_bound_device_expr_many_kernel,
    (TyA: SourceExprA / ValueExprA:
        source_a_slot0, source_a_slot1, source_a_slot2, source_a_slot3, source_a_offsets /
        value_a_slot0, value_a_slot1, value_a_slot2, value_a_slot3, value_a_offsets,
     TyB: SourceExprB / ValueExprB:
        source_b_slot0, source_b_slot1, source_b_slot2, source_b_slot3, source_b_offsets /
        value_b_slot0, value_b_slot1, value_b_slot2, value_b_slot3, value_b_offsets)
);
define_tuple_bound_many_device_expr_kernels!(
    tuple3_lower_bound_device_expr_many_kernel,
    tuple3_upper_bound_device_expr_many_kernel,
    (TyA: SourceExprA / ValueExprA:
        source_a_slot0, source_a_slot1, source_a_slot2, source_a_slot3, source_a_offsets /
        value_a_slot0, value_a_slot1, value_a_slot2, value_a_slot3, value_a_offsets,
     TyB: SourceExprB / ValueExprB:
        source_b_slot0, source_b_slot1, source_b_slot2, source_b_slot3, source_b_offsets /
        value_b_slot0, value_b_slot1, value_b_slot2, value_b_slot3, value_b_offsets,
     TyC: SourceExprC / ValueExprC:
        source_c_slot0, source_c_slot1, source_c_slot2, source_c_slot3, source_c_offsets /
        value_c_slot0, value_c_slot1, value_c_slot2, value_c_slot3, value_c_offsets)
);

macro_rules! define_tuple_membership_device_expr_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_expr:ident :
            $first_candidate_slot0:ident, $first_candidate_slot1:ident, $first_candidate_slot2:ident, $first_candidate_slot3:ident, $first_candidate_offsets:ident /
            $first_sorted_expr:ident :
            $first_sorted_slot0:ident, $first_sorted_slot1:ident, $first_sorted_slot2:ident, $first_sorted_slot3:ident, $first_sorted_offsets:ident
        $(, $ty:ident : $expr:ident :
            $candidate_slot0:ident, $candidate_slot1:ident, $candidate_slot2:ident, $candidate_slot3:ident, $candidate_offsets:ident /
            $sorted_expr:ident :
            $sorted_slot0:ident, $sorted_slot1:ident, $sorted_slot2:ident, $sorted_slot3:ident, $sorted_offsets:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            $first_sorted_expr: DeviceGpuExpr<$first_ty>,
            $( $sorted_expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_candidate_slot0: &[$first_ty],
            $first_candidate_slot1: &[$first_ty],
            $first_candidate_slot2: &[$first_ty],
            $first_candidate_slot3: &[$first_ty],
            $first_candidate_offsets: &[u32],
            $(
                $candidate_slot0: &[$ty],
                $candidate_slot1: &[$ty],
                $candidate_slot2: &[$ty],
                $candidate_slot3: &[$ty],
                $candidate_offsets: &[u32],
            )*
            candidate_len: &[u32],
            $first_sorted_slot0: &[$first_ty],
            $first_sorted_slot1: &[$first_ty],
            $first_sorted_slot2: &[$first_ty],
            $first_sorted_slot3: &[$first_ty],
            $first_sorted_offsets: &[u32],
            $(
                $sorted_slot0: &[$ty],
                $sorted_slot1: &[$ty],
                $sorted_slot2: &[$ty],
                $sorted_slot3: &[$ty],
                $sorted_offsets: &[u32],
            )*
            sorted_len: &[u32],
            keep_present: &[u32],
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let candidate_value = (
                    $first_expr::eval(
                        $first_candidate_slot0,
                        $first_candidate_slot1,
                        $first_candidate_slot2,
                        $first_candidate_slot3,
                        $first_candidate_offsets,
                        global,
                    ),
                    $(
                        $expr::eval(
                            $candidate_slot0,
                            $candidate_slot1,
                            $candidate_slot2,
                            $candidate_slot3,
                            $candidate_offsets,
                            global,
                        ),
                    )*
                );

                let candidate_first = RuntimeCell::<usize>::new(0usize);
                let candidate_count = RuntimeCell::<usize>::new(candidate_len[0] as usize);
                while candidate_count.read() > 0usize {
                    let step = candidate_count.read() / 2usize;
                    let mid = candidate_first.read() + step;
                    let mid_value = (
                        $first_expr::eval(
                            $first_candidate_slot0,
                            $first_candidate_slot1,
                            $first_candidate_slot2,
                            $first_candidate_slot3,
                            $first_candidate_offsets,
                            mid,
                        ),
                        $(
                            $expr::eval(
                                $candidate_slot0,
                                $candidate_slot1,
                                $candidate_slot2,
                                $candidate_slot3,
                                $candidate_offsets,
                                mid,
                            ),
                        )*
                    );
                    if Less::apply(mid_value, candidate_value) {
                        candidate_first.store(mid + 1usize);
                        candidate_count.store(candidate_count.read() - step - 1usize);
                    } else {
                        candidate_count.store(step);
                    }
                }

                let sorted_first = RuntimeCell::<usize>::new(0usize);
                let sorted_count = RuntimeCell::<usize>::new(sorted_len[0] as usize);
                while sorted_count.read() > 0usize {
                    let step = sorted_count.read() / 2usize;
                    let mid = sorted_first.read() + step;
                    let sorted_value = (
                        $first_sorted_expr::eval(
                            $first_sorted_slot0,
                            $first_sorted_slot1,
                            $first_sorted_slot2,
                            $first_sorted_slot3,
                            $first_sorted_offsets,
                            mid,
                        ),
                        $(
                            $sorted_expr::eval(
                                $sorted_slot0,
                                $sorted_slot1,
                                $sorted_slot2,
                                $sorted_slot3,
                                $sorted_offsets,
                                mid,
                            ),
                        )*
                    );
                    if Less::apply(sorted_value, candidate_value) {
                        sorted_first.store(mid + 1usize);
                        sorted_count.store(sorted_count.read() - step - 1usize);
                    } else {
                        sorted_count.store(step);
                    }
                }

                let sorted_after = RuntimeCell::<usize>::new(0usize);
                let sorted_after_count = RuntimeCell::<usize>::new(sorted_len[0] as usize);
                while sorted_after_count.read() > 0usize {
                    let step = sorted_after_count.read() / 2usize;
                    let mid = sorted_after.read() + step;
                    let sorted_value = (
                        $first_sorted_expr::eval(
                            $first_sorted_slot0,
                            $first_sorted_slot1,
                            $first_sorted_slot2,
                            $first_sorted_slot3,
                            $first_sorted_offsets,
                            mid,
                        ),
                        $(
                            $sorted_expr::eval(
                                $sorted_slot0,
                                $sorted_slot1,
                                $sorted_slot2,
                                $sorted_slot3,
                                $sorted_offsets,
                                mid,
                            ),
                        )*
                    );
                    if !Less::apply(candidate_value, sorted_value) {
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

define_tuple_membership_device_expr_flags_kernel!(
    tuple2_membership_device_expr_flags_kernel,
    (TyA: ExprA:
        candidate_a_slot0, candidate_a_slot1, candidate_a_slot2, candidate_a_slot3, candidate_a_offsets /
        SortedExprA:
        sorted_a_slot0, sorted_a_slot1, sorted_a_slot2, sorted_a_slot3, sorted_a_offsets,
     TyB: ExprB:
        candidate_b_slot0, candidate_b_slot1, candidate_b_slot2, candidate_b_slot3, candidate_b_offsets /
        SortedExprB:
        sorted_b_slot0, sorted_b_slot1, sorted_b_slot2, sorted_b_slot3, sorted_b_offsets)
);
define_tuple_membership_device_expr_flags_kernel!(
    tuple3_membership_device_expr_flags_kernel,
    (TyA: ExprA:
        candidate_a_slot0, candidate_a_slot1, candidate_a_slot2, candidate_a_slot3, candidate_a_offsets /
        SortedExprA:
        sorted_a_slot0, sorted_a_slot1, sorted_a_slot2, sorted_a_slot3, sorted_a_offsets,
     TyB: ExprB:
        candidate_b_slot0, candidate_b_slot1, candidate_b_slot2, candidate_b_slot3, candidate_b_offsets /
        SortedExprB:
        sorted_b_slot0, sorted_b_slot1, sorted_b_slot2, sorted_b_slot3, sorted_b_offsets,
     TyC: ExprC:
        candidate_c_slot0, candidate_c_slot1, candidate_c_slot2, candidate_c_slot3, candidate_c_offsets /
        SortedExprC:
        sorted_c_slot0, sorted_c_slot1, sorted_c_slot2, sorted_c_slot3, sorted_c_offsets)
);

macro_rules! define_tuple_minmax_device_expr_kernels {
    (
        $element_fn:ident,
        $index_fn:ident,
        ($first_ty:ident : $first_expr:ident :
            $first_slot0:ident, $first_slot1:ident, $first_slot2:ident, $first_slot3:ident, $first_offsets:ident
        $(, $ty:ident : $expr:ident :
            $slot0:ident, $slot1:ident, $slot2:ident, $slot3:ident, $offsets:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $element_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_slot0: &[$first_ty],
            $first_slot1: &[$first_ty],
            $first_slot2: &[$first_ty],
            $first_slot3: &[$first_ty],
            $first_offsets: &[u32],
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )*
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
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                i.read(),
                            ),
                            $(
                                $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, i.read()),
                            )*
                        ),
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                min_index.read(),
                            ),
                            $(
                                $expr::eval(
                                    $slot0,
                                    $slot1,
                                    $slot2,
                                    $slot3,
                                    $offsets,
                                    min_index.read(),
                                ),
                            )*
                        ),
                    ) {
                        min_index.store(i.read());
                    }
                    if Less::apply(
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                max_index.read(),
                            ),
                            $(
                                $expr::eval(
                                    $slot0,
                                    $slot1,
                                    $slot2,
                                    $slot3,
                                    $offsets,
                                    max_index.read(),
                                ),
                            )*
                        ),
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                i.read(),
                            ),
                            $(
                                $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, i.read()),
                            )*
                        ),
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
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    other_min,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, other_min),
                                )*
                            ),
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    current_min,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, current_min),
                                )*
                            ),
                        ) {
                            min_indices[unit] = other_min as u32;
                        }

                        let other_max = max_indices[unit + stride.read()] as usize;
                        let current_max = max_indices[unit] as usize;
                        if Less::apply(
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    current_max,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, current_max),
                                )*
                            ),
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    other_max,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, other_max),
                                )*
                            ),
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
        pub(crate) fn $index_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_slot0: &[$first_ty],
            $first_slot1: &[$first_ty],
            $first_slot2: &[$first_ty],
            $first_slot3: &[$first_ty],
            $first_offsets: &[u32],
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )*
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
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                candidate_min,
                            ),
                            $(
                                $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, candidate_min),
                            )*
                        ),
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                min_index.read(),
                            ),
                            $(
                                $expr::eval(
                                    $slot0,
                                    $slot1,
                                    $slot2,
                                    $slot3,
                                    $offsets,
                                    min_index.read(),
                                ),
                            )*
                        ),
                    ) {
                        min_index.store(candidate_min);
                    }
                    if Less::apply(
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                max_index.read(),
                            ),
                            $(
                                $expr::eval(
                                    $slot0,
                                    $slot1,
                                    $slot2,
                                    $slot3,
                                    $offsets,
                                    max_index.read(),
                                ),
                            )*
                        ),
                        (
                            $first_expr::eval(
                                $first_slot0,
                                $first_slot1,
                                $first_slot2,
                                $first_slot3,
                                $first_offsets,
                                candidate_max,
                            ),
                            $(
                                $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, candidate_max),
                            )*
                        ),
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
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    other_min,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, other_min),
                                )*
                            ),
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    current_min,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, current_min),
                                )*
                            ),
                        ) {
                            min_indices[unit] = other_min as u32;
                        }

                        let other_max = max_indices[unit + stride.read()] as usize;
                        let current_max = max_indices[unit] as usize;
                        if Less::apply(
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    current_max,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, current_max),
                                )*
                            ),
                            (
                                $first_expr::eval(
                                    $first_slot0,
                                    $first_slot1,
                                    $first_slot2,
                                    $first_slot3,
                                    $first_offsets,
                                    other_max,
                                ),
                                $(
                                    $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, other_max),
                                )*
                            ),
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
    };
}

define_tuple_minmax_device_expr_kernels!(
    tuple2_minmax_element_device_expr_partials_kernel,
    tuple2_minmax_index_device_expr_partials_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets)
);
define_tuple_minmax_device_expr_kernels!(
    tuple3_minmax_element_device_expr_partials_kernel,
    tuple3_minmax_index_device_expr_partials_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets)
);

macro_rules! define_tuple_lexicographical_device_expr_kernels {
    (
        $diff_fn:ident,
        $compare_fn:ident,
        ($first_ty:ident : $first_left_expr:ident / $first_right_expr:ident :
            $first_left_slot0:ident, $first_left_slot1:ident, $first_left_slot2:ident, $first_left_slot3:ident, $first_left_offsets:ident /
            $first_right_slot0:ident, $first_right_slot1:ident, $first_right_slot2:ident, $first_right_slot3:ident, $first_right_offsets:ident
        $(, $ty:ident : $left_expr:ident / $right_expr:ident :
            $left_slot0:ident, $left_slot1:ident, $left_slot2:ident, $left_slot3:ident, $left_offsets:ident /
            $right_slot0:ident, $right_slot1:ident, $right_slot2:ident, $right_slot3:ident, $right_offsets:ident
        )*)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $diff_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_left_expr: DeviceGpuExpr<$first_ty>,
            $first_right_expr: DeviceGpuExpr<$first_ty>,
            $( $left_expr: DeviceGpuExpr<$ty>, $right_expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_left_slot0: &[$first_ty],
            $first_left_slot1: &[$first_ty],
            $first_left_slot2: &[$first_ty],
            $first_left_slot3: &[$first_ty],
            $first_left_offsets: &[u32],
            $(
                $left_slot0: &[$ty],
                $left_slot1: &[$ty],
                $left_slot2: &[$ty],
                $left_slot3: &[$ty],
                $left_offsets: &[u32],
            )*
            $first_right_slot0: &[$first_ty],
            $first_right_slot1: &[$first_ty],
            $first_right_slot2: &[$first_ty],
            $first_right_slot3: &[$first_ty],
            $first_right_offsets: &[u32],
            $(
                $right_slot0: &[$ty],
                $right_slot1: &[$ty],
                $right_slot2: &[$ty],
                $right_slot3: &[$ty],
                $right_offsets: &[u32],
            )*
            flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let lhs = (
                    $first_left_expr::eval(
                        $first_left_slot0,
                        $first_left_slot1,
                        $first_left_slot2,
                        $first_left_slot3,
                        $first_left_offsets,
                        global,
                    ),
                    $(
                        $left_expr::eval(
                            $left_slot0,
                            $left_slot1,
                            $left_slot2,
                            $left_slot3,
                            $left_offsets,
                            global,
                        ),
                    )*
                );
                let rhs = (
                    $first_right_expr::eval(
                        $first_right_slot0,
                        $first_right_slot1,
                        $first_right_slot2,
                        $first_right_slot3,
                        $first_right_offsets,
                        global,
                    ),
                    $(
                        $right_expr::eval(
                            $right_slot0,
                            $right_slot1,
                            $right_slot2,
                            $right_slot3,
                            $right_offsets,
                            global,
                        ),
                    )*
                );
                if Less::apply(lhs, rhs) || Less::apply(rhs, lhs) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $compare_fn<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_left_expr: DeviceGpuExpr<$first_ty>,
            $first_right_expr: DeviceGpuExpr<$first_ty>,
            $( $left_expr: DeviceGpuExpr<$ty>, $right_expr: DeviceGpuExpr<$ty>, )*
            Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>,
        >(
            $first_left_slot0: &[$first_ty],
            $first_left_slot1: &[$first_ty],
            $first_left_slot2: &[$first_ty],
            $first_left_slot3: &[$first_ty],
            $first_left_offsets: &[u32],
            $(
                $left_slot0: &[$ty],
                $left_slot1: &[$ty],
                $left_slot2: &[$ty],
                $left_slot3: &[$ty],
                $left_offsets: &[u32],
            )*
            $first_right_slot0: &[$first_ty],
            $first_right_slot1: &[$first_ty],
            $first_right_slot2: &[$first_ty],
            $first_right_slot3: &[$first_ty],
            $first_right_offsets: &[u32],
            $(
                $right_slot0: &[$ty],
                $right_slot1: &[$ty],
                $right_slot2: &[$ty],
                $right_slot3: &[$ty],
                $right_offsets: &[u32],
            )*
            index: &[u32],
            output: &mut [u32],
        ) {
            if UNIT_POS == 0 {
                let i = index[0] as usize;
                if Less::apply(
                    (
                        $first_left_expr::eval(
                            $first_left_slot0,
                            $first_left_slot1,
                            $first_left_slot2,
                            $first_left_slot3,
                            $first_left_offsets,
                            i,
                        ),
                        $(
                            $left_expr::eval(
                                $left_slot0,
                                $left_slot1,
                                $left_slot2,
                                $left_slot3,
                                $left_offsets,
                                i,
                            ),
                        )*
                    ),
                    (
                        $first_right_expr::eval(
                            $first_right_slot0,
                            $first_right_slot1,
                            $first_right_slot2,
                            $first_right_slot3,
                            $first_right_offsets,
                            i,
                        ),
                        $(
                            $right_expr::eval(
                                $right_slot0,
                                $right_slot1,
                                $right_slot2,
                                $right_slot3,
                                $right_offsets,
                                i,
                            ),
                        )*
                    ),
                ) {
                    output[0] = 1u32;
                } else {
                    output[0] = 0u32;
                }
            }
        }
    };
}

define_tuple_lexicographical_device_expr_kernels!(
    tuple2_lexicographical_diff_device_expr_flags_kernel,
    tuple2_lexicographical_compare_at_device_expr_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets)
);
define_tuple_lexicographical_device_expr_kernels!(
    tuple3_lexicographical_diff_device_expr_flags_kernel,
    tuple3_lexicographical_compare_at_device_expr_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets)
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_expr_flag_only_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp<T>,
>(
    env: Pred::Env,
    input: &[T],
    indices: &[u32],
    rhs: &[T],
    rhs_indices: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply(env, Expr::eval(input, indices, rhs, rhs_indices, global));
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_device_expr_flag_only_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Pred: PredicateOp<T>,
>(
    env: Pred::Env,
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply(
            env,
            Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global),
        );
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_stencil_expr_flags_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    StencilExpr: GpuExpr<S>,
    Pred: PredicateOp<S>,
>(
    env: Pred::Env,
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
        let stencil = StencilExpr::eval(
            stencil_input,
            stencil_indices,
            stencil_rhs,
            stencil_rhs_indices,
            global,
        );
        values[global] = value;
        let selected = Pred::apply(env, stencil);
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
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
    Pred: PredicateOp<S>,
>(
    op_env: Op::Env,
    pred_env: Pred::Env,
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
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let stencil = StencilExpr::eval(
            stencil_input,
            stencil_indices,
            stencil_rhs,
            stencil_rhs_indices,
            global,
        );
        if Pred::apply(pred_env, stencil) {
            let value = ValueExpr::eval(
                value_input,
                value_indices,
                value_rhs,
                value_rhs_indices,
                global,
            );
            output[global] = Op::apply(op_env, value);
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
pub(crate) fn gather_device_expr_into_kernel<
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
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let index = IndexExpr::eval(
            index_input,
            index_indices,
            index_rhs,
            index_rhs_indices,
            global,
        );
        output[output_offset[0] as usize + global] = ValueExpr::eval(
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
pub(crate) fn scatter_expr_into_kernel<
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
    output_offset: &[u32],
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
        output[output_offset[0] as usize + index as usize] = value;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scatter_if_expr_kernel<
    T: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
    Pred: PredicateOp<T>,
>(
    env: Pred::Env,
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
        if Pred::apply(env, value) {
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
pub(crate) fn scalar_inclusive_scan_block_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
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
pub(crate) fn scalar_scan_add_block_prefix_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
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
pub(crate) fn scalar_reduce_last_finalize_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
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
    Op: BinaryOp<(A,)>,
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
pub(crate) fn tuple1_inclusive_scan_block_kernel<A: CubePrimitive, Op: BinaryOp<(A,)>>(
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
pub(crate) fn tuple1_scan_add_block_prefix_kernel<A: CubePrimitive, Op: BinaryOp<(A,)>>(
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
pub(crate) fn tuple1_scan_make_exclusive_kernel<A: CubePrimitive, Op: BinaryOp<(A,)>>(
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B, C)>,
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
    Op: BinaryOp<(A, B, C)>,
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
    Op: BinaryOp<(A, B, C)>,
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
    Op: BinaryOp<(A, B, C)>,
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
pub(crate) fn tuple7_device_inclusive_scan_expr_block_kernel<
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
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
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
    d_slot0: &[D],
    d_slot1: &[D],
    d_slot2: &[D],
    d_slot3: &[D],
    d_offsets: &[u32],
    e_slot0: &[E],
    e_slot1: &[E],
    e_slot2: &[E],
    e_slot3: &[E],
    e_offsets: &[u32],
    f_slot0: &[F],
    f_slot1: &[F],
    f_slot2: &[F],
    f_slot3: &[F],
    f_offsets: &[u32],
    g_slot0: &[G],
    g_slot1: &[G],
    g_slot2: &[G],
    g_slot3: &[G],
    g_offsets: &[u32],
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_d: &mut [D],
    output_e: &mut [E],
    output_f: &mut [F],
    output_g: &mut [G],
    block_sums_a: &mut [A],
    block_sums_b: &mut [B],
    block_sums_c: &mut [C],
    block_sums_d: &mut [D],
    block_sums_e: &mut [E],
    block_sums_f: &mut [F],
    block_sums_g: &mut [G],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut values_d = Shared::<[D]>::new_slice(cube_dim);
    let mut values_e = Shared::<[E]>::new_slice(cube_dim);
    let mut values_f = Shared::<[F]>::new_slice(cube_dim);
    let mut values_g = Shared::<[G]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = ExprA::eval(a_slot0, a_slot1, a_slot2, a_slot3, a_offsets, global);
        values_b[unit] = ExprB::eval(b_slot0, b_slot1, b_slot2, b_slot3, b_offsets, global);
        values_c[unit] = ExprC::eval(c_slot0, c_slot1, c_slot2, c_slot3, c_offsets, global);
        values_d[unit] = ExprD::eval(d_slot0, d_slot1, d_slot2, d_slot3, d_offsets, global);
        values_e[unit] = ExprE::eval(e_slot0, e_slot1, e_slot2, e_slot3, e_offsets, global);
        values_f[unit] = ExprF::eval(f_slot0, f_slot1, f_slot2, f_slot3, f_offsets, global);
        values_g[unit] = ExprG::eval(g_slot0, g_slot1, g_slot2, g_slot3, g_offsets, global);
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
        let addend_d = RuntimeCell::<D>::new(values_d[unit]);
        let addend_e = RuntimeCell::<E>::new(values_e[unit]);
        let addend_f = RuntimeCell::<F>::new(values_f[unit]);
        let addend_g = RuntimeCell::<G>::new(values_g[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_c.store(values_c[unit - stride.read()]);
            addend_d.store(values_d[unit - stride.read()]);
            addend_e.store(values_e[unit - stride.read()]);
            addend_f.store(values_f[unit - stride.read()]);
            addend_g.store(values_g[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply(
                    (
                        addend_a.read(),
                        addend_b.read(),
                        addend_c.read(),
                        addend_d.read(),
                        addend_e.read(),
                        addend_f.read(),
                        addend_g.read(),
                    ),
                    (
                        values_a[unit],
                        values_b[unit],
                        values_c[unit],
                        values_d[unit],
                        values_e[unit],
                        values_f[unit],
                        values_g[unit],
                    ),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
                values_c[unit] = value.2;
                values_d[unit] = value.3;
                values_e[unit] = value.4;
                values_f[unit] = value.5;
                values_g[unit] = value.6;
            } else {
                values_a[unit] = addend_a.read();
                values_b[unit] = addend_b.read();
                values_c[unit] = addend_c.read();
                values_d[unit] = addend_d.read();
                values_e[unit] = addend_e.read();
                values_f[unit] = addend_f.read();
                values_g[unit] = addend_g.read();
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
        output_d[global] = values_d[unit];
        output_e[global] = values_e[unit];
        output_f[global] = values_f[unit];
        output_g[global] = values_g[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
            block_sums_b[block] = values_b[unit];
            block_sums_c[block] = values_c[unit];
            block_sums_d[block] = values_d[unit];
            block_sums_e[block] = values_e[unit];
            block_sums_f[block] = values_f[unit];
            block_sums_g[block] = values_g[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_inclusive_scan_block_kernel<
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
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_d: &mut [D],
    output_e: &mut [E],
    output_f: &mut [F],
    output_g: &mut [G],
    block_sums_a: &mut [A],
    block_sums_b: &mut [B],
    block_sums_c: &mut [C],
    block_sums_d: &mut [D],
    block_sums_e: &mut [E],
    block_sums_f: &mut [F],
    block_sums_g: &mut [G],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut values_d = Shared::<[D]>::new_slice(cube_dim);
    let mut values_e = Shared::<[E]>::new_slice(cube_dim);
    let mut values_f = Shared::<[F]>::new_slice(cube_dim);
    let mut values_g = Shared::<[G]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = input_a[global];
        values_b[unit] = input_b[global];
        values_c[unit] = input_c[global];
        values_d[unit] = input_d[global];
        values_e[unit] = input_e[global];
        values_f[unit] = input_f[global];
        values_g[unit] = input_g[global];
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
        let addend_d = RuntimeCell::<D>::new(values_d[unit]);
        let addend_e = RuntimeCell::<E>::new(values_e[unit]);
        let addend_f = RuntimeCell::<F>::new(values_f[unit]);
        let addend_g = RuntimeCell::<G>::new(values_g[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_c.store(values_c[unit - stride.read()]);
            addend_d.store(values_d[unit - stride.read()]);
            addend_e.store(values_e[unit - stride.read()]);
            addend_f.store(values_f[unit - stride.read()]);
            addend_g.store(values_g[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply(
                    (
                        addend_a.read(),
                        addend_b.read(),
                        addend_c.read(),
                        addend_d.read(),
                        addend_e.read(),
                        addend_f.read(),
                        addend_g.read(),
                    ),
                    (
                        values_a[unit],
                        values_b[unit],
                        values_c[unit],
                        values_d[unit],
                        values_e[unit],
                        values_f[unit],
                        values_g[unit],
                    ),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
                values_c[unit] = value.2;
                values_d[unit] = value.3;
                values_e[unit] = value.4;
                values_f[unit] = value.5;
                values_g[unit] = value.6;
            } else {
                values_a[unit] = addend_a.read();
                values_b[unit] = addend_b.read();
                values_c[unit] = addend_c.read();
                values_d[unit] = addend_d.read();
                values_e[unit] = addend_e.read();
                values_f[unit] = addend_f.read();
                values_g[unit] = addend_g.read();
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
        output_d[global] = values_d[unit];
        output_e[global] = values_e[unit];
        output_f[global] = values_f[unit];
        output_g[global] = values_g[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
            block_sums_b[block] = values_b[unit];
            block_sums_c[block] = values_c[unit];
            block_sums_d[block] = values_d[unit];
            block_sums_e[block] = values_e[unit];
            block_sums_f[block] = values_f[unit];
            block_sums_g[block] = values_g[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_inclusive_scan_block_kernel<
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
    len: &[u32],
    output_a: &mut [A],
    output_b: &mut [B],
    output_c: &mut [C],
    output_d: &mut [D],
    output_e: &mut [E],
    output_f: &mut [F],
    output_g: &mut [G],
    block_sums_a: &mut [A],
    block_sums_b: &mut [B],
    block_sums_c: &mut [C],
    block_sums_d: &mut [D],
    block_sums_e: &mut [E],
    block_sums_f: &mut [F],
    block_sums_g: &mut [G],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[A]>::new_slice(cube_dim);
    let mut values_b = Shared::<[B]>::new_slice(cube_dim);
    let mut values_c = Shared::<[C]>::new_slice(cube_dim);
    let mut values_d = Shared::<[D]>::new_slice(cube_dim);
    let mut values_e = Shared::<[E]>::new_slice(cube_dim);
    let mut values_f = Shared::<[F]>::new_slice(cube_dim);
    let mut values_g = Shared::<[G]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values_a[unit] = input_a[offsets[0] as usize + global];
        values_b[unit] = input_b[offsets[1] as usize + global];
        values_c[unit] = input_c[offsets[2] as usize + global];
        values_d[unit] = input_d[offsets[3] as usize + global];
        values_e[unit] = input_e[offsets[4] as usize + global];
        values_f[unit] = input_f[offsets[5] as usize + global];
        values_g[unit] = input_g[offsets[6] as usize + global];
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
        let addend_d = RuntimeCell::<D>::new(values_d[unit]);
        let addend_e = RuntimeCell::<E>::new(values_e[unit]);
        let addend_f = RuntimeCell::<F>::new(values_f[unit]);
        let addend_g = RuntimeCell::<G>::new(values_g[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0u32 {
            addend_a.store(values_a[unit - stride.read()]);
            addend_b.store(values_b[unit - stride.read()]);
            addend_c.store(values_c[unit - stride.read()]);
            addend_d.store(values_d[unit - stride.read()]);
            addend_e.store(values_e[unit - stride.read()]);
            addend_f.store(values_f[unit - stride.read()]);
            addend_g.store(values_g[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0u32 {
            if valid[unit] != 0u32 {
                let value = Op::apply(
                    (
                        addend_a.read(),
                        addend_b.read(),
                        addend_c.read(),
                        addend_d.read(),
                        addend_e.read(),
                        addend_f.read(),
                        addend_g.read(),
                    ),
                    (
                        values_a[unit],
                        values_b[unit],
                        values_c[unit],
                        values_d[unit],
                        values_e[unit],
                        values_f[unit],
                        values_g[unit],
                    ),
                );
                values_a[unit] = value.0;
                values_b[unit] = value.1;
                values_c[unit] = value.2;
                values_d[unit] = value.3;
                values_e[unit] = value.4;
                values_f[unit] = value.5;
                values_g[unit] = value.6;
            } else {
                values_a[unit] = addend_a.read();
                values_b[unit] = addend_b.read();
                values_c[unit] = addend_c.read();
                values_d[unit] = addend_d.read();
                values_e[unit] = addend_e.read();
                values_f[unit] = addend_f.read();
                values_g[unit] = addend_g.read();
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
        output_d[global] = values_d[unit];
        output_e[global] = values_e[unit];
        output_f[global] = values_f[unit];
        output_g[global] = values_g[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_sums_a[block] = values_a[unit];
            block_sums_b[block] = values_b[unit];
            block_sums_c[block] = values_c[unit];
            block_sums_d[block] = values_d[unit];
            block_sums_e[block] = values_e[unit];
            block_sums_f[block] = values_f[unit];
            block_sums_g[block] = values_g[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_scan_add_block_prefix_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    Op: BinaryOp<(A, B, C, D, E, F, G)>,
>(
    block_prefixes_a: &[A],
    block_prefixes_b: &[B],
    block_prefixes_c: &[C],
    block_prefixes_d: &[D],
    block_prefixes_e: &[E],
    block_prefixes_f: &[F],
    block_prefixes_g: &[G],
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
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;

    if block > 0usize && global < (len[0] as usize) {
        let value = Op::apply(
            (
                block_prefixes_a[block - 1usize],
                block_prefixes_b[block - 1usize],
                block_prefixes_c[block - 1usize],
                block_prefixes_d[block - 1usize],
                block_prefixes_e[block - 1usize],
                block_prefixes_f[block - 1usize],
                block_prefixes_g[block - 1usize],
            ),
            (
                output_a[global],
                output_b[global],
                output_c[global],
                output_d[global],
                output_e[global],
                output_f[global],
                output_g[global],
            ),
        );
        output_a[global] = value.0;
        output_b[global] = value.1;
        output_c[global] = value.2;
        output_d[global] = value.3;
        output_e[global] = value.4;
        output_f[global] = value.5;
        output_g[global] = value.6;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_scan_make_exclusive_kernel<
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
    if global < output_a.len() {
        if global == 0usize {
            output_a[global] = init_a[0];
            output_b[global] = init_b[0];
            output_c[global] = init_c[0];
            output_d[global] = init_d[0];
            output_e[global] = init_e[0];
            output_f[global] = init_f[0];
            output_g[global] = init_g[0];
        } else {
            let value = Op::apply(
                (
                    init_a[0], init_b[0], init_c[0], init_d[0], init_e[0], init_f[0], init_g[0],
                ),
                (
                    inclusive_a[global - 1usize],
                    inclusive_b[global - 1usize],
                    inclusive_c[global - 1usize],
                    inclusive_d[global - 1usize],
                    inclusive_e[global - 1usize],
                    inclusive_f[global - 1usize],
                    inclusive_g[global - 1usize],
                ),
            );
            output_a[global] = value.0;
            output_b[global] = value.1;
            output_c[global] = value.2;
            output_d[global] = value.3;
            output_e[global] = value.4;
            output_f[global] = value.5;
            output_g[global] = value.6;
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
    Pred: BinaryPredicateOp<K>,
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
pub(crate) fn compact_rejected_scatter_device_expr_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
>(
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
    if global < flags.len() && flags[global] == 0u32 {
        let selected_before_or_at = positions[global];
        let rejected_before = (global as u32) - selected_before_or_at;
        output[rejected_before as usize] =
            Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
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
    Op: BinaryOp<T>,
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B, C)>,
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
    KeyEq: crate::detail::op::kernel::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
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
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && KeyEq::apply(keys[start.read() - 1usize], keys[global]) {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(Expr::eval(input, indices, rhs, rhs_indices, start.read()));
        let index = RuntimeCell::<usize>::new(start.read() + 1usize);
        while index.read() <= global {
            acc.store(Op::apply(
                acc.read(),
                Expr::eval(input, indices, rhs, rhs_indices, index.read()),
            ));
            index.store(index.read() + 1usize);
        }

        output[global] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn inclusive_scan_by_key_expr_keys_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: GpuExpr<K>,
    ValueExpr: GpuExpr<T>,
    KeyEq: crate::detail::op::kernel::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
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
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let current_key = KeyExpr::eval(key_input, key_indices, key_rhs, key_rhs_indices, global);
        let start = RuntimeCell::<usize>::new(global);
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
        while index.read() <= global {
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

        output[global] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_by_key_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    KeyEq: crate::detail::op::kernel::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
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
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let start = RuntimeCell::<usize>::new(global);
        while start.read() > 0usize && KeyEq::apply(keys[start.read() - 1usize], keys[global]) {
            start.store(start.read() - 1usize);
        }

        let acc = RuntimeCell::<T>::new(init[0]);
        let index = RuntimeCell::<usize>::new(start.read());
        while index.read() < global {
            acc.store(Op::apply(
                acc.read(),
                Expr::eval(input, indices, rhs, rhs_indices, index.read()),
            ));
            index.store(index.read() + 1usize);
        }

        output[global] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn exclusive_scan_by_key_expr_keys_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: GpuExpr<K>,
    ValueExpr: GpuExpr<T>,
    KeyEq: crate::detail::op::kernel::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
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
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let current_key = KeyExpr::eval(key_input, key_indices, key_rhs, key_rhs_indices, global);
        let start = RuntimeCell::<usize>::new(global);
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
        while index.read() < global {
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

        output[global] = acc.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_expr_partials_kernel<T: CubePrimitive, Expr: GpuExpr<T>, Op: BinaryOp<T>>(
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
pub(crate) fn tuple1_reduce_last_finalize_kernel<A: CubePrimitive, Op: BinaryOp<(A,)>>(
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B)>,
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
    Op: BinaryOp<(A, B, C)>,
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
    Op: BinaryOp<(A, B, C)>,
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
    Op: BinaryOp<(A, B, C)>,
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

macro_rules! define_tuple_reduce_device_expr_partials_kernel {
    (
        $fn_name:ident,
        $first_partial:ident,
        ($( $ty:ident : $expr:ident : $slot0:ident : $slot1:ident : $slot2:ident : $slot3:ident : $offsets:ident : $partial:ident : $value:ident : $acc:ident : $field:tt ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $( $ty: CubePrimitive, )+
            $( $expr: DeviceGpuExpr<$ty>, )+
            Op: BinaryOp<($( $ty, )+)>,
        >(
            $(
                $slot0: &[$ty],
                $slot1: &[$ty],
                $slot2: &[$ty],
                $slot3: &[$ty],
                $offsets: &[u32],
            )+
            len: &[u32],
            $( $partial: &mut [$ty], )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let logical_len = len[0] as usize;
            $(
                let mut $value = Shared::<[$ty]>::new_slice(cube_dim);
            )+
            let mut valid = Shared::<[u32]>::new_slice(cube_dim);

            let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
            let step = (CUBE_DIM as usize) * $first_partial.len();
            let has_value = RuntimeCell::<u32>::new(0u32);
            $(
                let $acc = RuntimeCell::<$ty>::new($expr::eval(
                    $slot0,
                    $slot1,
                    $slot2,
                    $slot3,
                    $offsets,
                    0,
                ));
            )+

            while i.read() < logical_len {
                let item = (
                    $(
                        $expr::eval($slot0, $slot1, $slot2, $slot3, $offsets, i.read()),
                    )+
                );
                if has_value.read() != 0 {
                    let next = Op::apply(($( $acc.read(), )+), item);
                    $(
                        $acc.store(next.$field);
                    )+
                } else {
                    $(
                        $acc.store(item.$field);
                    )+
                    has_value.store(1u32);
                }
                i.store(i.read() + step);
            }

            $(
                $value[unit] = $acc.read();
            )+
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
                        let next = Op::apply(
                            ($( $value[unit], )+),
                            ($( $value[unit + stride.read()], )+),
                        );
                        $(
                            $value[unit] = next.$field;
                        )+
                    } else {
                        $(
                            $value[unit] = $value[unit + stride.read()];
                        )+
                        valid[unit] = 1u32;
                    }
                }
                sync_cube();
                stride.store(stride.read() / 2);
            }

            if unit == 0 && valid[0] != 0 {
                $(
                    $partial[CUBE_POS as usize] = $value[0];
                )+
            }
        }
    };
}

macro_rules! define_tuple_reduce_partials_kernel {
    (
        $fn_name:ident,
        $first_partial:ident,
        ($( $ty:ident : $input:ident : $partial:ident : $value:ident : $acc:ident : $field:tt ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $( $ty: CubePrimitive, )+
            Op: BinaryOp<($( $ty, )+)>,
        >(
            $( $input: &[$ty], )+
            len: &[u32],
            $( $partial: &mut [$ty], )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let logical_len = len[0] as usize;
            $(
                let mut $value = Shared::<[$ty]>::new_slice(cube_dim);
            )+
            let mut valid = Shared::<[u32]>::new_slice(cube_dim);

            let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
            let step = (CUBE_DIM as usize) * $first_partial.len();
            let has_value = RuntimeCell::<u32>::new(0u32);
            $(
                let $acc = RuntimeCell::<$ty>::new($input[0]);
            )+

            while i.read() < logical_len {
                let item = ($( $input[i.read()], )+);
                if has_value.read() != 0 {
                    let next = Op::apply(($( $acc.read(), )+), item);
                    $(
                        $acc.store(next.$field);
                    )+
                } else {
                    $(
                        $acc.store(item.$field);
                    )+
                    has_value.store(1u32);
                }
                i.store(i.read() + step);
            }

            $(
                $value[unit] = $acc.read();
            )+
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
                        let next = Op::apply(
                            ($( $value[unit], )+),
                            ($( $value[unit + stride.read()], )+),
                        );
                        $(
                            $value[unit] = next.$field;
                        )+
                    } else {
                        $(
                            $value[unit] = $value[unit + stride.read()];
                        )+
                        valid[unit] = 1u32;
                    }
                }
                sync_cube();
                stride.store(stride.read() / 2);
            }

            if unit == 0 && valid[0] != 0 {
                $(
                    $partial[CUBE_POS as usize] = $value[0];
                )+
            }
        }
    };
}

macro_rules! define_tuple_reduce_finalize_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $partial:ident : $init:ident : $output:ident : $field:tt ),+)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $fn_name<
            $( $ty: CubePrimitive, )+
            Op: BinaryOp<($( $ty, )+)>,
        >(
            $( $partial: &[$ty], )+
            $( $init: &[$ty], )+
            $( $output: &mut [$ty], )+
        ) {
            if UNIT_POS == 0 {
                let item = Op::apply(($( $init[0], )+), ($( $partial[0], )+));
                $(
                    $output[0] = item.$field;
                )+
            }
        }
    };
}

define_tuple_reduce_device_expr_partials_kernel!(
    tuple7_device_reduce_expr_partials_kernel,
    partial_a,
    (
        A: ExprA: a_slot0: a_slot1: a_slot2: a_slot3: a_offsets: partial_a: values_a: acc_a: 0,
        B: ExprB: b_slot0: b_slot1: b_slot2: b_slot3: b_offsets: partial_b: values_b: acc_b: 1,
        C: ExprC: c_slot0: c_slot1: c_slot2: c_slot3: c_offsets: partial_c: values_c: acc_c: 2,
        D: ExprD: d_slot0: d_slot1: d_slot2: d_slot3: d_offsets: partial_d: values_d: acc_d: 3,
        E: ExprE: e_slot0: e_slot1: e_slot2: e_slot3: e_offsets: partial_e: values_e: acc_e: 4,
        F: ExprF: f_slot0: f_slot1: f_slot2: f_slot3: f_offsets: partial_f: values_f: acc_f: 5,
        G: ExprG: g_slot0: g_slot1: g_slot2: g_slot3: g_offsets: partial_g: values_g: acc_g: 6
    )
);

define_tuple_reduce_partials_kernel!(
    tuple7_reduce_partials_kernel,
    partial_a,
    (
        A: input_a: partial_a: values_a: acc_a: 0,
        B: input_b: partial_b: values_b: acc_b: 1,
        C: input_c: partial_c: values_c: acc_c: 2,
        D: input_d: partial_d: values_d: acc_d: 3,
        E: input_e: partial_e: values_e: acc_e: 4,
        F: input_f: partial_f: values_f: acc_f: 5,
        G: input_g: partial_g: values_g: acc_g: 6
    )
);

define_tuple_reduce_finalize_kernel!(
    tuple7_reduce_finalize_kernel,
    (
        A: partial_a: init_a: output_a: 0,
        B: partial_b: init_b: output_b: 1,
        C: partial_c: init_c: output_c: 2,
        D: partial_d: init_d: output_d: 3,
        E: partial_e: init_e: output_e: 4,
        F: partial_f: init_f: output_f: 5,
        G: partial_g: init_g: output_g: 6
    )
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn count_if_expr_partials_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp<T>,
>(
    env: Pred::Env,
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
        if Pred::apply(
            env.clone(),
            Expr::eval(input, indices, rhs, rhs_indices, i.read()),
        ) {
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
    Pred: PredicateOp<T>,
>(
    env: Pred::Env,
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
        let matches = Pred::apply(
            env.clone(),
            Expr::eval(input, indices, rhs, rhs_indices, i.read()),
        );
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
    Pred: BinaryPredicateOp<T>,
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
    Less: BinaryPredicateOp<T>,
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
