use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp},
    expr::{
        DeviceGpuExpr, GpuExpr, LogicalDeviceExpr3, LogicalDeviceExpr7, LogicalDevicePack3,
        LogicalDevicePack7,
    },
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
    transform_tuple2_to_tuple4_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple2_to_tuple5_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple2_to_tuple6_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple2_to_tuple7_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
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
    transform_tuple3_to_tuple4_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical3_to_tuple1_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA,)>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        output_a[global] = output.0;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical3_to_tuple2_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB)>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        output_a[global] = output.0;
        output_b[global] = output.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical3_to_tuple3_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC)>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical3_to_tuple4_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD)>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical3_to_tuple5_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    OutE: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD, OutE)>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
    output_e: &mut [OutE],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
        output_e[global] = output.4;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical3_to_tuple6_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    OutE: CubePrimitive,
    OutF: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD, OutE, OutF)>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
    output_e: &mut [OutE],
    output_f: &mut [OutF],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
        output_e[global] = output.4;
        output_f[global] = output.5;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical3_to_tuple7_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    OutE: CubePrimitive,
    OutF: CubePrimitive,
    OutG: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD, OutE, OutF, OutG)>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
    output_e: &mut [OutE],
    output_f: &mut [OutF],
    output_g: &mut [OutG],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
        output_e[global] = output.4;
        output_f[global] = output.5;
        output_g[global] = output.6;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical3_predicate_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    Pred: PredicateOp<Input>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply(Expr::eval3(slot0, slot1, slot2, slot_offsets, global));
        flags[global] = if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            1u32
        } else {
            0u32
        };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical3_mismatch_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeftLeafA: CubePrimitive,
    LeftLeafB: CubePrimitive,
    LeftLeafC: CubePrimitive,
    RightLeafA: CubePrimitive,
    RightLeafB: CubePrimitive,
    RightLeafC: CubePrimitive,
    LeftExpr: LogicalDeviceExpr3<Input, LeftLeafA, LeftLeafB, LeftLeafC>,
    RightExpr: LogicalDeviceExpr3<Input, RightLeafA, RightLeafB, RightLeafC>,
    Eq: BinaryPredicateOp<Input>,
>(
    left_slot0: &[LeftLeafA],
    left_slot1: &[LeftLeafB],
    left_slot2: &[LeftLeafC],
    left_slot_offsets: &[u32],
    right_slot0: &[RightLeafA],
    right_slot1: &[RightLeafB],
    right_slot2: &[RightLeafC],
    right_slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let left = LeftExpr::eval3(
            left_slot0,
            left_slot1,
            left_slot2,
            left_slot_offsets,
            global,
        );
        let right = RightExpr::eval3(
            right_slot0,
            right_slot1,
            right_slot2,
            right_slot_offsets,
            global,
        );
        flags[global] = if Eq::apply(left, right) { 0u32 } else { 1u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical3_adjacent_find_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    Pred: BinaryPredicateOp<Input>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let left = Expr::eval3(slot0, slot1, slot2, slot_offsets, global);
        let right = Expr::eval3(slot0, slot1, slot2, slot_offsets, global + 1);
        flags[global] = if Pred::apply(left, right) { 1u32 } else { 0u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical3_sorted_break_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    Less: BinaryPredicateOp<Input>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let current = Expr::eval3(slot0, slot1, slot2, slot_offsets, global);
        let next = Expr::eval3(slot0, slot1, slot2, slot_offsets, global + 1);
        flags[global] = if Less::apply(next, current) {
            1u32
        } else {
            0u32
        };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical3_minmax_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    Less: BinaryPredicateOp<Input>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    min_flags: &mut [u32],
    max_flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let mut is_min = true;
        let mut is_max = true;
        let mut index = 0usize;
        while index < (len[0] as usize) {
            if Less::apply(
                Expr::eval3(slot0, slot1, slot2, slot_offsets, index),
                Expr::eval3(slot0, slot1, slot2, slot_offsets, global),
            ) {
                is_min = false;
            }
            if Less::apply(
                Expr::eval3(slot0, slot1, slot2, slot_offsets, global),
                Expr::eval3(slot0, slot1, slot2, slot_offsets, index),
            ) {
                is_max = false;
            }
            if index < global
                && !Less::apply(
                    Expr::eval3(slot0, slot1, slot2, slot_offsets, global),
                    Expr::eval3(slot0, slot1, slot2, slot_offsets, index),
                )
                && !Less::apply(
                    Expr::eval3(slot0, slot1, slot2, slot_offsets, index),
                    Expr::eval3(slot0, slot1, slot2, slot_offsets, global),
                )
            {
                is_min = false;
                is_max = false;
            }
            index += 1usize;
        }
        min_flags[global] = if is_min { 1u32 } else { 0u32 };
        max_flags[global] = if is_max { 1u32 } else { 0u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical3_lower_bound_many_kernel<
    Input: CubeType + 'static + Send + Sync,
    SourceLeafA: CubePrimitive,
    SourceLeafB: CubePrimitive,
    SourceLeafC: CubePrimitive,
    ValueLeafA: CubePrimitive,
    ValueLeafB: CubePrimitive,
    ValueLeafC: CubePrimitive,
    SourceExpr: LogicalDeviceExpr3<Input, SourceLeafA, SourceLeafB, SourceLeafC>,
    ValueExpr: LogicalDeviceExpr3<Input, ValueLeafA, ValueLeafB, ValueLeafC>,
    Less: BinaryPredicateOp<Input>,
>(
    source0: &[SourceLeafA],
    source1: &[SourceLeafB],
    source2: &[SourceLeafC],
    source_offsets: &[u32],
    value0: &[ValueLeafA],
    value1: &[ValueLeafB],
    value2: &[ValueLeafC],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let candidate = SourceExpr::eval3(source0, source1, source2, source_offsets, mid);
            if Less::apply(
                candidate,
                ValueExpr::eval3(value0, value1, value2, value_offsets, global),
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
pub(crate) fn logical3_upper_bound_many_kernel<
    Input: CubeType + 'static + Send + Sync,
    SourceLeafA: CubePrimitive,
    SourceLeafB: CubePrimitive,
    SourceLeafC: CubePrimitive,
    ValueLeafA: CubePrimitive,
    ValueLeafB: CubePrimitive,
    ValueLeafC: CubePrimitive,
    SourceExpr: LogicalDeviceExpr3<Input, SourceLeafA, SourceLeafB, SourceLeafC>,
    ValueExpr: LogicalDeviceExpr3<Input, ValueLeafA, ValueLeafB, ValueLeafC>,
    Less: BinaryPredicateOp<Input>,
>(
    source0: &[SourceLeafA],
    source1: &[SourceLeafB],
    source2: &[SourceLeafC],
    source_offsets: &[u32],
    value0: &[ValueLeafA],
    value1: &[ValueLeafB],
    value2: &[ValueLeafC],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let candidate = SourceExpr::eval3(source0, source1, source2, source_offsets, mid);
            if !Less::apply(
                ValueExpr::eval3(value0, value1, value2, value_offsets, global),
                candidate,
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
pub(crate) fn logical3_reduce_expr_partials_kernel<
    Item: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Expr: LogicalDeviceExpr3<Item, LeafA, LeafB, LeafC>,
    Pack: LogicalDevicePack3<Item, LeafA, LeafB, LeafC> + 'static + Send + Sync,
    Op: BinaryOp<Item>,
>(
    slot0: &[LeafA],
    slot1: &[LeafB],
    slot2: &[LeafC],
    slot_offsets: &[u32],
    len: &[u32],
    partial_a: &mut [LeafA],
    partial_b: &mut [LeafB],
    partial_c: &mut [LeafC],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[LeafA]>::new_slice(cube_dim);
    let mut values_b = Shared::<[LeafB]>::new_slice(cube_dim);
    let mut values_c = Shared::<[LeafC]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_a.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let first = Expr::eval3(slot0, slot1, slot2, slot_offsets, 0);
    let first = Pack::unpack(first);
    let acc_a = RuntimeCell::<LeafA>::new(first.0);
    let acc_b = RuntimeCell::<LeafB>::new(first.1);
    let acc_c = RuntimeCell::<LeafC>::new(first.2);

    while i.read() < logical_len {
        if has_value.read() != 0 {
            let value = Expr::eval3(slot0, slot1, slot2, slot_offsets, i.read());
            let next = Op::apply(Pack::pack(acc_a.read(), acc_b.read(), acc_c.read()), value);
            let next = Pack::unpack(next);
            acc_a.store(next.0);
            acc_b.store(next.1);
            acc_c.store(next.2);
        } else {
            let value = Expr::eval3(slot0, slot1, slot2, slot_offsets, i.read());
            let value = Pack::unpack(value);
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
    valid[unit] = if has_value.read() != 0 { 1u32 } else { 0u32 };
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let next = Op::apply(
                    Pack::pack(values_a[unit], values_b[unit], values_c[unit]),
                    Pack::pack(
                        values_a[unit + stride.read()],
                        values_b[unit + stride.read()],
                        values_c[unit + stride.read()],
                    ),
                );
                let next = Pack::unpack(next);
                values_a[unit] = next.0;
                values_b[unit] = next.1;
                values_c[unit] = next.2;
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
pub(crate) fn logical3_reduce_partials_kernel<
    Item: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Pack: LogicalDevicePack3<Item, LeafA, LeafB, LeafC> + 'static + Send + Sync,
    Op: BinaryOp<Item>,
>(
    input_a: &[LeafA],
    input_b: &[LeafB],
    input_c: &[LeafC],
    len: &[u32],
    partial_a: &mut [LeafA],
    partial_b: &mut [LeafB],
    partial_c: &mut [LeafC],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values_a = Shared::<[LeafA]>::new_slice(cube_dim);
    let mut values_b = Shared::<[LeafB]>::new_slice(cube_dim);
    let mut values_c = Shared::<[LeafC]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial_a.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc_a = RuntimeCell::<LeafA>::new(input_a[0]);
    let acc_b = RuntimeCell::<LeafB>::new(input_b[0]);
    let acc_c = RuntimeCell::<LeafC>::new(input_c[0]);

    while i.read() < logical_len {
        if has_value.read() != 0 {
            let value = Pack::pack(input_a[i.read()], input_b[i.read()], input_c[i.read()]);
            let next = Op::apply(Pack::pack(acc_a.read(), acc_b.read(), acc_c.read()), value);
            let next = Pack::unpack(next);
            acc_a.store(next.0);
            acc_b.store(next.1);
            acc_c.store(next.2);
        } else {
            let value = Pack::pack(input_a[i.read()], input_b[i.read()], input_c[i.read()]);
            let value = Pack::unpack(value);
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
    valid[unit] = if has_value.read() != 0 { 1u32 } else { 0u32 };
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let next = Op::apply(
                    Pack::pack(values_a[unit], values_b[unit], values_c[unit]),
                    Pack::pack(
                        values_a[unit + stride.read()],
                        values_b[unit + stride.read()],
                        values_c[unit + stride.read()],
                    ),
                );
                let next = Pack::unpack(next);
                values_a[unit] = next.0;
                values_b[unit] = next.1;
                values_c[unit] = next.2;
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
pub(crate) fn logical3_reduce_finalize_kernel<
    Item: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive,
    LeafB: CubePrimitive,
    LeafC: CubePrimitive,
    Pack: LogicalDevicePack3<Item, LeafA, LeafB, LeafC> + 'static + Send + Sync,
    Op: BinaryOp<Item>,
>(
    partial_a: &[LeafA],
    partial_b: &[LeafB],
    partial_c: &[LeafC],
    init_a: &[LeafA],
    init_b: &[LeafB],
    init_c: &[LeafC],
    output_a: &mut [LeafA],
    output_b: &mut [LeafB],
    output_c: &mut [LeafC],
) {
    if UNIT_POS == 0 {
        let output = Op::apply(
            Pack::pack(init_a[0], init_b[0], init_c[0]),
            Pack::pack(partial_a[0], partial_b[0], partial_c[0]),
        );
        let output = Pack::unpack(output);
        output_a[0] = output.0;
        output_b[0] = output.1;
        output_c[0] = output.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_reduce_expr_partials_kernel<
    Item: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    Pack: LogicalDevicePack7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
        + 'static
        + Send
        + Sync,
    Op: BinaryOp<Item>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    partial0: &mut [Leaf0],
    partial1: &mut [Leaf1],
    partial2: &mut [Leaf2],
    partial3: &mut [Leaf3],
    partial4: &mut [Leaf4],
    partial5: &mut [Leaf5],
    partial6: &mut [Leaf6],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values0 = Shared::<[Leaf0]>::new_slice(cube_dim);
    let mut values1 = Shared::<[Leaf1]>::new_slice(cube_dim);
    let mut values2 = Shared::<[Leaf2]>::new_slice(cube_dim);
    let mut values3 = Shared::<[Leaf3]>::new_slice(cube_dim);
    let mut values4 = Shared::<[Leaf4]>::new_slice(cube_dim);
    let mut values5 = Shared::<[Leaf5]>::new_slice(cube_dim);
    let mut values6 = Shared::<[Leaf6]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial0.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let first = Expr::eval7(
        slot0,
        slot1,
        slot2,
        slot3,
        slot4,
        slot5,
        slot6,
        slot_offsets,
        0,
    );
    let first = Pack::unpack(first);
    let acc0 = RuntimeCell::<Leaf0>::new(first.0);
    let acc1 = RuntimeCell::<Leaf1>::new(first.1);
    let acc2 = RuntimeCell::<Leaf2>::new(first.2);
    let acc3 = RuntimeCell::<Leaf3>::new(first.3);
    let acc4 = RuntimeCell::<Leaf4>::new(first.4);
    let acc5 = RuntimeCell::<Leaf5>::new(first.5);
    let acc6 = RuntimeCell::<Leaf6>::new(first.6);

    while i.read() < logical_len {
        if has_value.read() != 0 {
            let value = Expr::eval7(
                slot0,
                slot1,
                slot2,
                slot3,
                slot4,
                slot5,
                slot6,
                slot_offsets,
                i.read(),
            );
            let next = Op::apply(
                Pack::pack(
                    acc0.read(),
                    acc1.read(),
                    acc2.read(),
                    acc3.read(),
                    acc4.read(),
                    acc5.read(),
                    acc6.read(),
                ),
                value,
            );
            let next = Pack::unpack(next);
            acc0.store(next.0);
            acc1.store(next.1);
            acc2.store(next.2);
            acc3.store(next.3);
            acc4.store(next.4);
            acc5.store(next.5);
            acc6.store(next.6);
        } else {
            let value = Expr::eval7(
                slot0,
                slot1,
                slot2,
                slot3,
                slot4,
                slot5,
                slot6,
                slot_offsets,
                i.read(),
            );
            let value = Pack::unpack(value);
            acc0.store(value.0);
            acc1.store(value.1);
            acc2.store(value.2);
            acc3.store(value.3);
            acc4.store(value.4);
            acc5.store(value.5);
            acc6.store(value.6);
            has_value.store(1u32);
        }
        i.store(i.read() + step);
    }

    values0[unit] = acc0.read();
    values1[unit] = acc1.read();
    values2[unit] = acc2.read();
    values3[unit] = acc3.read();
    values4[unit] = acc4.read();
    values5[unit] = acc5.read();
    values6[unit] = acc6.read();
    valid[unit] = if has_value.read() != 0 { 1u32 } else { 0u32 };
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let next = Op::apply(
                    Pack::pack(
                        values0[unit],
                        values1[unit],
                        values2[unit],
                        values3[unit],
                        values4[unit],
                        values5[unit],
                        values6[unit],
                    ),
                    Pack::pack(
                        values0[unit + stride.read()],
                        values1[unit + stride.read()],
                        values2[unit + stride.read()],
                        values3[unit + stride.read()],
                        values4[unit + stride.read()],
                        values5[unit + stride.read()],
                        values6[unit + stride.read()],
                    ),
                );
                let next = Pack::unpack(next);
                values0[unit] = next.0;
                values1[unit] = next.1;
                values2[unit] = next.2;
                values3[unit] = next.3;
                values4[unit] = next.4;
                values5[unit] = next.5;
                values6[unit] = next.6;
            } else {
                values0[unit] = values0[unit + stride.read()];
                values1[unit] = values1[unit + stride.read()];
                values2[unit] = values2[unit + stride.read()];
                values3[unit] = values3[unit + stride.read()];
                values4[unit] = values4[unit + stride.read()];
                values5[unit] = values5[unit + stride.read()];
                values6[unit] = values6[unit + stride.read()];
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 && valid[0] != 0 {
        partial0[CUBE_POS as usize] = values0[0];
        partial1[CUBE_POS as usize] = values1[0];
        partial2[CUBE_POS as usize] = values2[0];
        partial3[CUBE_POS as usize] = values3[0];
        partial4[CUBE_POS as usize] = values4[0];
        partial5[CUBE_POS as usize] = values5[0];
        partial6[CUBE_POS as usize] = values6[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_reduce_partials_kernel<
    Item: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Pack: LogicalDevicePack7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
        + 'static
        + Send
        + Sync,
    Op: BinaryOp<Item>,
>(
    input0: &[Leaf0],
    input1: &[Leaf1],
    input2: &[Leaf2],
    input3: &[Leaf3],
    input4: &[Leaf4],
    input5: &[Leaf5],
    input6: &[Leaf6],
    len: &[u32],
    partial0: &mut [Leaf0],
    partial1: &mut [Leaf1],
    partial2: &mut [Leaf2],
    partial3: &mut [Leaf3],
    partial4: &mut [Leaf4],
    partial5: &mut [Leaf5],
    partial6: &mut [Leaf6],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values0 = Shared::<[Leaf0]>::new_slice(cube_dim);
    let mut values1 = Shared::<[Leaf1]>::new_slice(cube_dim);
    let mut values2 = Shared::<[Leaf2]>::new_slice(cube_dim);
    let mut values3 = Shared::<[Leaf3]>::new_slice(cube_dim);
    let mut values4 = Shared::<[Leaf4]>::new_slice(cube_dim);
    let mut values5 = Shared::<[Leaf5]>::new_slice(cube_dim);
    let mut values6 = Shared::<[Leaf6]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partial0.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc0 = RuntimeCell::<Leaf0>::new(input0[0]);
    let acc1 = RuntimeCell::<Leaf1>::new(input1[0]);
    let acc2 = RuntimeCell::<Leaf2>::new(input2[0]);
    let acc3 = RuntimeCell::<Leaf3>::new(input3[0]);
    let acc4 = RuntimeCell::<Leaf4>::new(input4[0]);
    let acc5 = RuntimeCell::<Leaf5>::new(input5[0]);
    let acc6 = RuntimeCell::<Leaf6>::new(input6[0]);

    while i.read() < logical_len {
        if has_value.read() != 0 {
            let value = Pack::pack(
                input0[i.read()],
                input1[i.read()],
                input2[i.read()],
                input3[i.read()],
                input4[i.read()],
                input5[i.read()],
                input6[i.read()],
            );
            let next = Op::apply(
                Pack::pack(
                    acc0.read(),
                    acc1.read(),
                    acc2.read(),
                    acc3.read(),
                    acc4.read(),
                    acc5.read(),
                    acc6.read(),
                ),
                value,
            );
            let next = Pack::unpack(next);
            acc0.store(next.0);
            acc1.store(next.1);
            acc2.store(next.2);
            acc3.store(next.3);
            acc4.store(next.4);
            acc5.store(next.5);
            acc6.store(next.6);
        } else {
            acc0.store(input0[i.read()]);
            acc1.store(input1[i.read()]);
            acc2.store(input2[i.read()]);
            acc3.store(input3[i.read()]);
            acc4.store(input4[i.read()]);
            acc5.store(input5[i.read()]);
            acc6.store(input6[i.read()]);
            has_value.store(1u32);
        }
        i.store(i.read() + step);
    }

    values0[unit] = acc0.read();
    values1[unit] = acc1.read();
    values2[unit] = acc2.read();
    values3[unit] = acc3.read();
    values4[unit] = acc4.read();
    values5[unit] = acc5.read();
    values6[unit] = acc6.read();
    valid[unit] = if has_value.read() != 0 { 1u32 } else { 0u32 };
    sync_cube();

    let stride = RuntimeCell::<usize>::new(cube_dim / 2);
    while stride.read() > 0 {
        if unit < stride.read() && valid[unit + stride.read()] != 0 {
            if valid[unit] != 0 {
                let next = Op::apply(
                    Pack::pack(
                        values0[unit],
                        values1[unit],
                        values2[unit],
                        values3[unit],
                        values4[unit],
                        values5[unit],
                        values6[unit],
                    ),
                    Pack::pack(
                        values0[unit + stride.read()],
                        values1[unit + stride.read()],
                        values2[unit + stride.read()],
                        values3[unit + stride.read()],
                        values4[unit + stride.read()],
                        values5[unit + stride.read()],
                        values6[unit + stride.read()],
                    ),
                );
                let next = Pack::unpack(next);
                values0[unit] = next.0;
                values1[unit] = next.1;
                values2[unit] = next.2;
                values3[unit] = next.3;
                values4[unit] = next.4;
                values5[unit] = next.5;
                values6[unit] = next.6;
            } else {
                values0[unit] = values0[unit + stride.read()];
                values1[unit] = values1[unit + stride.read()];
                values2[unit] = values2[unit + stride.read()];
                values3[unit] = values3[unit + stride.read()];
                values4[unit] = values4[unit + stride.read()];
                values5[unit] = values5[unit + stride.read()];
                values6[unit] = values6[unit + stride.read()];
                valid[unit] = 1u32;
            }
        }
        sync_cube();
        stride.store(stride.read() / 2);
    }

    if unit == 0 && valid[0] != 0 {
        partial0[CUBE_POS as usize] = values0[0];
        partial1[CUBE_POS as usize] = values1[0];
        partial2[CUBE_POS as usize] = values2[0];
        partial3[CUBE_POS as usize] = values3[0];
        partial4[CUBE_POS as usize] = values4[0];
        partial5[CUBE_POS as usize] = values5[0];
        partial6[CUBE_POS as usize] = values6[0];
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_reduce_finalize_kernel<
    Item: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Pack: LogicalDevicePack7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
        + 'static
        + Send
        + Sync,
    Op: BinaryOp<Item>,
>(
    partial0: &[Leaf0],
    partial1: &[Leaf1],
    partial2: &[Leaf2],
    partial3: &[Leaf3],
    partial4: &[Leaf4],
    partial5: &[Leaf5],
    partial6: &[Leaf6],
    init0: &[Leaf0],
    init1: &[Leaf1],
    init2: &[Leaf2],
    init3: &[Leaf3],
    init4: &[Leaf4],
    init5: &[Leaf5],
    init6: &[Leaf6],
    output0: &mut [Leaf0],
    output1: &mut [Leaf1],
    output2: &mut [Leaf2],
    output3: &mut [Leaf3],
    output4: &mut [Leaf4],
    output5: &mut [Leaf5],
    output6: &mut [Leaf6],
) {
    if UNIT_POS == 0 {
        let output = Op::apply(
            Pack::pack(
                init0[0], init1[0], init2[0], init3[0], init4[0], init5[0], init6[0],
            ),
            Pack::pack(
                partial0[0],
                partial1[0],
                partial2[0],
                partial3[0],
                partial4[0],
                partial5[0],
                partial6[0],
            ),
        );
        let output = Pack::unpack(output);
        output0[0] = output.0;
        output1[0] = output.1;
        output2[0] = output.2;
        output3[0] = output.3;
        output4[0] = output.4;
        output5[0] = output.5;
        output6[0] = output.6;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical7_to_tuple1_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    OutA: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA,)>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        output_a[global] = output.0;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical7_to_tuple2_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB)>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        output_a[global] = output.0;
        output_b[global] = output.1;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical7_to_tuple3_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC)>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical7_to_tuple4_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD)>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical7_to_tuple5_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    OutE: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD, OutE)>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
    output_e: &mut [OutE],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
        output_e[global] = output.4;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical7_to_tuple6_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    OutE: CubePrimitive,
    OutF: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD, OutE, OutF)>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
    output_e: &mut [OutE],
    output_f: &mut [OutF],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
        output_e[global] = output.4;
        output_f[global] = output.5;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_logical7_to_tuple7_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    OutD: CubePrimitive,
    OutE: CubePrimitive,
    OutF: CubePrimitive,
    OutG: CubePrimitive,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC, OutD, OutE, OutF, OutG)>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    output_a: &mut [OutA],
    output_b: &mut [OutB],
    output_c: &mut [OutC],
    output_d: &mut [OutD],
    output_e: &mut [OutE],
    output_f: &mut [OutF],
    output_g: &mut [OutG],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
        output_e[global] = output.4;
        output_f[global] = output.5;
        output_g[global] = output.6;
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_predicate_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    Pred: PredicateOp<Input>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply(Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        ));
        flags[global] = if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            1u32
        } else {
            0u32
        };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_mismatch_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeftLeaf0: CubePrimitive,
    LeftLeaf1: CubePrimitive,
    LeftLeaf2: CubePrimitive,
    LeftLeaf3: CubePrimitive,
    LeftLeaf4: CubePrimitive,
    LeftLeaf5: CubePrimitive,
    LeftLeaf6: CubePrimitive,
    RightLeaf0: CubePrimitive,
    RightLeaf1: CubePrimitive,
    RightLeaf2: CubePrimitive,
    RightLeaf3: CubePrimitive,
    RightLeaf4: CubePrimitive,
    RightLeaf5: CubePrimitive,
    RightLeaf6: CubePrimitive,
    LeftExpr: LogicalDeviceExpr7<
            Input,
            LeftLeaf0,
            LeftLeaf1,
            LeftLeaf2,
            LeftLeaf3,
            LeftLeaf4,
            LeftLeaf5,
            LeftLeaf6,
        >,
    RightExpr: LogicalDeviceExpr7<
            Input,
            RightLeaf0,
            RightLeaf1,
            RightLeaf2,
            RightLeaf3,
            RightLeaf4,
            RightLeaf5,
            RightLeaf6,
        >,
    Eq: BinaryPredicateOp<Input>,
>(
    left_slot0: &[LeftLeaf0],
    left_slot1: &[LeftLeaf1],
    left_slot2: &[LeftLeaf2],
    left_slot3: &[LeftLeaf3],
    left_slot4: &[LeftLeaf4],
    left_slot5: &[LeftLeaf5],
    left_slot6: &[LeftLeaf6],
    left_slot_offsets: &[u32],
    right_slot0: &[RightLeaf0],
    right_slot1: &[RightLeaf1],
    right_slot2: &[RightLeaf2],
    right_slot3: &[RightLeaf3],
    right_slot4: &[RightLeaf4],
    right_slot5: &[RightLeaf5],
    right_slot6: &[RightLeaf6],
    right_slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let left = LeftExpr::eval7(
            left_slot0,
            left_slot1,
            left_slot2,
            left_slot3,
            left_slot4,
            left_slot5,
            left_slot6,
            left_slot_offsets,
            global,
        );
        let right = RightExpr::eval7(
            right_slot0,
            right_slot1,
            right_slot2,
            right_slot3,
            right_slot4,
            right_slot5,
            right_slot6,
            right_slot_offsets,
            global,
        );
        flags[global] = if Eq::apply(left, right) { 0u32 } else { 1u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_find_first_of_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    InputLeaf0: CubePrimitive,
    InputLeaf1: CubePrimitive,
    InputLeaf2: CubePrimitive,
    InputLeaf3: CubePrimitive,
    InputLeaf4: CubePrimitive,
    InputLeaf5: CubePrimitive,
    InputLeaf6: CubePrimitive,
    NeedleLeaf0: CubePrimitive,
    NeedleLeaf1: CubePrimitive,
    NeedleLeaf2: CubePrimitive,
    NeedleLeaf3: CubePrimitive,
    NeedleLeaf4: CubePrimitive,
    NeedleLeaf5: CubePrimitive,
    NeedleLeaf6: CubePrimitive,
    InputExpr: LogicalDeviceExpr7<
            Input,
            InputLeaf0,
            InputLeaf1,
            InputLeaf2,
            InputLeaf3,
            InputLeaf4,
            InputLeaf5,
            InputLeaf6,
        >,
    NeedleExpr: LogicalDeviceExpr7<
            Input,
            NeedleLeaf0,
            NeedleLeaf1,
            NeedleLeaf2,
            NeedleLeaf3,
            NeedleLeaf4,
            NeedleLeaf5,
            NeedleLeaf6,
        >,
    Eq: BinaryPredicateOp<Input>,
>(
    input_slot0: &[InputLeaf0],
    input_slot1: &[InputLeaf1],
    input_slot2: &[InputLeaf2],
    input_slot3: &[InputLeaf3],
    input_slot4: &[InputLeaf4],
    input_slot5: &[InputLeaf5],
    input_slot6: &[InputLeaf6],
    input_slot_offsets: &[u32],
    needle_slot0: &[NeedleLeaf0],
    needle_slot1: &[NeedleLeaf1],
    needle_slot2: &[NeedleLeaf2],
    needle_slot3: &[NeedleLeaf3],
    needle_slot4: &[NeedleLeaf4],
    needle_slot5: &[NeedleLeaf5],
    needle_slot6: &[NeedleLeaf6],
    needle_slot_offsets: &[u32],
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
            let value = InputExpr::eval7(
                input_slot0,
                input_slot1,
                input_slot2,
                input_slot3,
                input_slot4,
                input_slot5,
                input_slot6,
                input_slot_offsets,
                global,
            );
            let candidate = NeedleExpr::eval7(
                needle_slot0,
                needle_slot1,
                needle_slot2,
                needle_slot3,
                needle_slot4,
                needle_slot5,
                needle_slot6,
                needle_slot_offsets,
                needle.read(),
            );
            if Eq::apply(value, candidate) {
                found.store(1u32);
            }
            needle.store(needle.read() + 1usize);
        }
        flags[global] = found.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_lexicographical_diff_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeftLeaf0: CubePrimitive,
    LeftLeaf1: CubePrimitive,
    LeftLeaf2: CubePrimitive,
    LeftLeaf3: CubePrimitive,
    LeftLeaf4: CubePrimitive,
    LeftLeaf5: CubePrimitive,
    LeftLeaf6: CubePrimitive,
    RightLeaf0: CubePrimitive,
    RightLeaf1: CubePrimitive,
    RightLeaf2: CubePrimitive,
    RightLeaf3: CubePrimitive,
    RightLeaf4: CubePrimitive,
    RightLeaf5: CubePrimitive,
    RightLeaf6: CubePrimitive,
    LeftExpr: LogicalDeviceExpr7<
            Input,
            LeftLeaf0,
            LeftLeaf1,
            LeftLeaf2,
            LeftLeaf3,
            LeftLeaf4,
            LeftLeaf5,
            LeftLeaf6,
        >,
    RightExpr: LogicalDeviceExpr7<
            Input,
            RightLeaf0,
            RightLeaf1,
            RightLeaf2,
            RightLeaf3,
            RightLeaf4,
            RightLeaf5,
            RightLeaf6,
        >,
    Less: BinaryPredicateOp<Input>,
>(
    left_slot0: &[LeftLeaf0],
    left_slot1: &[LeftLeaf1],
    left_slot2: &[LeftLeaf2],
    left_slot3: &[LeftLeaf3],
    left_slot4: &[LeftLeaf4],
    left_slot5: &[LeftLeaf5],
    left_slot6: &[LeftLeaf6],
    left_slot_offsets: &[u32],
    right_slot0: &[RightLeaf0],
    right_slot1: &[RightLeaf1],
    right_slot2: &[RightLeaf2],
    right_slot3: &[RightLeaf3],
    right_slot4: &[RightLeaf4],
    right_slot5: &[RightLeaf5],
    right_slot6: &[RightLeaf6],
    right_slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let left = LeftExpr::eval7(
            left_slot0,
            left_slot1,
            left_slot2,
            left_slot3,
            left_slot4,
            left_slot5,
            left_slot6,
            left_slot_offsets,
            global,
        );
        let right = RightExpr::eval7(
            right_slot0,
            right_slot1,
            right_slot2,
            right_slot3,
            right_slot4,
            right_slot5,
            right_slot6,
            right_slot_offsets,
            global,
        );
        if Less::apply(left, right) {
            flags[global] = 1u32;
        } else {
            let left = LeftExpr::eval7(
                left_slot0,
                left_slot1,
                left_slot2,
                left_slot3,
                left_slot4,
                left_slot5,
                left_slot6,
                left_slot_offsets,
                global,
            );
            let right = RightExpr::eval7(
                right_slot0,
                right_slot1,
                right_slot2,
                right_slot3,
                right_slot4,
                right_slot5,
                right_slot6,
                right_slot_offsets,
                global,
            );
            flags[global] = if Less::apply(right, left) { 1u32 } else { 0u32 };
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_lexicographical_compare_at_kernel<
    Input: CubeType + 'static + Send + Sync,
    LeftLeaf0: CubePrimitive,
    LeftLeaf1: CubePrimitive,
    LeftLeaf2: CubePrimitive,
    LeftLeaf3: CubePrimitive,
    LeftLeaf4: CubePrimitive,
    LeftLeaf5: CubePrimitive,
    LeftLeaf6: CubePrimitive,
    RightLeaf0: CubePrimitive,
    RightLeaf1: CubePrimitive,
    RightLeaf2: CubePrimitive,
    RightLeaf3: CubePrimitive,
    RightLeaf4: CubePrimitive,
    RightLeaf5: CubePrimitive,
    RightLeaf6: CubePrimitive,
    LeftExpr: LogicalDeviceExpr7<
            Input,
            LeftLeaf0,
            LeftLeaf1,
            LeftLeaf2,
            LeftLeaf3,
            LeftLeaf4,
            LeftLeaf5,
            LeftLeaf6,
        >,
    RightExpr: LogicalDeviceExpr7<
            Input,
            RightLeaf0,
            RightLeaf1,
            RightLeaf2,
            RightLeaf3,
            RightLeaf4,
            RightLeaf5,
            RightLeaf6,
        >,
    Less: BinaryPredicateOp<Input>,
>(
    left_slot0: &[LeftLeaf0],
    left_slot1: &[LeftLeaf1],
    left_slot2: &[LeftLeaf2],
    left_slot3: &[LeftLeaf3],
    left_slot4: &[LeftLeaf4],
    left_slot5: &[LeftLeaf5],
    left_slot6: &[LeftLeaf6],
    left_slot_offsets: &[u32],
    right_slot0: &[RightLeaf0],
    right_slot1: &[RightLeaf1],
    right_slot2: &[RightLeaf2],
    right_slot3: &[RightLeaf3],
    right_slot4: &[RightLeaf4],
    right_slot5: &[RightLeaf5],
    right_slot6: &[RightLeaf6],
    right_slot_offsets: &[u32],
    index: &[u32],
    output: &mut [u32],
) {
    if UNIT_POS == 0 {
        let i = index[0] as usize;
        let left = LeftExpr::eval7(
            left_slot0,
            left_slot1,
            left_slot2,
            left_slot3,
            left_slot4,
            left_slot5,
            left_slot6,
            left_slot_offsets,
            i,
        );
        let right = RightExpr::eval7(
            right_slot0,
            right_slot1,
            right_slot2,
            right_slot3,
            right_slot4,
            right_slot5,
            right_slot6,
            right_slot_offsets,
            i,
        );
        output[0] = if Less::apply(left, right) { 1u32 } else { 0u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_adjacent_find_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    Pred: BinaryPredicateOp<Input>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let left = Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        );
        let right = Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global + 1,
        );
        flags[global] = if Pred::apply(left, right) { 1u32 } else { 0u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_scan_by_key_head_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    KeyEq: BinaryPredicateOp<Input>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if global == 0usize {
            flags[global] = 1u32;
        } else {
            let previous = Expr::eval7(
                slot0,
                slot1,
                slot2,
                slot3,
                slot4,
                slot5,
                slot6,
                slot_offsets,
                global - 1usize,
            );
            let current = Expr::eval7(
                slot0,
                slot1,
                slot2,
                slot3,
                slot4,
                slot5,
                slot6,
                slot_offsets,
                global,
            );
            flags[global] = if KeyEq::apply(previous, current) {
                0u32
            } else {
                1u32
            };
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_sorted_break_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    Less: BinaryPredicateOp<Input>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let current = Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global,
        );
        let next = Expr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot_offsets,
            global + 1,
        );
        flags[global] = if Less::apply(next, current) {
            1u32
        } else {
            0u32
        };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_minmax_flags_kernel<
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    Less: BinaryPredicateOp<Input>,
>(
    slot0: &[Leaf0],
    slot1: &[Leaf1],
    slot2: &[Leaf2],
    slot3: &[Leaf3],
    slot4: &[Leaf4],
    slot5: &[Leaf5],
    slot6: &[Leaf6],
    slot_offsets: &[u32],
    len: &[u32],
    min_flags: &mut [u32],
    max_flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let mut is_min = true;
        let mut is_max = true;
        let mut index = 0usize;
        while index < (len[0] as usize) {
            if Less::apply(
                Expr::eval7(
                    slot0,
                    slot1,
                    slot2,
                    slot3,
                    slot4,
                    slot5,
                    slot6,
                    slot_offsets,
                    index,
                ),
                Expr::eval7(
                    slot0,
                    slot1,
                    slot2,
                    slot3,
                    slot4,
                    slot5,
                    slot6,
                    slot_offsets,
                    global,
                ),
            ) {
                is_min = false;
            }
            if Less::apply(
                Expr::eval7(
                    slot0,
                    slot1,
                    slot2,
                    slot3,
                    slot4,
                    slot5,
                    slot6,
                    slot_offsets,
                    global,
                ),
                Expr::eval7(
                    slot0,
                    slot1,
                    slot2,
                    slot3,
                    slot4,
                    slot5,
                    slot6,
                    slot_offsets,
                    index,
                ),
            ) {
                is_max = false;
            }
            if index < global
                && !Less::apply(
                    Expr::eval7(
                        slot0,
                        slot1,
                        slot2,
                        slot3,
                        slot4,
                        slot5,
                        slot6,
                        slot_offsets,
                        global,
                    ),
                    Expr::eval7(
                        slot0,
                        slot1,
                        slot2,
                        slot3,
                        slot4,
                        slot5,
                        slot6,
                        slot_offsets,
                        index,
                    ),
                )
                && !Less::apply(
                    Expr::eval7(
                        slot0,
                        slot1,
                        slot2,
                        slot3,
                        slot4,
                        slot5,
                        slot6,
                        slot_offsets,
                        index,
                    ),
                    Expr::eval7(
                        slot0,
                        slot1,
                        slot2,
                        slot3,
                        slot4,
                        slot5,
                        slot6,
                        slot_offsets,
                        global,
                    ),
                )
            {
                is_min = false;
                is_max = false;
            }
            index += 1usize;
        }
        min_flags[global] = if is_min { 1u32 } else { 0u32 };
        max_flags[global] = if is_max { 1u32 } else { 0u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn logical7_lower_bound_many_kernel<
    Input: CubeType + 'static + Send + Sync,
    SourceLeaf0: CubePrimitive,
    SourceLeaf1: CubePrimitive,
    SourceLeaf2: CubePrimitive,
    SourceLeaf3: CubePrimitive,
    SourceLeaf4: CubePrimitive,
    SourceLeaf5: CubePrimitive,
    SourceLeaf6: CubePrimitive,
    ValueLeaf0: CubePrimitive,
    ValueLeaf1: CubePrimitive,
    ValueLeaf2: CubePrimitive,
    ValueLeaf3: CubePrimitive,
    ValueLeaf4: CubePrimitive,
    ValueLeaf5: CubePrimitive,
    ValueLeaf6: CubePrimitive,
    SourceExpr: LogicalDeviceExpr7<
            Input,
            SourceLeaf0,
            SourceLeaf1,
            SourceLeaf2,
            SourceLeaf3,
            SourceLeaf4,
            SourceLeaf5,
            SourceLeaf6,
        >,
    ValueExpr: LogicalDeviceExpr7<
            Input,
            ValueLeaf0,
            ValueLeaf1,
            ValueLeaf2,
            ValueLeaf3,
            ValueLeaf4,
            ValueLeaf5,
            ValueLeaf6,
        >,
    Less: BinaryPredicateOp<Input>,
>(
    source0: &[SourceLeaf0],
    source1: &[SourceLeaf1],
    source2: &[SourceLeaf2],
    source3: &[SourceLeaf3],
    source4: &[SourceLeaf4],
    source5: &[SourceLeaf5],
    source6: &[SourceLeaf6],
    source_offsets: &[u32],
    value0: &[ValueLeaf0],
    value1: &[ValueLeaf1],
    value2: &[ValueLeaf2],
    value3: &[ValueLeaf3],
    value4: &[ValueLeaf4],
    value5: &[ValueLeaf5],
    value6: &[ValueLeaf6],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let candidate = SourceExpr::eval7(
                source0,
                source1,
                source2,
                source3,
                source4,
                source5,
                source6,
                source_offsets,
                mid,
            );
            if Less::apply(
                candidate,
                ValueExpr::eval7(
                    value0,
                    value1,
                    value2,
                    value3,
                    value4,
                    value5,
                    value6,
                    value_offsets,
                    global,
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
pub(crate) fn logical7_upper_bound_many_kernel<
    Input: CubeType + 'static + Send + Sync,
    SourceLeaf0: CubePrimitive,
    SourceLeaf1: CubePrimitive,
    SourceLeaf2: CubePrimitive,
    SourceLeaf3: CubePrimitive,
    SourceLeaf4: CubePrimitive,
    SourceLeaf5: CubePrimitive,
    SourceLeaf6: CubePrimitive,
    ValueLeaf0: CubePrimitive,
    ValueLeaf1: CubePrimitive,
    ValueLeaf2: CubePrimitive,
    ValueLeaf3: CubePrimitive,
    ValueLeaf4: CubePrimitive,
    ValueLeaf5: CubePrimitive,
    ValueLeaf6: CubePrimitive,
    SourceExpr: LogicalDeviceExpr7<
            Input,
            SourceLeaf0,
            SourceLeaf1,
            SourceLeaf2,
            SourceLeaf3,
            SourceLeaf4,
            SourceLeaf5,
            SourceLeaf6,
        >,
    ValueExpr: LogicalDeviceExpr7<
            Input,
            ValueLeaf0,
            ValueLeaf1,
            ValueLeaf2,
            ValueLeaf3,
            ValueLeaf4,
            ValueLeaf5,
            ValueLeaf6,
        >,
    Less: BinaryPredicateOp<Input>,
>(
    source0: &[SourceLeaf0],
    source1: &[SourceLeaf1],
    source2: &[SourceLeaf2],
    source3: &[SourceLeaf3],
    source4: &[SourceLeaf4],
    source5: &[SourceLeaf5],
    source6: &[SourceLeaf6],
    source_offsets: &[u32],
    value0: &[ValueLeaf0],
    value1: &[ValueLeaf1],
    value2: &[ValueLeaf2],
    value3: &[ValueLeaf3],
    value4: &[ValueLeaf4],
    value5: &[ValueLeaf5],
    value6: &[ValueLeaf6],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let candidate = SourceExpr::eval7(
                source0,
                source1,
                source2,
                source3,
                source4,
                source5,
                source6,
                source_offsets,
                mid,
            );
            if !Less::apply(
                ValueExpr::eval7(
                    value0,
                    value1,
                    value2,
                    value3,
                    value4,
                    value5,
                    value6,
                    value_offsets,
                    global,
                ),
                candidate,
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

define_transform_tuple_to_tuple_kernel!(
    transform_tuple3_to_tuple5_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple3_to_tuple6_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple3_to_tuple7_kernel,
    (TyA: input_a: input_a_offset, TyB: input_b: input_b_offset, TyC: input_c: input_c_offset),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
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
    transform_tuple4_to_tuple2_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple4_to_tuple3_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple4_to_tuple4_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple4_to_tuple5_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple4_to_tuple6_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple4_to_tuple7_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
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
    transform_tuple5_to_tuple2_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple5_to_tuple3_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple5_to_tuple5_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple5_to_tuple4_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple5_to_tuple6_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple5_to_tuple7_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
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
    transform_tuple6_to_tuple2_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset, TyF: input_f: input_f_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple6_to_tuple3_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset, TyF: input_f: input_f_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple6_to_tuple6_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset, TyF: input_f: input_f_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple6_to_tuple4_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset, TyF: input_f: input_f_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple6_to_tuple5_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset, TyF: input_f: input_f_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_tuple_to_tuple_kernel!(
    transform_tuple6_to_tuple7_kernel,
    (
        TyA: input_a: input_a_offset, TyB: input_b: input_b_offset,
        TyC: input_c: input_c_offset, TyD: input_d: input_d_offset,
        TyE: input_e: input_e_offset, TyF: input_f: input_f_offset
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
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
        let selected = Pred::apply((
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
        ));
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
        let selected = Pred::apply((
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
        ));
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_predicate_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Pred: PredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    input_a: &[TyA],
    input_b: &[TyB],
    input_c: &[TyC],
    input_d: &[TyD],
    input_e: &[TyE],
    input_f: &[TyF],
    input_g: &[TyG],
    offsets: &[u32],
    len: &[u32],
    invert: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply((
            input_a[offsets[0] as usize + global],
            input_b[offsets[1] as usize + global],
            input_c[offsets[2] as usize + global],
            input_d[offsets[3] as usize + global],
            input_e[offsets[4] as usize + global],
            input_f[offsets[5] as usize + global],
            input_g[offsets[6] as usize + global],
        ));
        if (invert[0] == 0u32 && selected) || (invert[0] != 0u32 && !selected) {
            flags[global] = 1u32;
        } else {
            flags[global] = 0u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_mismatch_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Eq: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    left_a: &[TyA],
    left_b: &[TyB],
    left_c: &[TyC],
    left_d: &[TyD],
    left_e: &[TyE],
    left_f: &[TyF],
    left_g: &[TyG],
    left_offsets: &[u32],
    right_a: &[TyA],
    right_b: &[TyB],
    right_c: &[TyC],
    right_d: &[TyD],
    right_e: &[TyE],
    right_f: &[TyF],
    right_g: &[TyG],
    right_offsets: &[u32],
    len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let eq = Eq::apply(
            (
                left_a[left_offsets[0] as usize + global],
                left_b[left_offsets[1] as usize + global],
                left_c[left_offsets[2] as usize + global],
                left_d[left_offsets[3] as usize + global],
                left_e[left_offsets[4] as usize + global],
                left_f[left_offsets[5] as usize + global],
                left_g[left_offsets[6] as usize + global],
            ),
            (
                right_a[right_offsets[0] as usize + global],
                right_b[right_offsets[1] as usize + global],
                right_c[right_offsets[2] as usize + global],
                right_d[right_offsets[3] as usize + global],
                right_e[right_offsets[4] as usize + global],
                right_f[right_offsets[5] as usize + global],
                right_g[right_offsets[6] as usize + global],
            ),
        );
        if eq {
            flags[global] = 0u32;
        } else {
            flags[global] = 1u32;
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_adjacent_find_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Eq: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    input_a: &[TyA],
    input_b: &[TyB],
    input_c: &[TyC],
    input_d: &[TyD],
    input_e: &[TyE],
    input_f: &[TyF],
    input_g: &[TyG],
    offsets: &[u32],
    len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        if global + 1usize < len[0] as usize
            && Eq::apply(
                (
                    input_a[offsets[0] as usize + global],
                    input_b[offsets[1] as usize + global],
                    input_c[offsets[2] as usize + global],
                    input_d[offsets[3] as usize + global],
                    input_e[offsets[4] as usize + global],
                    input_f[offsets[5] as usize + global],
                    input_g[offsets[6] as usize + global],
                ),
                (
                    input_a[offsets[0] as usize + global + 1usize],
                    input_b[offsets[1] as usize + global + 1usize],
                    input_c[offsets[2] as usize + global + 1usize],
                    input_d[offsets[3] as usize + global + 1usize],
                    input_e[offsets[4] as usize + global + 1usize],
                    input_f[offsets[5] as usize + global + 1usize],
                    input_g[offsets[6] as usize + global + 1usize],
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
pub(crate) fn tuple7_view_find_first_of_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Eq: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    input_a: &[TyA],
    input_b: &[TyB],
    input_c: &[TyC],
    input_d: &[TyD],
    input_e: &[TyE],
    input_f: &[TyF],
    input_g: &[TyG],
    input_offsets: &[u32],
    needle_a: &[TyA],
    needle_b: &[TyB],
    needle_c: &[TyC],
    needle_d: &[TyD],
    needle_e: &[TyE],
    needle_f: &[TyF],
    needle_g: &[TyG],
    needle_offsets: &[u32],
    needle_len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let value = (
            input_a[input_offsets[0] as usize + global],
            input_b[input_offsets[1] as usize + global],
            input_c[input_offsets[2] as usize + global],
            input_d[input_offsets[3] as usize + global],
            input_e[input_offsets[4] as usize + global],
            input_f[input_offsets[5] as usize + global],
            input_g[input_offsets[6] as usize + global],
        );
        let needle = RuntimeCell::<usize>::new(0usize);
        let found = RuntimeCell::<u32>::new(0u32);
        while needle.read() < needle_len[0] as usize {
            if Eq::apply(
                value,
                (
                    needle_a[needle_offsets[0] as usize + needle.read()],
                    needle_b[needle_offsets[1] as usize + needle.read()],
                    needle_c[needle_offsets[2] as usize + needle.read()],
                    needle_d[needle_offsets[3] as usize + needle.read()],
                    needle_e[needle_offsets[4] as usize + needle.read()],
                    needle_f[needle_offsets[5] as usize + needle.read()],
                    needle_g[needle_offsets[6] as usize + needle.read()],
                ),
            ) {
                found.store(1u32);
            }
            needle.store(needle.read() + 1usize);
        }
        flags[global] = found.read();
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_lower_bound_many_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Less: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    source_a: &[TyA],
    source_b: &[TyB],
    source_c: &[TyC],
    source_d: &[TyD],
    source_e: &[TyE],
    source_f: &[TyF],
    source_g: &[TyG],
    source_offsets: &[u32],
    value_a: &[TyA],
    value_b: &[TyB],
    value_c: &[TyC],
    value_d: &[TyD],
    value_e: &[TyE],
    value_f: &[TyF],
    value_g: &[TyG],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let value = (
            value_a[value_offsets[0] as usize + global],
            value_b[value_offsets[1] as usize + global],
            value_c[value_offsets[2] as usize + global],
            value_d[value_offsets[3] as usize + global],
            value_e[value_offsets[4] as usize + global],
            value_f[value_offsets[5] as usize + global],
            value_g[value_offsets[6] as usize + global],
        );
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let candidate = (
                source_a[source_offsets[0] as usize + mid],
                source_b[source_offsets[1] as usize + mid],
                source_c[source_offsets[2] as usize + mid],
                source_d[source_offsets[3] as usize + mid],
                source_e[source_offsets[4] as usize + mid],
                source_f[source_offsets[5] as usize + mid],
                source_g[source_offsets[6] as usize + mid],
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
pub(crate) fn tuple7_view_upper_bound_many_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Less: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    source_a: &[TyA],
    source_b: &[TyB],
    source_c: &[TyC],
    source_d: &[TyD],
    source_e: &[TyE],
    source_f: &[TyF],
    source_g: &[TyG],
    source_offsets: &[u32],
    value_a: &[TyA],
    value_b: &[TyB],
    value_c: &[TyC],
    value_d: &[TyD],
    value_e: &[TyE],
    value_f: &[TyF],
    value_g: &[TyG],
    value_offsets: &[u32],
    source_len: &[u32],
    value_len: &[u32],
    output: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (value_len[0] as usize) {
        let value = (
            value_a[value_offsets[0] as usize + global],
            value_b[value_offsets[1] as usize + global],
            value_c[value_offsets[2] as usize + global],
            value_d[value_offsets[3] as usize + global],
            value_e[value_offsets[4] as usize + global],
            value_f[value_offsets[5] as usize + global],
            value_g[value_offsets[6] as usize + global],
        );
        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let candidate = (
                source_a[source_offsets[0] as usize + mid],
                source_b[source_offsets[1] as usize + mid],
                source_c[source_offsets[2] as usize + mid],
                source_d[source_offsets[3] as usize + mid],
                source_e[source_offsets[4] as usize + mid],
                source_f[source_offsets[5] as usize + mid],
                source_g[source_offsets[6] as usize + mid],
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

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_lexicographical_diff_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Less: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    left_a: &[TyA],
    left_b: &[TyB],
    left_c: &[TyC],
    left_d: &[TyD],
    left_e: &[TyE],
    left_f: &[TyF],
    left_g: &[TyG],
    left_offsets: &[u32],
    right_a: &[TyA],
    right_b: &[TyB],
    right_c: &[TyC],
    right_d: &[TyD],
    right_e: &[TyE],
    right_f: &[TyF],
    right_g: &[TyG],
    right_offsets: &[u32],
    len: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let lhs = (
            left_a[left_offsets[0] as usize + global],
            left_b[left_offsets[1] as usize + global],
            left_c[left_offsets[2] as usize + global],
            left_d[left_offsets[3] as usize + global],
            left_e[left_offsets[4] as usize + global],
            left_f[left_offsets[5] as usize + global],
            left_g[left_offsets[6] as usize + global],
        );
        let rhs = (
            right_a[right_offsets[0] as usize + global],
            right_b[right_offsets[1] as usize + global],
            right_c[right_offsets[2] as usize + global],
            right_d[right_offsets[3] as usize + global],
            right_e[right_offsets[4] as usize + global],
            right_f[right_offsets[5] as usize + global],
            right_g[right_offsets[6] as usize + global],
        );
        flags[global] = if Less::apply(lhs, rhs) || Less::apply(rhs, lhs) {
            1u32
        } else {
            0u32
        };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_lexicographical_compare_at_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Less: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    left_a: &[TyA],
    left_b: &[TyB],
    left_c: &[TyC],
    left_d: &[TyD],
    left_e: &[TyE],
    left_f: &[TyF],
    left_g: &[TyG],
    left_offsets: &[u32],
    right_a: &[TyA],
    right_b: &[TyB],
    right_c: &[TyC],
    right_d: &[TyD],
    right_e: &[TyE],
    right_f: &[TyF],
    right_g: &[TyG],
    right_offsets: &[u32],
    index: &[u32],
    output: &mut [u32],
) {
    if UNIT_POS == 0 {
        let i = index[0] as usize;
        let lhs = (
            left_a[left_offsets[0] as usize + i],
            left_b[left_offsets[1] as usize + i],
            left_c[left_offsets[2] as usize + i],
            left_d[left_offsets[3] as usize + i],
            left_e[left_offsets[4] as usize + i],
            left_f[left_offsets[5] as usize + i],
            left_g[left_offsets[6] as usize + i],
        );
        let rhs = (
            right_a[right_offsets[0] as usize + i],
            right_b[right_offsets[1] as usize + i],
            right_c[right_offsets[2] as usize + i],
            right_d[right_offsets[3] as usize + i],
            right_e[right_offsets[4] as usize + i],
            right_f[right_offsets[5] as usize + i],
            right_g[right_offsets[6] as usize + i],
        );
        output[0] = if Less::apply(lhs, rhs) { 1u32 } else { 0u32 };
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn tuple7_view_set_membership_flags_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    TyD: CubePrimitive,
    TyE: CubePrimitive,
    TyF: CubePrimitive,
    TyG: CubePrimitive,
    Less: BinaryPredicateOp<(TyA, TyB, TyC, TyD, TyE, TyF, TyG)>,
>(
    candidate_a: &[TyA],
    candidate_b: &[TyB],
    candidate_c: &[TyC],
    candidate_d: &[TyD],
    candidate_e: &[TyE],
    candidate_f: &[TyF],
    candidate_g: &[TyG],
    candidate_offsets: &[u32],
    source_a: &[TyA],
    source_b: &[TyB],
    source_c: &[TyC],
    source_d: &[TyD],
    source_e: &[TyE],
    source_f: &[TyF],
    source_g: &[TyG],
    source_offsets: &[u32],
    candidate_len: &[u32],
    source_len: &[u32],
    keep_intersection: &[u32],
    flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (candidate_len[0] as usize) {
        let value = (
            candidate_a[candidate_offsets[0] as usize + global],
            candidate_b[candidate_offsets[1] as usize + global],
            candidate_c[candidate_offsets[2] as usize + global],
            candidate_d[candidate_offsets[3] as usize + global],
            candidate_e[candidate_offsets[4] as usize + global],
            candidate_f[candidate_offsets[5] as usize + global],
            candidate_g[candidate_offsets[6] as usize + global],
        );

        let mut first = 0usize;
        let mut count = source_len[0] as usize;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let probe = (
                source_a[source_offsets[0] as usize + mid],
                source_b[source_offsets[1] as usize + mid],
                source_c[source_offsets[2] as usize + mid],
                source_d[source_offsets[3] as usize + mid],
                source_e[source_offsets[4] as usize + mid],
                source_f[source_offsets[5] as usize + mid],
                source_g[source_offsets[6] as usize + mid],
            );
            if Less::apply(probe, value) {
                first = mid + 1usize;
                count = count - step - 1usize;
            } else {
                count = step;
            }
        }

        let lower = first;
        first = lower;
        count = (source_len[0] as usize) - lower;
        while count > 0usize {
            let step = count / 2usize;
            let mid = first + step;
            let probe = (
                source_a[source_offsets[0] as usize + mid],
                source_b[source_offsets[1] as usize + mid],
                source_c[source_offsets[2] as usize + mid],
                source_d[source_offsets[3] as usize + mid],
                source_e[source_offsets[4] as usize + mid],
                source_f[source_offsets[5] as usize + mid],
                source_g[source_offsets[6] as usize + mid],
            );
            if !Less::apply(value, probe) {
                first = mid + 1usize;
                count = count - step - 1usize;
            } else {
                count = step;
            }
        }
        let source_count = first - lower;

        let mut rank = 0usize;
        let mut cursor = global;
        while cursor > 0usize {
            let prev = (
                candidate_a[candidate_offsets[0] as usize + cursor - 1usize],
                candidate_b[candidate_offsets[1] as usize + cursor - 1usize],
                candidate_c[candidate_offsets[2] as usize + cursor - 1usize],
                candidate_d[candidate_offsets[3] as usize + cursor - 1usize],
                candidate_e[candidate_offsets[4] as usize + cursor - 1usize],
                candidate_f[candidate_offsets[5] as usize + cursor - 1usize],
                candidate_g[candidate_offsets[6] as usize + cursor - 1usize],
            );
            if Less::apply(prev, value) || Less::apply(value, prev) {
                cursor = 0usize;
            } else {
                rank += 1usize;
                cursor -= 1usize;
            }
        }

        let keep = if keep_intersection[0] != 0u32 {
            rank < source_count
        } else {
            rank >= source_count
        };
        flags[global] = if keep { 1u32 } else { 0u32 };
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
define_tuple_mismatch_device_expr_flags_kernel!(
    tuple4_mismatch_device_expr_flags_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets)
);
define_tuple_mismatch_device_expr_flags_kernel!(
    tuple5_mismatch_device_expr_flags_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets,
     TyE: LeftEExpr / RightEExpr:
        left_e_slot0, left_e_slot1, left_e_slot2, left_e_slot3, left_e_offsets /
        right_e_slot0, right_e_slot1, right_e_slot2, right_e_slot3, right_e_offsets)
);
define_tuple_mismatch_device_expr_flags_kernel!(
    tuple6_mismatch_device_expr_flags_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets,
     TyE: LeftEExpr / RightEExpr:
        left_e_slot0, left_e_slot1, left_e_slot2, left_e_slot3, left_e_offsets /
        right_e_slot0, right_e_slot1, right_e_slot2, right_e_slot3, right_e_offsets,
     TyF: LeftFExpr / RightFExpr:
        left_f_slot0, left_f_slot1, left_f_slot2, left_f_slot3, left_f_offsets /
        right_f_slot0, right_f_slot1, right_f_slot2, right_f_slot3, right_f_offsets)
);
define_tuple_mismatch_device_expr_flags_kernel!(
    tuple7_mismatch_device_expr_flags_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets,
     TyE: LeftEExpr / RightEExpr:
        left_e_slot0, left_e_slot1, left_e_slot2, left_e_slot3, left_e_offsets /
        right_e_slot0, right_e_slot1, right_e_slot2, right_e_slot3, right_e_offsets,
     TyF: LeftFExpr / RightFExpr:
        left_f_slot0, left_f_slot1, left_f_slot2, left_f_slot3, left_f_offsets /
        right_f_slot0, right_f_slot1, right_f_slot2, right_f_slot3, right_f_offsets,
     TyG: LeftGExpr / RightGExpr:
        left_g_slot0, left_g_slot1, left_g_slot2, left_g_slot3, left_g_offsets /
        right_g_slot0, right_g_slot1, right_g_slot2, right_g_slot3, right_g_offsets)
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
define_tuple_find_first_of_device_expr_flags_kernel!(
    tuple4_find_first_of_device_expr_flags_kernel,
    (TyA: InputAExpr / NeedleAExpr:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets /
        needle_a_slot0, needle_a_slot1, needle_a_slot2, needle_a_slot3, needle_a_offsets,
     TyB: InputBExpr / NeedleBExpr:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets /
        needle_b_slot0, needle_b_slot1, needle_b_slot2, needle_b_slot3, needle_b_offsets,
     TyC: InputCExpr / NeedleCExpr:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets /
        needle_c_slot0, needle_c_slot1, needle_c_slot2, needle_c_slot3, needle_c_offsets,
     TyD: InputDExpr / NeedleDExpr:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets /
        needle_d_slot0, needle_d_slot1, needle_d_slot2, needle_d_slot3, needle_d_offsets)
);
define_tuple_find_first_of_device_expr_flags_kernel!(
    tuple5_find_first_of_device_expr_flags_kernel,
    (TyA: InputAExpr / NeedleAExpr:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets /
        needle_a_slot0, needle_a_slot1, needle_a_slot2, needle_a_slot3, needle_a_offsets,
     TyB: InputBExpr / NeedleBExpr:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets /
        needle_b_slot0, needle_b_slot1, needle_b_slot2, needle_b_slot3, needle_b_offsets,
     TyC: InputCExpr / NeedleCExpr:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets /
        needle_c_slot0, needle_c_slot1, needle_c_slot2, needle_c_slot3, needle_c_offsets,
     TyD: InputDExpr / NeedleDExpr:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets /
        needle_d_slot0, needle_d_slot1, needle_d_slot2, needle_d_slot3, needle_d_offsets,
     TyE: InputEExpr / NeedleEExpr:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets /
        needle_e_slot0, needle_e_slot1, needle_e_slot2, needle_e_slot3, needle_e_offsets)
);
define_tuple_find_first_of_device_expr_flags_kernel!(
    tuple6_find_first_of_device_expr_flags_kernel,
    (TyA: InputAExpr / NeedleAExpr:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets /
        needle_a_slot0, needle_a_slot1, needle_a_slot2, needle_a_slot3, needle_a_offsets,
     TyB: InputBExpr / NeedleBExpr:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets /
        needle_b_slot0, needle_b_slot1, needle_b_slot2, needle_b_slot3, needle_b_offsets,
     TyC: InputCExpr / NeedleCExpr:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets /
        needle_c_slot0, needle_c_slot1, needle_c_slot2, needle_c_slot3, needle_c_offsets,
     TyD: InputDExpr / NeedleDExpr:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets /
        needle_d_slot0, needle_d_slot1, needle_d_slot2, needle_d_slot3, needle_d_offsets,
     TyE: InputEExpr / NeedleEExpr:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets /
        needle_e_slot0, needle_e_slot1, needle_e_slot2, needle_e_slot3, needle_e_offsets,
     TyF: InputFExpr / NeedleFExpr:
        input_f_slot0, input_f_slot1, input_f_slot2, input_f_slot3, input_f_offsets /
        needle_f_slot0, needle_f_slot1, needle_f_slot2, needle_f_slot3, needle_f_offsets)
);
define_tuple_find_first_of_device_expr_flags_kernel!(
    tuple7_find_first_of_device_expr_flags_kernel,
    (TyA: InputAExpr / NeedleAExpr:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets /
        needle_a_slot0, needle_a_slot1, needle_a_slot2, needle_a_slot3, needle_a_offsets,
     TyB: InputBExpr / NeedleBExpr:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets /
        needle_b_slot0, needle_b_slot1, needle_b_slot2, needle_b_slot3, needle_b_offsets,
     TyC: InputCExpr / NeedleCExpr:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets /
        needle_c_slot0, needle_c_slot1, needle_c_slot2, needle_c_slot3, needle_c_offsets,
     TyD: InputDExpr / NeedleDExpr:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets /
        needle_d_slot0, needle_d_slot1, needle_d_slot2, needle_d_slot3, needle_d_offsets,
     TyE: InputEExpr / NeedleEExpr:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets /
        needle_e_slot0, needle_e_slot1, needle_e_slot2, needle_e_slot3, needle_e_offsets,
     TyF: InputFExpr / NeedleFExpr:
        input_f_slot0, input_f_slot1, input_f_slot2, input_f_slot3, input_f_offsets /
        needle_f_slot0, needle_f_slot1, needle_f_slot2, needle_f_slot3, needle_f_offsets,
     TyG: InputGExpr / NeedleGExpr:
        input_g_slot0, input_g_slot1, input_g_slot2, input_g_slot3, input_g_offsets /
        needle_g_slot0, needle_g_slot1, needle_g_slot2, needle_g_slot3, needle_g_offsets)
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
define_tuple_search_device_expr_kernels!(
    tuple4_adjacent_device_expr_flags_kernel,
    tuple4_sorted_break_device_expr_flags_kernel,
    tuple4_lower_bound_device_expr_flags_kernel,
    tuple4_upper_bound_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets / value_a,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets / value_b,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets / value_c,
     TyD: ExprD:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets / value_d)
);
define_tuple_search_device_expr_kernels!(
    tuple5_adjacent_device_expr_flags_kernel,
    tuple5_sorted_break_device_expr_flags_kernel,
    tuple5_lower_bound_device_expr_flags_kernel,
    tuple5_upper_bound_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets / value_a,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets / value_b,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets / value_c,
     TyD: ExprD:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets / value_d,
     TyE: ExprE:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets / value_e)
);
define_tuple_search_device_expr_kernels!(
    tuple6_adjacent_device_expr_flags_kernel,
    tuple6_sorted_break_device_expr_flags_kernel,
    tuple6_lower_bound_device_expr_flags_kernel,
    tuple6_upper_bound_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets / value_a,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets / value_b,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets / value_c,
     TyD: ExprD:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets / value_d,
     TyE: ExprE:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets / value_e,
     TyF: ExprF:
        input_f_slot0, input_f_slot1, input_f_slot2, input_f_slot3, input_f_offsets / value_f)
);
define_tuple_search_device_expr_kernels!(
    tuple7_adjacent_device_expr_flags_kernel,
    tuple7_sorted_break_device_expr_flags_kernel,
    tuple7_lower_bound_device_expr_flags_kernel,
    tuple7_upper_bound_device_expr_flags_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets / value_a,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets / value_b,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets / value_c,
     TyD: ExprD:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets / value_d,
     TyE: ExprE:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets / value_e,
     TyF: ExprF:
        input_f_slot0, input_f_slot1, input_f_slot2, input_f_slot3, input_f_offsets / value_f,
     TyG: ExprG:
        input_g_slot0, input_g_slot1, input_g_slot2, input_g_slot3, input_g_offsets / value_g)
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
define_tuple_bound_many_device_expr_kernels!(
    tuple4_lower_bound_device_expr_many_kernel,
    tuple4_upper_bound_device_expr_many_kernel,
    (TyA: SourceExprA / ValueExprA:
        source_a_slot0, source_a_slot1, source_a_slot2, source_a_slot3, source_a_offsets /
        value_a_slot0, value_a_slot1, value_a_slot2, value_a_slot3, value_a_offsets,
     TyB: SourceExprB / ValueExprB:
        source_b_slot0, source_b_slot1, source_b_slot2, source_b_slot3, source_b_offsets /
        value_b_slot0, value_b_slot1, value_b_slot2, value_b_slot3, value_b_offsets,
     TyC: SourceExprC / ValueExprC:
        source_c_slot0, source_c_slot1, source_c_slot2, source_c_slot3, source_c_offsets /
        value_c_slot0, value_c_slot1, value_c_slot2, value_c_slot3, value_c_offsets,
     TyD: SourceExprD / ValueExprD:
        source_d_slot0, source_d_slot1, source_d_slot2, source_d_slot3, source_d_offsets /
        value_d_slot0, value_d_slot1, value_d_slot2, value_d_slot3, value_d_offsets)
);
define_tuple_bound_many_device_expr_kernels!(
    tuple5_lower_bound_device_expr_many_kernel,
    tuple5_upper_bound_device_expr_many_kernel,
    (TyA: SourceExprA / ValueExprA:
        source_a_slot0, source_a_slot1, source_a_slot2, source_a_slot3, source_a_offsets /
        value_a_slot0, value_a_slot1, value_a_slot2, value_a_slot3, value_a_offsets,
     TyB: SourceExprB / ValueExprB:
        source_b_slot0, source_b_slot1, source_b_slot2, source_b_slot3, source_b_offsets /
        value_b_slot0, value_b_slot1, value_b_slot2, value_b_slot3, value_b_offsets,
     TyC: SourceExprC / ValueExprC:
        source_c_slot0, source_c_slot1, source_c_slot2, source_c_slot3, source_c_offsets /
        value_c_slot0, value_c_slot1, value_c_slot2, value_c_slot3, value_c_offsets,
     TyD: SourceExprD / ValueExprD:
        source_d_slot0, source_d_slot1, source_d_slot2, source_d_slot3, source_d_offsets /
        value_d_slot0, value_d_slot1, value_d_slot2, value_d_slot3, value_d_offsets,
     TyE: SourceExprE / ValueExprE:
        source_e_slot0, source_e_slot1, source_e_slot2, source_e_slot3, source_e_offsets /
        value_e_slot0, value_e_slot1, value_e_slot2, value_e_slot3, value_e_offsets)
);
define_tuple_bound_many_device_expr_kernels!(
    tuple6_lower_bound_device_expr_many_kernel,
    tuple6_upper_bound_device_expr_many_kernel,
    (TyA: SourceExprA / ValueExprA:
        source_a_slot0, source_a_slot1, source_a_slot2, source_a_slot3, source_a_offsets /
        value_a_slot0, value_a_slot1, value_a_slot2, value_a_slot3, value_a_offsets,
     TyB: SourceExprB / ValueExprB:
        source_b_slot0, source_b_slot1, source_b_slot2, source_b_slot3, source_b_offsets /
        value_b_slot0, value_b_slot1, value_b_slot2, value_b_slot3, value_b_offsets,
     TyC: SourceExprC / ValueExprC:
        source_c_slot0, source_c_slot1, source_c_slot2, source_c_slot3, source_c_offsets /
        value_c_slot0, value_c_slot1, value_c_slot2, value_c_slot3, value_c_offsets,
     TyD: SourceExprD / ValueExprD:
        source_d_slot0, source_d_slot1, source_d_slot2, source_d_slot3, source_d_offsets /
        value_d_slot0, value_d_slot1, value_d_slot2, value_d_slot3, value_d_offsets,
     TyE: SourceExprE / ValueExprE:
        source_e_slot0, source_e_slot1, source_e_slot2, source_e_slot3, source_e_offsets /
        value_e_slot0, value_e_slot1, value_e_slot2, value_e_slot3, value_e_offsets,
     TyF: SourceExprF / ValueExprF:
        source_f_slot0, source_f_slot1, source_f_slot2, source_f_slot3, source_f_offsets /
        value_f_slot0, value_f_slot1, value_f_slot2, value_f_slot3, value_f_offsets)
);
define_tuple_bound_many_device_expr_kernels!(
    tuple7_lower_bound_device_expr_many_kernel,
    tuple7_upper_bound_device_expr_many_kernel,
    (TyA: SourceExprA / ValueExprA:
        source_a_slot0, source_a_slot1, source_a_slot2, source_a_slot3, source_a_offsets /
        value_a_slot0, value_a_slot1, value_a_slot2, value_a_slot3, value_a_offsets,
     TyB: SourceExprB / ValueExprB:
        source_b_slot0, source_b_slot1, source_b_slot2, source_b_slot3, source_b_offsets /
        value_b_slot0, value_b_slot1, value_b_slot2, value_b_slot3, value_b_offsets,
     TyC: SourceExprC / ValueExprC:
        source_c_slot0, source_c_slot1, source_c_slot2, source_c_slot3, source_c_offsets /
        value_c_slot0, value_c_slot1, value_c_slot2, value_c_slot3, value_c_offsets,
     TyD: SourceExprD / ValueExprD:
        source_d_slot0, source_d_slot1, source_d_slot2, source_d_slot3, source_d_offsets /
        value_d_slot0, value_d_slot1, value_d_slot2, value_d_slot3, value_d_offsets,
     TyE: SourceExprE / ValueExprE:
        source_e_slot0, source_e_slot1, source_e_slot2, source_e_slot3, source_e_offsets /
        value_e_slot0, value_e_slot1, value_e_slot2, value_e_slot3, value_e_offsets,
     TyF: SourceExprF / ValueExprF:
        source_f_slot0, source_f_slot1, source_f_slot2, source_f_slot3, source_f_offsets /
        value_f_slot0, value_f_slot1, value_f_slot2, value_f_slot3, value_f_offsets,
     TyG: SourceExprG / ValueExprG:
        source_g_slot0, source_g_slot1, source_g_slot2, source_g_slot3, source_g_offsets /
        value_g_slot0, value_g_slot1, value_g_slot2, value_g_slot3, value_g_offsets)
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
define_tuple_membership_device_expr_flags_kernel!(
    tuple4_membership_device_expr_flags_kernel,
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
        sorted_c_slot0, sorted_c_slot1, sorted_c_slot2, sorted_c_slot3, sorted_c_offsets,
     TyD: ExprD:
        candidate_d_slot0, candidate_d_slot1, candidate_d_slot2, candidate_d_slot3, candidate_d_offsets /
        SortedExprD:
        sorted_d_slot0, sorted_d_slot1, sorted_d_slot2, sorted_d_slot3, sorted_d_offsets)
);
define_tuple_membership_device_expr_flags_kernel!(
    tuple5_membership_device_expr_flags_kernel,
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
        sorted_c_slot0, sorted_c_slot1, sorted_c_slot2, sorted_c_slot3, sorted_c_offsets,
     TyD: ExprD:
        candidate_d_slot0, candidate_d_slot1, candidate_d_slot2, candidate_d_slot3, candidate_d_offsets /
        SortedExprD:
        sorted_d_slot0, sorted_d_slot1, sorted_d_slot2, sorted_d_slot3, sorted_d_offsets,
     TyE: ExprE:
        candidate_e_slot0, candidate_e_slot1, candidate_e_slot2, candidate_e_slot3, candidate_e_offsets /
        SortedExprE:
        sorted_e_slot0, sorted_e_slot1, sorted_e_slot2, sorted_e_slot3, sorted_e_offsets)
);
define_tuple_membership_device_expr_flags_kernel!(
    tuple6_membership_device_expr_flags_kernel,
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
        sorted_c_slot0, sorted_c_slot1, sorted_c_slot2, sorted_c_slot3, sorted_c_offsets,
     TyD: ExprD:
        candidate_d_slot0, candidate_d_slot1, candidate_d_slot2, candidate_d_slot3, candidate_d_offsets /
        SortedExprD:
        sorted_d_slot0, sorted_d_slot1, sorted_d_slot2, sorted_d_slot3, sorted_d_offsets,
     TyE: ExprE:
        candidate_e_slot0, candidate_e_slot1, candidate_e_slot2, candidate_e_slot3, candidate_e_offsets /
        SortedExprE:
        sorted_e_slot0, sorted_e_slot1, sorted_e_slot2, sorted_e_slot3, sorted_e_offsets,
     TyF: ExprF:
        candidate_f_slot0, candidate_f_slot1, candidate_f_slot2, candidate_f_slot3, candidate_f_offsets /
        SortedExprF:
        sorted_f_slot0, sorted_f_slot1, sorted_f_slot2, sorted_f_slot3, sorted_f_offsets)
);
define_tuple_membership_device_expr_flags_kernel!(
    tuple7_membership_device_expr_flags_kernel,
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
        sorted_c_slot0, sorted_c_slot1, sorted_c_slot2, sorted_c_slot3, sorted_c_offsets,
     TyD: ExprD:
        candidate_d_slot0, candidate_d_slot1, candidate_d_slot2, candidate_d_slot3, candidate_d_offsets /
        SortedExprD:
        sorted_d_slot0, sorted_d_slot1, sorted_d_slot2, sorted_d_slot3, sorted_d_offsets,
     TyE: ExprE:
        candidate_e_slot0, candidate_e_slot1, candidate_e_slot2, candidate_e_slot3, candidate_e_offsets /
        SortedExprE:
        sorted_e_slot0, sorted_e_slot1, sorted_e_slot2, sorted_e_slot3, sorted_e_offsets,
     TyF: ExprF:
        candidate_f_slot0, candidate_f_slot1, candidate_f_slot2, candidate_f_slot3, candidate_f_offsets /
        SortedExprF:
        sorted_f_slot0, sorted_f_slot1, sorted_f_slot2, sorted_f_slot3, sorted_f_offsets,
     TyG: ExprG:
        candidate_g_slot0, candidate_g_slot1, candidate_g_slot2, candidate_g_slot3, candidate_g_offsets /
        SortedExprG:
        sorted_g_slot0, sorted_g_slot1, sorted_g_slot2, sorted_g_slot3, sorted_g_offsets)
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
define_tuple_minmax_device_expr_kernels!(
    tuple4_minmax_element_device_expr_partials_kernel,
    tuple4_minmax_index_device_expr_partials_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets,
     TyD: ExprD:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets)
);
define_tuple_minmax_device_expr_kernels!(
    tuple5_minmax_element_device_expr_partials_kernel,
    tuple5_minmax_index_device_expr_partials_kernel,
    (TyA: ExprA:
        input_a_slot0, input_a_slot1, input_a_slot2, input_a_slot3, input_a_offsets,
     TyB: ExprB:
        input_b_slot0, input_b_slot1, input_b_slot2, input_b_slot3, input_b_offsets,
     TyC: ExprC:
        input_c_slot0, input_c_slot1, input_c_slot2, input_c_slot3, input_c_offsets,
     TyD: ExprD:
        input_d_slot0, input_d_slot1, input_d_slot2, input_d_slot3, input_d_offsets,
     TyE: ExprE:
        input_e_slot0, input_e_slot1, input_e_slot2, input_e_slot3, input_e_offsets)
);
define_tuple_minmax_device_expr_kernels!(
    tuple6_minmax_element_device_expr_partials_kernel,
    tuple6_minmax_index_device_expr_partials_kernel,
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
        input_f_slot0, input_f_slot1, input_f_slot2, input_f_slot3, input_f_offsets)
);
define_tuple_minmax_device_expr_kernels!(
    tuple7_minmax_element_device_expr_partials_kernel,
    tuple7_minmax_index_device_expr_partials_kernel,
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
define_tuple_lexicographical_device_expr_kernels!(
    tuple4_lexicographical_diff_device_expr_flags_kernel,
    tuple4_lexicographical_compare_at_device_expr_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets)
);
define_tuple_lexicographical_device_expr_kernels!(
    tuple5_lexicographical_diff_device_expr_flags_kernel,
    tuple5_lexicographical_compare_at_device_expr_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets,
     TyE: LeftEExpr / RightEExpr:
        left_e_slot0, left_e_slot1, left_e_slot2, left_e_slot3, left_e_offsets /
        right_e_slot0, right_e_slot1, right_e_slot2, right_e_slot3, right_e_offsets)
);
define_tuple_lexicographical_device_expr_kernels!(
    tuple6_lexicographical_diff_device_expr_flags_kernel,
    tuple6_lexicographical_compare_at_device_expr_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets,
     TyE: LeftEExpr / RightEExpr:
        left_e_slot0, left_e_slot1, left_e_slot2, left_e_slot3, left_e_offsets /
        right_e_slot0, right_e_slot1, right_e_slot2, right_e_slot3, right_e_offsets,
     TyF: LeftFExpr / RightFExpr:
        left_f_slot0, left_f_slot1, left_f_slot2, left_f_slot3, left_f_offsets /
        right_f_slot0, right_f_slot1, right_f_slot2, right_f_slot3, right_f_offsets)
);
define_tuple_lexicographical_device_expr_kernels!(
    tuple7_lexicographical_diff_device_expr_flags_kernel,
    tuple7_lexicographical_compare_at_device_expr_kernel,
    (TyA: LeftAExpr / RightAExpr:
        left_a_slot0, left_a_slot1, left_a_slot2, left_a_slot3, left_a_offsets /
        right_a_slot0, right_a_slot1, right_a_slot2, right_a_slot3, right_a_offsets,
     TyB: LeftBExpr / RightBExpr:
        left_b_slot0, left_b_slot1, left_b_slot2, left_b_slot3, left_b_offsets /
        right_b_slot0, right_b_slot1, right_b_slot2, right_b_slot3, right_b_offsets,
     TyC: LeftCExpr / RightCExpr:
        left_c_slot0, left_c_slot1, left_c_slot2, left_c_slot3, left_c_offsets /
        right_c_slot0, right_c_slot1, right_c_slot2, right_c_slot3, right_c_offsets,
     TyD: LeftDExpr / RightDExpr:
        left_d_slot0, left_d_slot1, left_d_slot2, left_d_slot3, left_d_offsets /
        right_d_slot0, right_d_slot1, right_d_slot2, right_d_slot3, right_d_offsets,
     TyE: LeftEExpr / RightEExpr:
        left_e_slot0, left_e_slot1, left_e_slot2, left_e_slot3, left_e_offsets /
        right_e_slot0, right_e_slot1, right_e_slot2, right_e_slot3, right_e_offsets,
     TyF: LeftFExpr / RightFExpr:
        left_f_slot0, left_f_slot1, left_f_slot2, left_f_slot3, left_f_offsets /
        right_f_slot0, right_f_slot1, right_f_slot2, right_f_slot3, right_f_offsets,
     TyG: LeftGExpr / RightGExpr:
        left_g_slot0, left_g_slot1, left_g_slot2, left_g_slot3, left_g_offsets /
        right_g_slot0, right_g_slot1, right_g_slot2, right_g_slot3, right_g_offsets)
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn copy_if_expr_flag_only_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp<T>,
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
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let selected = Pred::apply(Expr::eval(input, indices, rhs, rhs_indices, global));
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
        let selected = Pred::apply(Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global));
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
        let selected = Pred::apply(stencil);
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
        if Pred::apply(stencil) {
            let value = ValueExpr::eval(
                value_input,
                value_indices,
                value_rhs,
                value_rhs_indices,
                global,
            );
            output[global] = Op::apply(value);
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
    output_offset: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let position = positions[global];
        output[output_offset[0] as usize + (position - 1u32) as usize] =
            Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
    }
}

macro_rules! define_selected_apply_tuple_device_expr_kernel {
    (
        $name:ident,
        ($first_ty:ident: $first_expr:ident:
            $first_slot0:ident, $first_slot1:ident, $first_slot2:ident, $first_slot3:ident,
            $first_offsets:ident => $first_output:ident
        $(,
            $ty:ident: $expr:ident:
                $slot0:ident, $slot1:ident, $slot2:ident, $slot3:ident, $offsets:ident => $output:ident
        )* $(,)?)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $name<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
        >(
            flags: &[u32],
            positions: &[u32],
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
            $first_output: &mut [$first_ty],
            $(
                $output: &mut [$ty],
            )*
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() && flags[global] != 0u32 {
                let position = (positions[global] - 1u32) as usize;
                $first_output[position] = $first_expr::eval(
                    $first_slot0,
                    $first_slot1,
                    $first_slot2,
                    $first_slot3,
                    $first_offsets,
                    global,
                );
                $(
                    $output[position] = $expr::eval(
                        $slot0,
                        $slot1,
                        $slot2,
                        $slot3,
                        $offsets,
                        global,
                    );
                )*
            }
        }
    };
}

define_selected_apply_tuple_device_expr_kernel!(
    selected_apply_tuple2_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => output_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => output_b)
);
define_selected_apply_tuple_device_expr_kernel!(
    selected_apply_tuple3_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => output_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => output_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => output_c)
);
define_selected_apply_tuple_device_expr_kernel!(
    selected_apply_tuple4_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => output_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => output_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => output_c,
     D: ExprD: d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets => output_d)
);
define_selected_apply_tuple_device_expr_kernel!(
    selected_apply_tuple5_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => output_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => output_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => output_c,
     D: ExprD: d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets => output_d,
     E: ExprE: e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets => output_e)
);
define_selected_apply_tuple_device_expr_kernel!(
    selected_apply_tuple6_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => output_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => output_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => output_c,
     D: ExprD: d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets => output_d,
     E: ExprE: e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets => output_e,
     F: ExprF: f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets => output_f)
);
define_selected_apply_tuple_device_expr_kernel!(
    selected_apply_tuple7_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => output_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => output_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => output_c,
     D: ExprD: d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets => output_d,
     E: ExprE: e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets => output_e,
     F: ExprF: f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets => output_f,
     G: ExprG: g_slot0, g_slot1, g_slot2, g_slot3, g_slot_offsets => output_g)
);

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
pub(crate) fn compact_split_scatter_device_expr_kernel<T: CubePrimitive, Expr: DeviceGpuExpr<T>>(
    flags: &[u32],
    positions: &[u32],
    slot0: &[T],
    slot1: &[T],
    slot2: &[T],
    slot3: &[T],
    slot_offsets: &[u32],
    selected_output: &mut [T],
    rejected_output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        let value = Expr::eval(slot0, slot1, slot2, slot3, slot_offsets, global);
        if flags[global] != 0u32 {
            let selected_position = positions[global] - 1u32;
            selected_output[selected_position as usize] = value;
        } else {
            let selected_before_or_at = positions[global];
            let rejected_before = (global as u32) - selected_before_or_at;
            rejected_output[rejected_before as usize] = value;
        }
    }
}

macro_rules! define_split_apply_tuple_device_expr_kernel {
    (
        $name:ident,
        ($first_ty:ident: $first_expr:ident:
            $first_slot0:ident, $first_slot1:ident, $first_slot2:ident, $first_slot3:ident,
            $first_offsets:ident => $first_selected:ident, $first_rejected:ident
        $(,
            $ty:ident: $expr:ident:
                $slot0:ident, $slot1:ident, $slot2:ident, $slot3:ident, $offsets:ident
                => $selected:ident, $rejected:ident
        )* $(,)?)
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $name<
            $first_ty: CubePrimitive,
            $( $ty: CubePrimitive, )*
            $first_expr: DeviceGpuExpr<$first_ty>,
            $( $expr: DeviceGpuExpr<$ty>, )*
        >(
            flags: &[u32],
            positions: &[u32],
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
            $first_selected: &mut [$first_ty],
            $first_rejected: &mut [$first_ty],
            $(
                $selected: &mut [$ty],
                $rejected: &mut [$ty],
            )*
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let first_value = $first_expr::eval(
                    $first_slot0,
                    $first_slot1,
                    $first_slot2,
                    $first_slot3,
                    $first_offsets,
                    global,
                );
                if flags[global] != 0u32 {
                    let selected_position = (positions[global] - 1u32) as usize;
                    $first_selected[selected_position] = first_value;
                    $(
                        $selected[selected_position] = $expr::eval(
                            $slot0,
                            $slot1,
                            $slot2,
                            $slot3,
                            $offsets,
                            global,
                        );
                    )*
                } else {
                    let selected_before_or_at = positions[global];
                    let rejected_before = ((global as u32) - selected_before_or_at) as usize;
                    $first_rejected[rejected_before] = first_value;
                    $(
                        $rejected[rejected_before] = $expr::eval(
                            $slot0,
                            $slot1,
                            $slot2,
                            $slot3,
                            $offsets,
                            global,
                        );
                    )*
                }
            }
        }
    };
}

define_split_apply_tuple_device_expr_kernel!(
    split_apply_tuple2_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => selected_a, rejected_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => selected_b, rejected_b)
);
define_split_apply_tuple_device_expr_kernel!(
    split_apply_tuple3_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => selected_a, rejected_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => selected_b, rejected_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => selected_c, rejected_c)
);
define_split_apply_tuple_device_expr_kernel!(
    split_apply_tuple4_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => selected_a, rejected_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => selected_b, rejected_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => selected_c, rejected_c,
     D: ExprD: d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets => selected_d, rejected_d)
);
define_split_apply_tuple_device_expr_kernel!(
    split_apply_tuple5_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => selected_a, rejected_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => selected_b, rejected_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => selected_c, rejected_c,
     D: ExprD: d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets => selected_d, rejected_d,
     E: ExprE: e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets => selected_e, rejected_e)
);
define_split_apply_tuple_device_expr_kernel!(
    split_apply_tuple6_device_expr_kernel,
    (A: ExprA: a_slot0, a_slot1, a_slot2, a_slot3, a_slot_offsets => selected_a, rejected_a,
     B: ExprB: b_slot0, b_slot1, b_slot2, b_slot3, b_slot_offsets => selected_b, rejected_b,
     C: ExprC: c_slot0, c_slot1, c_slot2, c_slot3, c_slot_offsets => selected_c, rejected_c,
     D: ExprD: d_slot0, d_slot1, d_slot2, d_slot3, d_slot_offsets => selected_d, rejected_d,
     E: ExprE: e_slot0, e_slot1, e_slot2, e_slot3, e_slot_offsets => selected_e, rejected_e,
     F: ExprF: f_slot0, f_slot1, f_slot2, f_slot3, f_slot_offsets => selected_f, rejected_f)
);

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
pub(crate) fn tuple7_view_adjacent_difference_kernel<
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
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let current = (
            input_a[offsets[0] as usize + global],
            input_b[offsets[1] as usize + global],
            input_c[offsets[2] as usize + global],
            input_d[offsets[3] as usize + global],
            input_e[offsets[4] as usize + global],
            input_f[offsets[5] as usize + global],
            input_g[offsets[6] as usize + global],
        );
        let output = if global == 0usize {
            current
        } else {
            Op::apply(
                current,
                (
                    input_a[offsets[0] as usize + global - 1usize],
                    input_b[offsets[1] as usize + global - 1usize],
                    input_c[offsets[2] as usize + global - 1usize],
                    input_d[offsets[3] as usize + global - 1usize],
                    input_e[offsets[4] as usize + global - 1usize],
                    input_f[offsets[5] as usize + global - 1usize],
                    input_g[offsets[6] as usize + global - 1usize],
                ),
            )
        };
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
        output_d[global] = output.3;
        output_e[global] = output.4;
        output_f[global] = output.5;
        output_g[global] = output.6;
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
    Pred: PredicateOp<T>,
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
