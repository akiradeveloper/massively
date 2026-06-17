use crate::{
    expr::{DeviceGpuExpr, GpuExpr},
    op::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp},
};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn collect_expr_block_kernel<T: CubePrimitive, Expr: GpuExpr<T>>(
    output: &mut Array<T>,
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        output[global] = Expr::eval(input, indices, rhs, rhs_indices, global);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn device_collect_expr_block_kernel<T: CubePrimitive, Expr: DeviceGpuExpr<T>>(
    output: &mut Array<T>,
    slot0: &Array<T>,
    slot1: &Array<T>,
    slot2: &Array<T>,
    slot3: &Array<T>,
    len: &Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        output[global] = Expr::eval(slot0, slot1, slot2, slot3, global);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn transform_unary_kernel<
    T: CubePrimitive,
    Out: CubePrimitive,
    Op: UnaryOp<T, Output = Out>,
>(
    input: &Array<T>,
    len: &Array<u32>,
    output: &mut Array<Out>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        output[global] = Op::apply(input[global]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn transform_unary_tuple2_kernel<
    T: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    Op: UnaryOp<T, Output = (A, B)>,
>(
    input: &Array<T>,
    len: &Array<u32>,
    output_a: &mut Array<A>,
    output_b: &mut Array<B>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(input[global]);
        output_a[global] = output.0;
        output_b[global] = output.1;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn transform_unary_tuple3_kernel<
    T: CubePrimitive,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    Op: UnaryOp<T, Output = (A, B, C)>,
>(
    input: &Array<T>,
    len: &Array<u32>,
    output_a: &mut Array<A>,
    output_b: &mut Array<B>,
    output_c: &mut Array<C>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply(input[global]);
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
    }
}

macro_rules! define_transform_tuple_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Out: CubePrimitive, Op: UnaryOp<($( $ty, )+), Output = Out>>(
            $( $input: &Array<$ty>, )+
            len: &Array<u32>,
            output: &mut Array<Out>,
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (len[0] as usize) {
                output[global] = Op::apply((
                    $( $input[global], )+
                ));
            }
        }
    };
}

define_transform_tuple_kernel!(transform_tuple2_kernel, (TyA: input_a, TyB: input_b));
define_transform_tuple_kernel!(transform_tuple3_kernel, (TyA: input_a, TyB: input_b, TyC: input_c));

#[cube(launch_unchecked)]
pub(crate) fn transform_tuple2_to_tuple2_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    Op: UnaryOp<(TyA, TyB), Output = (OutA, OutB)>,
>(
    input_a: &Array<TyA>,
    input_b: &Array<TyB>,
    len: &Array<u32>,
    output_a: &mut Array<OutA>,
    output_b: &mut Array<OutB>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply((input_a[global], input_b[global]));
        output_a[global] = output.0;
        output_b[global] = output.1;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn transform_tuple3_to_tuple3_kernel<
    TyA: CubePrimitive,
    TyB: CubePrimitive,
    TyC: CubePrimitive,
    OutA: CubePrimitive,
    OutB: CubePrimitive,
    OutC: CubePrimitive,
    Op: UnaryOp<(TyA, TyB, TyC), Output = (OutA, OutB, OutC)>,
>(
    input_a: &Array<TyA>,
    input_b: &Array<TyB>,
    input_c: &Array<TyC>,
    len: &Array<u32>,
    output_a: &mut Array<OutA>,
    output_b: &mut Array<OutB>,
    output_c: &mut Array<OutC>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        let output = Op::apply((input_a[global], input_b[global], input_c[global]));
        output_a[global] = output.0;
        output_b[global] = output.1;
        output_c[global] = output.2;
    }
}

define_transform_tuple_kernel!(
    transform_tuple4_kernel,
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d)
);
define_transform_tuple_kernel!(
    transform_tuple5_kernel,
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e)
);
define_transform_tuple_kernel!(
    transform_tuple6_kernel,
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f)
);
define_transform_tuple_kernel!(
    transform_tuple7_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g
    )
);
define_transform_tuple_kernel!(
    transform_tuple8_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h
    )
);
define_transform_tuple_kernel!(
    transform_tuple9_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i
    )
);
define_transform_tuple_kernel!(
    transform_tuple10_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j
    )
);
define_transform_tuple_kernel!(
    transform_tuple11_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j,
        TyK: input_k
    )
);
define_transform_tuple_kernel!(
    transform_tuple12_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j,
        TyK: input_k, TyL: input_l
    )
);

macro_rules! define_tuple_predicate_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Pred: PredicateOp<($( $ty, )+)>>(
            $( $input: &Array<$ty>, )+
            len: &Array<u32>,
            invert: &Array<u32>,
            flags: &mut Array<u32>,
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
define_tuple_predicate_flags_kernel!(
    tuple4_predicate_flags_kernel,
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d)
);
define_tuple_predicate_flags_kernel!(
    tuple5_predicate_flags_kernel,
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e)
);
define_tuple_predicate_flags_kernel!(
    tuple6_predicate_flags_kernel,
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f)
);
define_tuple_predicate_flags_kernel!(
    tuple7_predicate_flags_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g
    )
);
define_tuple_predicate_flags_kernel!(
    tuple8_predicate_flags_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h
    )
);
define_tuple_predicate_flags_kernel!(
    tuple9_predicate_flags_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i
    )
);
define_tuple_predicate_flags_kernel!(
    tuple10_predicate_flags_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j
    )
);
define_tuple_predicate_flags_kernel!(
    tuple11_predicate_flags_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j,
        TyK: input_k
    )
);
define_tuple_predicate_flags_kernel!(
    tuple12_predicate_flags_kernel,
    (
        TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j,
        TyK: input_k, TyL: input_l
    )
);

#[cube(launch_unchecked)]
pub(crate) fn copy_if_expr_flags_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    invert: &Array<u32>,
    flags: &mut Array<u32>,
    values: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn copy_if_expr_flag_only_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    invert: &Array<u32>,
    flags: &mut Array<u32>,
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

#[cube(launch_unchecked)]
pub(crate) fn copy_if_stencil_expr_flags_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    StencilExpr: GpuExpr<S>,
    Pred: PredicateOp<S>,
>(
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    stencil_input: &Array<S>,
    stencil_indices: &Array<u32>,
    stencil_rhs: &Array<S>,
    stencil_rhs_indices: &Array<u32>,
    len: &Array<u32>,
    invert: &Array<u32>,
    flags: &mut Array<u32>,
    values: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn transform_if_stencil_expr_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    StencilExpr: GpuExpr<S>,
    Op: UnaryOp<T, Output = T>,
    Pred: PredicateOp<S>,
>(
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    stencil_input: &Array<S>,
    stencil_indices: &Array<u32>,
    stencil_rhs: &Array<S>,
    stencil_rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn gather_expr_kernel<T: CubePrimitive, IndexExpr: GpuExpr<u32>>(
    output: &mut Array<T>,
    index_input: &Array<u32>,
    index_indices: &Array<u32>,
    index_rhs: &Array<u32>,
    index_rhs_indices: &Array<u32>,
    input: &Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn gather_device_expr_kernel<
    T: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
>(
    output: &mut Array<T>,
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    index_input: &Array<u32>,
    index_indices: &Array<u32>,
    index_rhs: &Array<u32>,
    index_rhs_indices: &Array<u32>,
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

#[cube(launch_unchecked)]
pub(crate) fn scatter_expr_kernel<
    T: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
>(
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    index_input: &Array<u32>,
    index_indices: &Array<u32>,
    index_rhs: &Array<u32>,
    index_rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn scatter_if_expr_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
    StencilExpr: GpuExpr<S>,
    Pred: PredicateOp<S>,
>(
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    index_input: &Array<u32>,
    index_indices: &Array<u32>,
    index_rhs: &Array<u32>,
    index_rhs_indices: &Array<u32>,
    stencil_input: &Array<S>,
    stencil_indices: &Array<u32>,
    stencil_rhs: &Array<S>,
    stencil_rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn compact_count_kernel(positions: &Array<u32>, count: &mut Array<u32>) {
    if UNIT_POS == 0 {
        count[0] = positions[positions.len() - 1usize];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn u32_block_inclusive_scan_kernel(
    input: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<u32>,
    block_sums: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn u32_add_block_prefix_kernel(
    block_prefixes: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;

    if block > 0usize && global < (len[0] as usize) {
        output[global] = output[global] + block_prefixes[block - 1usize];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn inclusive_scan_expr_block_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Op: BinaryOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<T>,
    block_sums: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = SharedMemory::<T>::new(cube_dim);
    let mut valid = SharedMemory::<u32>::new(cube_dim);

    if global < logical_len {
        values[unit] = Expr::eval(input, indices, rhs, rhs_indices, global);
        valid[unit] = 1u32;
    } else {
        valid[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend = RuntimeCell::<T>::new(values[unit]);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() && valid[unit - stride.read()] != 0 {
            addend.store(values[unit - stride.read()]);
            addend_valid.store(1u32);
        }
        sync_cube();
        if addend_valid.read() != 0 {
            if valid[unit] != 0 {
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

#[cube(launch_unchecked)]
pub(crate) fn scan_add_block_prefix_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
    block_prefixes: &Array<T>,
    len: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;

    if block > 0usize && global < (len[0] as usize) {
        output[global] = Op::apply(block_prefixes[block - 1usize], output[global]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn device_inclusive_scan_expr_block_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Op: BinaryOp<T>,
>(
    slot0: &Array<T>,
    slot1: &Array<T>,
    slot2: &Array<T>,
    slot3: &Array<T>,
    len: &Array<u32>,
    output: &mut Array<T>,
    block_sums: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = SharedMemory::<T>::new(cube_dim);
    let mut valid = SharedMemory::<u32>::new(cube_dim);

    if global < logical_len {
        values[unit] = Expr::eval(slot0, slot1, slot2, slot3, global);
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

#[cube(launch_unchecked)]
pub(crate) fn scan_make_exclusive_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
    inclusive: &Array<T>,
    init: &Array<T>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output.len() {
        if global == 0usize {
            output[global] = init[0];
        } else {
            output[global] = Op::apply(init[0], inclusive[global - 1usize]);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn compact_scatter_kernel<T: CubePrimitive>(
    flags: &Array<u32>,
    positions: &Array<u32>,
    values: &Array<T>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() && flags[global] != 0u32 {
        let position = positions[global];
        output[(position - 1u32) as usize] = values[global];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn partition_scatter_kernel<T: CubePrimitive>(
    flags: &Array<u32>,
    positions: &Array<u32>,
    selected_count: &Array<u32>,
    values: &Array<T>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < flags.len() {
        if flags[global] != 0u32 {
            let position = positions[global];
            output[(position - 1u32) as usize] = values[global];
        } else {
            let position = positions[global];
            let total_selected = selected_count[0];
            let non_selected_before = (global as u32) - position;
            output[(total_selected + non_selected_before) as usize] = values[global];
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn adjacent_difference_expr_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Op: BinaryOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < (len[0] as usize) {
        if global == 0usize {
            output[global] = Expr::eval(input, indices, rhs, rhs_indices, global);
        } else {
            output[global] = Op::apply(
                Expr::eval(input, indices, rhs, rhs_indices, global),
                Expr::eval(input, indices, rhs, rhs_indices, global - 1usize),
            );
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn inclusive_scan_by_key_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    KeyEq: crate::op::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &Array<K>,
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn inclusive_scan_by_key_expr_keys_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: GpuExpr<K>,
    ValueExpr: GpuExpr<T>,
    KeyEq: crate::op::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    key_input: &Array<K>,
    key_indices: &Array<u32>,
    key_rhs: &Array<K>,
    key_rhs_indices: &Array<u32>,
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn exclusive_scan_by_key_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    KeyEq: crate::op::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &Array<K>,
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    init: &Array<T>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn exclusive_scan_by_key_expr_keys_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: GpuExpr<K>,
    ValueExpr: GpuExpr<T>,
    KeyEq: crate::op::BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    key_input: &Array<K>,
    key_indices: &Array<u32>,
    key_rhs: &Array<K>,
    key_rhs_indices: &Array<u32>,
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    len: &Array<u32>,
    init: &Array<T>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn reduce_expr_partials_kernel<T: CubePrimitive, Expr: GpuExpr<T>, Op: BinaryOp<T>>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    partials: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values = SharedMemory::<T>::new(cube_dim);
    let mut valid = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn device_reduce_expr_partials_kernel<
    T: CubePrimitive,
    Expr: DeviceGpuExpr<T>,
    Op: BinaryOp<T>,
>(
    slot0: &Array<T>,
    slot1: &Array<T>,
    slot2: &Array<T>,
    slot3: &Array<T>,
    len: &Array<u32>,
    partials: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut values = SharedMemory::<T>::new(cube_dim);
    let mut valid = SharedMemory::<u32>::new(cube_dim);

    let i = RuntimeCell::<usize>::new((CUBE_POS as usize) * cube_dim + unit);
    let step = (CUBE_DIM as usize) * partials.len();
    let has_value = RuntimeCell::<u32>::new(0u32);
    let acc = RuntimeCell::<T>::new(Expr::eval(slot0, slot1, slot2, slot3, 0));

    while i.read() < logical_len {
        let value = Expr::eval(slot0, slot1, slot2, slot3, i.read());
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

#[cube(launch_unchecked)]
pub(crate) fn reduce_finalize_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
    partial: &Array<T>,
    init: &Array<T>,
    output: &mut Array<T>,
) {
    if UNIT_POS == 0 {
        output[0] = Op::apply(init[0], partial[0]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_by_key_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Op: BinaryOp<T>,
>(
    keys: &Array<K>,
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    init: &Array<T>,
    flags: &mut Array<u32>,
    values: &mut Array<T>,
) {
    let pos = CUBE_POS as usize;
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;

    if pos < logical_len {
        if unit == 0 {
            flags[pos] = 0u32;
        }

        let is_start = pos == 0 || keys[pos] != keys[pos - 1];
        if is_start {
            let key = keys[pos];
            let mut segment_end = SharedMemory::<u32>::new(1usize);
            if unit == 0 {
                let end = RuntimeCell::<usize>::new(pos + 1usize);
                while end.read() < logical_len && keys[end.read()] == key {
                    end.store(end.read() + 1usize);
                }
                segment_end[0] = end.read() as u32;
            }
            sync_cube();

            let end = segment_end[0] as usize;
            let mut values_smem = SharedMemory::<T>::new(cube_dim);
            let mut valid = SharedMemory::<u32>::new(cube_dim);
            let i = RuntimeCell::<usize>::new(pos + unit);
            let has_value = RuntimeCell::<u32>::new(0u32);
            let acc = RuntimeCell::<T>::new(Expr::eval(input, indices, rhs, rhs_indices, pos));

            while i.read() < end {
                let value = Expr::eval(input, indices, rhs, rhs_indices, i.read());
                if has_value.read() != 0 {
                    acc.store(Op::apply(acc.read(), value));
                } else {
                    acc.store(value);
                    has_value.store(1u32);
                }
                i.store(i.read() + cube_dim);
            }

            values_smem[unit] = acc.read();
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
                        values_smem[unit] =
                            Op::apply(values_smem[unit], values_smem[unit + stride.read()]);
                    } else {
                        values_smem[unit] = values_smem[unit + stride.read()];
                        valid[unit] = 1u32;
                    }
                }
                sync_cube();
                stride.store(stride.read() / 2);
            }

            if unit == 0 && valid[0] != 0 {
                flags[pos] = 1u32;
                values[pos] = Op::apply(init[0], values_smem[0]);
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_by_key_expr_keys_expr_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyExpr: GpuExpr<K>,
    ValueExpr: GpuExpr<T>,
    Op: BinaryOp<T>,
>(
    key_input: &Array<K>,
    key_indices: &Array<u32>,
    key_rhs: &Array<K>,
    key_rhs_indices: &Array<u32>,
    value_input: &Array<T>,
    value_indices: &Array<u32>,
    value_rhs: &Array<T>,
    value_rhs_indices: &Array<u32>,
    len: &Array<u32>,
    init: &Array<T>,
    flags: &mut Array<u32>,
    out_keys: &mut Array<K>,
    values: &mut Array<T>,
) {
    let pos = CUBE_POS as usize;
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;

    if pos < logical_len {
        if unit == 0 {
            flags[pos] = 0u32;
        }

        let key = KeyExpr::eval(key_input, key_indices, key_rhs, key_rhs_indices, pos);
        let is_start = pos == 0usize
            || KeyExpr::eval(
                key_input,
                key_indices,
                key_rhs,
                key_rhs_indices,
                pos - 1usize,
            ) != key;
        if is_start {
            let mut segment_end = SharedMemory::<u32>::new(1usize);
            if unit == 0 {
                let end = RuntimeCell::<usize>::new(pos + 1usize);
                while end.read() < logical_len
                    && KeyExpr::eval(key_input, key_indices, key_rhs, key_rhs_indices, end.read())
                        == key
                {
                    end.store(end.read() + 1usize);
                }
                segment_end[0] = end.read() as u32;
            }
            sync_cube();

            let end = segment_end[0] as usize;
            let mut values_smem = SharedMemory::<T>::new(cube_dim);
            let mut valid = SharedMemory::<u32>::new(cube_dim);
            let i = RuntimeCell::<usize>::new(pos + unit);
            let has_value = RuntimeCell::<u32>::new(0u32);
            let acc = RuntimeCell::<T>::new(ValueExpr::eval(
                value_input,
                value_indices,
                value_rhs,
                value_rhs_indices,
                pos,
            ));

            while i.read() < end {
                let value = ValueExpr::eval(
                    value_input,
                    value_indices,
                    value_rhs,
                    value_rhs_indices,
                    i.read(),
                );
                if has_value.read() != 0 {
                    acc.store(Op::apply(acc.read(), value));
                } else {
                    acc.store(value);
                    has_value.store(1u32);
                }
                i.store(i.read() + cube_dim);
            }

            values_smem[unit] = acc.read();
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
                        values_smem[unit] =
                            Op::apply(values_smem[unit], values_smem[unit + stride.read()]);
                    } else {
                        values_smem[unit] = values_smem[unit + stride.read()];
                        valid[unit] = 1u32;
                    }
                }
                sync_cube();
                stride.store(stride.read() / 2);
            }

            if unit == 0 && valid[0] != 0 {
                flags[pos] = 1u32;
                out_keys[pos] = key;
                values[pos] = Op::apply(init[0], values_smem[0]);
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn count_if_expr_partials_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut counts = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn sum_u32_partials_kernel(input: &Array<u32>, partials: &mut Array<u32>) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let mut counts = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn find_if_expr_partials_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: PredicateOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    invert: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut best_indices = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn min_u32_partials_kernel(input: &Array<u32>, partials: &mut Array<u32>) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let mut best_indices = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn adjacent_find_expr_partials_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Pred: BinaryPredicateOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    partials: &mut Array<u32>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let logical_len = len[0] as usize;
    let mut best_indices = SharedMemory::<u32>::new(cube_dim);

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

#[cube(launch_unchecked)]
pub(crate) fn minmax_element_expr_kernel<
    T: CubePrimitive,
    Expr: GpuExpr<T>,
    Less: BinaryPredicateOp<T>,
>(
    input: &Array<T>,
    indices: &Array<u32>,
    rhs: &Array<T>,
    rhs_indices: &Array<u32>,
    len: &Array<u32>,
    output: &mut Array<u32>,
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
