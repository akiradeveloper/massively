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

macro_rules! define_transform_unary_tuple_kernel {
    (
        $fn_name:ident,
        ($( $out_ty:ident : $output:ident : $field:tt ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<
            T: CubePrimitive,
            $( $out_ty: CubePrimitive, )+
            Op: UnaryOp<T, Output = ($( $out_ty, )+)>,
        >(
            input: &Array<T>,
            len: &Array<u32>,
            $( $output: &mut Array<$out_ty>, )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (len[0] as usize) {
                let output = Op::apply(input[global]);
                $(
                    $output[global] = output.$field;
                )+
            }
        }
    };
}

define_transform_unary_tuple_kernel!(
    transform_unary_tuple4_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple5_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple6_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple7_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple8_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple9_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple10_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8, OutJ: output_j: 9)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple11_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8, OutJ: output_j: 9, OutK: output_k: 10)
);
define_transform_unary_tuple_kernel!(
    transform_unary_tuple12_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8, OutJ: output_j: 9, OutK: output_k: 10, OutL: output_l: 11)
);

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

macro_rules! define_transform_tuple_to_tuple_kernel {
    (
        $fn_name:ident,
        ($( $in_ty:ident : $input:ident ),+),
        ($( $out_ty:ident : $output:ident : $field:tt ),+)
    ) => {
        #[allow(dead_code)]
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<
            $( $in_ty: CubePrimitive, )+
            $( $out_ty: CubePrimitive, )+
            Op: UnaryOp<($( $in_ty, )+), Output = ($( $out_ty, )+)>,
        >(
            $( $input: &Array<$in_ty>, )+
            len: &Array<u32>,
            $( $output: &mut Array<$out_ty>, )+
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < (len[0] as usize) {
                let output = Op::apply((
                    $( $input[global], )+
                ));
                $(
                    $output[global] = output.$field;
                )+
            }
        }
    };
}

macro_rules! define_transform_tuple_to_tuple_kernels {
    (
        ($( $in_ty:ident : $input:ident ),+),
        $kernel2:ident,
        $kernel3:ident,
        $kernel4:ident,
        $kernel5:ident,
        $kernel6:ident,
        $kernel7:ident,
        $kernel8:ident,
        $kernel9:ident,
        $kernel10:ident,
        $kernel11:ident,
        $kernel12:ident $(,)?
    ) => {
        define_transform_tuple_to_tuple_kernel!(
            $kernel2,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel3,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel4,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel5,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel6,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel7,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel8,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel9,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel10,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8, OutJ: output_j: 9)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel11,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8, OutJ: output_j: 9, OutK: output_k: 10)
        );
        define_transform_tuple_to_tuple_kernel!(
            $kernel12,
            ($( $in_ty: $input ),+),
            (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6, OutH: output_h: 7, OutI: output_i: 8, OutJ: output_j: 9, OutK: output_k: 10, OutL: output_l: 11)
        );
    };
}

define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b),
    transform_tuple2_to_tuple2_kernel,
    transform_tuple2_to_tuple3_kernel,
    transform_tuple2_to_tuple4_kernel,
    transform_tuple2_to_tuple5_kernel,
    transform_tuple2_to_tuple6_kernel,
    transform_tuple2_to_tuple7_kernel,
    transform_tuple2_to_tuple8_kernel,
    transform_tuple2_to_tuple9_kernel,
    transform_tuple2_to_tuple10_kernel,
    transform_tuple2_to_tuple11_kernel,
    transform_tuple2_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c),
    transform_tuple3_to_tuple2_kernel,
    transform_tuple3_to_tuple3_kernel,
    transform_tuple3_to_tuple4_kernel,
    transform_tuple3_to_tuple5_kernel,
    transform_tuple3_to_tuple6_kernel,
    transform_tuple3_to_tuple7_kernel,
    transform_tuple3_to_tuple8_kernel,
    transform_tuple3_to_tuple9_kernel,
    transform_tuple3_to_tuple10_kernel,
    transform_tuple3_to_tuple11_kernel,
    transform_tuple3_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d),
    transform_tuple4_to_tuple2_kernel,
    transform_tuple4_to_tuple3_kernel,
    transform_tuple4_to_tuple4_kernel,
    transform_tuple4_to_tuple5_kernel,
    transform_tuple4_to_tuple6_kernel,
    transform_tuple4_to_tuple7_kernel,
    transform_tuple4_to_tuple8_kernel,
    transform_tuple4_to_tuple9_kernel,
    transform_tuple4_to_tuple10_kernel,
    transform_tuple4_to_tuple11_kernel,
    transform_tuple4_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e),
    transform_tuple5_to_tuple2_kernel,
    transform_tuple5_to_tuple3_kernel,
    transform_tuple5_to_tuple4_kernel,
    transform_tuple5_to_tuple5_kernel,
    transform_tuple5_to_tuple6_kernel,
    transform_tuple5_to_tuple7_kernel,
    transform_tuple5_to_tuple8_kernel,
    transform_tuple5_to_tuple9_kernel,
    transform_tuple5_to_tuple10_kernel,
    transform_tuple5_to_tuple11_kernel,
    transform_tuple5_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f),
    transform_tuple6_to_tuple2_kernel,
    transform_tuple6_to_tuple3_kernel,
    transform_tuple6_to_tuple4_kernel,
    transform_tuple6_to_tuple5_kernel,
    transform_tuple6_to_tuple6_kernel,
    transform_tuple6_to_tuple7_kernel,
    transform_tuple6_to_tuple8_kernel,
    transform_tuple6_to_tuple9_kernel,
    transform_tuple6_to_tuple10_kernel,
    transform_tuple6_to_tuple11_kernel,
    transform_tuple6_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g),
    transform_tuple7_to_tuple2_kernel,
    transform_tuple7_to_tuple3_kernel,
    transform_tuple7_to_tuple4_kernel,
    transform_tuple7_to_tuple5_kernel,
    transform_tuple7_to_tuple6_kernel,
    transform_tuple7_to_tuple7_kernel,
    transform_tuple7_to_tuple8_kernel,
    transform_tuple7_to_tuple9_kernel,
    transform_tuple7_to_tuple10_kernel,
    transform_tuple7_to_tuple11_kernel,
    transform_tuple7_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h),
    transform_tuple8_to_tuple2_kernel,
    transform_tuple8_to_tuple3_kernel,
    transform_tuple8_to_tuple4_kernel,
    transform_tuple8_to_tuple5_kernel,
    transform_tuple8_to_tuple6_kernel,
    transform_tuple8_to_tuple7_kernel,
    transform_tuple8_to_tuple8_kernel,
    transform_tuple8_to_tuple9_kernel,
    transform_tuple8_to_tuple10_kernel,
    transform_tuple8_to_tuple11_kernel,
    transform_tuple8_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i),
    transform_tuple9_to_tuple2_kernel,
    transform_tuple9_to_tuple3_kernel,
    transform_tuple9_to_tuple4_kernel,
    transform_tuple9_to_tuple5_kernel,
    transform_tuple9_to_tuple6_kernel,
    transform_tuple9_to_tuple7_kernel,
    transform_tuple9_to_tuple8_kernel,
    transform_tuple9_to_tuple9_kernel,
    transform_tuple9_to_tuple10_kernel,
    transform_tuple9_to_tuple11_kernel,
    transform_tuple9_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j),
    transform_tuple10_to_tuple2_kernel,
    transform_tuple10_to_tuple3_kernel,
    transform_tuple10_to_tuple4_kernel,
    transform_tuple10_to_tuple5_kernel,
    transform_tuple10_to_tuple6_kernel,
    transform_tuple10_to_tuple7_kernel,
    transform_tuple10_to_tuple8_kernel,
    transform_tuple10_to_tuple9_kernel,
    transform_tuple10_to_tuple10_kernel,
    transform_tuple10_to_tuple11_kernel,
    transform_tuple10_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k),
    transform_tuple11_to_tuple2_kernel,
    transform_tuple11_to_tuple3_kernel,
    transform_tuple11_to_tuple4_kernel,
    transform_tuple11_to_tuple5_kernel,
    transform_tuple11_to_tuple6_kernel,
    transform_tuple11_to_tuple7_kernel,
    transform_tuple11_to_tuple8_kernel,
    transform_tuple11_to_tuple9_kernel,
    transform_tuple11_to_tuple10_kernel,
    transform_tuple11_to_tuple11_kernel,
    transform_tuple11_to_tuple12_kernel,
);
define_transform_tuple_to_tuple_kernels!(
    (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k, TyL: input_l),
    transform_tuple12_to_tuple2_kernel,
    transform_tuple12_to_tuple3_kernel,
    transform_tuple12_to_tuple4_kernel,
    transform_tuple12_to_tuple5_kernel,
    transform_tuple12_to_tuple6_kernel,
    transform_tuple12_to_tuple7_kernel,
    transform_tuple12_to_tuple8_kernel,
    transform_tuple12_to_tuple9_kernel,
    transform_tuple12_to_tuple10_kernel,
    transform_tuple12_to_tuple11_kernel,
    transform_tuple12_to_tuple12_kernel,
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

macro_rules! define_tuple_adjacent_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Pred: BinaryPredicateOp<($( $ty, )+)>>(
            $( $input: &Array<$ty>, )+
            flags: &mut Array<u32>,
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
define_tuple_adjacent_flags_kernel!(tuple4_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d));
define_tuple_adjacent_flags_kernel!(tuple5_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e));
define_tuple_adjacent_flags_kernel!(tuple6_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f));
define_tuple_adjacent_flags_kernel!(tuple7_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g));
define_tuple_adjacent_flags_kernel!(tuple8_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h));
define_tuple_adjacent_flags_kernel!(tuple9_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i));
define_tuple_adjacent_flags_kernel!(tuple10_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j));
define_tuple_adjacent_flags_kernel!(tuple11_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k));
define_tuple_adjacent_flags_kernel!(tuple12_adjacent_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k, TyL: input_l));

macro_rules! define_tuple_unique_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Pred: BinaryPredicateOp<($( $ty, )+)>>(
            $( $input: &Array<$ty>, )+
            flags: &mut Array<u32>,
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
define_tuple_unique_flags_kernel!(tuple4_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d));
define_tuple_unique_flags_kernel!(tuple5_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e));
define_tuple_unique_flags_kernel!(tuple6_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f));
define_tuple_unique_flags_kernel!(tuple7_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g));
define_tuple_unique_flags_kernel!(tuple8_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h));
define_tuple_unique_flags_kernel!(tuple9_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i));
define_tuple_unique_flags_kernel!(tuple10_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j));
define_tuple_unique_flags_kernel!(tuple11_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k));
define_tuple_unique_flags_kernel!(tuple12_unique_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k, TyL: input_l));

macro_rules! define_tuple_mismatch_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $left:ident / $right:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Eq: BinaryPredicateOp<($( $ty, )+)>>(
            $( $left: &Array<$ty>, )+
            $( $right: &Array<$ty>, )+
            flags: &mut Array<u32>,
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
define_tuple_mismatch_flags_kernel!(tuple4_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d));
define_tuple_mismatch_flags_kernel!(tuple5_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e));
define_tuple_mismatch_flags_kernel!(tuple6_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f));
define_tuple_mismatch_flags_kernel!(tuple7_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g));
define_tuple_mismatch_flags_kernel!(tuple8_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h));
define_tuple_mismatch_flags_kernel!(tuple9_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i));
define_tuple_mismatch_flags_kernel!(tuple10_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j));
define_tuple_mismatch_flags_kernel!(tuple11_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k));
define_tuple_mismatch_flags_kernel!(tuple12_mismatch_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k, TyL: left_l / right_l));

macro_rules! define_tuple_sorted_break_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Less: BinaryPredicateOp<($( $ty, )+)>>(
            $( $input: &Array<$ty>, )+
            flags: &mut Array<u32>,
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
define_tuple_sorted_break_flags_kernel!(tuple4_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d));
define_tuple_sorted_break_flags_kernel!(tuple5_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e));
define_tuple_sorted_break_flags_kernel!(tuple6_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f));
define_tuple_sorted_break_flags_kernel!(tuple7_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g));
define_tuple_sorted_break_flags_kernel!(tuple8_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h));
define_tuple_sorted_break_flags_kernel!(tuple9_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i));
define_tuple_sorted_break_flags_kernel!(tuple10_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j));
define_tuple_sorted_break_flags_kernel!(tuple11_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k));
define_tuple_sorted_break_flags_kernel!(tuple12_sorted_break_flags_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k, TyL: input_l));

macro_rules! define_tuple_bound_flags_kernel {
    (
        $lower_fn:ident,
        $upper_fn:ident,
        ($first_ty:ident : $first_input:ident / $first_value:ident $(, $ty:ident : $input:ident / $value:ident )*)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $lower_fn<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>>(
            $first_input: &Array<$first_ty>,
            $( $input: &Array<$ty>, )*
            $first_value: &Array<$first_ty>,
            $( $value: &Array<$ty>, )*
            flags: &mut Array<u32>,
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

        #[cube(launch_unchecked)]
        pub(crate) fn $upper_fn<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>>(
            $first_input: &Array<$first_ty>,
            $( $input: &Array<$ty>, )*
            $first_value: &Array<$first_ty>,
            $( $value: &Array<$ty>, )*
            flags: &mut Array<u32>,
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
define_tuple_bound_flags_kernel!(tuple4_lower_bound_flags_kernel, tuple4_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d));
define_tuple_bound_flags_kernel!(tuple5_lower_bound_flags_kernel, tuple5_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e));
define_tuple_bound_flags_kernel!(tuple6_lower_bound_flags_kernel, tuple6_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f));
define_tuple_bound_flags_kernel!(tuple7_lower_bound_flags_kernel, tuple7_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g));
define_tuple_bound_flags_kernel!(tuple8_lower_bound_flags_kernel, tuple8_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h));
define_tuple_bound_flags_kernel!(tuple9_lower_bound_flags_kernel, tuple9_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i));
define_tuple_bound_flags_kernel!(tuple10_lower_bound_flags_kernel, tuple10_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i, TyJ: input_j / value_j));
define_tuple_bound_flags_kernel!(tuple11_lower_bound_flags_kernel, tuple11_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i, TyJ: input_j / value_j, TyK: input_k / value_k));
define_tuple_bound_flags_kernel!(tuple12_lower_bound_flags_kernel, tuple12_upper_bound_flags_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i, TyJ: input_j / value_j, TyK: input_k / value_k, TyL: input_l / value_l));

macro_rules! define_tuple_binary_search_at_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_input:ident / $first_value:ident $(, $ty:ident : $input:ident / $value:ident )*)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>>(
            $first_input: &Array<$first_ty>,
            $( $input: &Array<$ty>, )*
            $first_value: &Array<$first_ty>,
            $( $value: &Array<$ty>, )*
            index: &Array<u32>,
            output: &mut Array<u32>,
        ) {
            if UNIT_POS == 0 {
                let i = index[0] as usize;
                if i < $first_input.len()
                    && !Less::apply(
                        ($first_input[i], $( $input[i], )*),
                        ($first_value[0], $( $value[0], )*),
                    )
                    && !Less::apply(
                        ($first_value[0], $( $value[0], )*),
                        ($first_input[i], $( $input[i], )*),
                    )
                {
                    output[0] = 1u32;
                } else {
                    output[0] = 0u32;
                }
            }
        }
    };
}

define_tuple_binary_search_at_kernel!(tuple2_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b));
define_tuple_binary_search_at_kernel!(tuple3_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c));
define_tuple_binary_search_at_kernel!(tuple4_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d));
define_tuple_binary_search_at_kernel!(tuple5_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e));
define_tuple_binary_search_at_kernel!(tuple6_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f));
define_tuple_binary_search_at_kernel!(tuple7_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g));
define_tuple_binary_search_at_kernel!(tuple8_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h));
define_tuple_binary_search_at_kernel!(tuple9_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i));
define_tuple_binary_search_at_kernel!(tuple10_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i, TyJ: input_j / value_j));
define_tuple_binary_search_at_kernel!(tuple11_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i, TyJ: input_j / value_j, TyK: input_k / value_k));
define_tuple_binary_search_at_kernel!(tuple12_binary_search_at_kernel, (TyA: input_a / value_a, TyB: input_b / value_b, TyC: input_c / value_c, TyD: input_d / value_d, TyE: input_e / value_e, TyF: input_f / value_f, TyG: input_g / value_g, TyH: input_h / value_h, TyI: input_i / value_i, TyJ: input_j / value_j, TyK: input_k / value_k, TyL: input_l / value_l));

macro_rules! define_tuple_membership_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_candidate:ident / $first_sorted:ident $(, $ty:ident : $candidate:ident / $sorted:ident )*)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>>(
            $first_candidate: &Array<$first_ty>,
            $( $candidate: &Array<$ty>, )*
            $first_sorted: &Array<$first_ty>,
            $( $sorted: &Array<$ty>, )*
            keep_present: &Array<u32>,
            flags: &mut Array<u32>,
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
define_tuple_membership_flags_kernel!(tuple4_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d));
define_tuple_membership_flags_kernel!(tuple5_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e));
define_tuple_membership_flags_kernel!(tuple6_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e, TyF: candidate_f / sorted_f));
define_tuple_membership_flags_kernel!(tuple7_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e, TyF: candidate_f / sorted_f, TyG: candidate_g / sorted_g));
define_tuple_membership_flags_kernel!(tuple8_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e, TyF: candidate_f / sorted_f, TyG: candidate_g / sorted_g, TyH: candidate_h / sorted_h));
define_tuple_membership_flags_kernel!(tuple9_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e, TyF: candidate_f / sorted_f, TyG: candidate_g / sorted_g, TyH: candidate_h / sorted_h, TyI: candidate_i / sorted_i));
define_tuple_membership_flags_kernel!(tuple10_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e, TyF: candidate_f / sorted_f, TyG: candidate_g / sorted_g, TyH: candidate_h / sorted_h, TyI: candidate_i / sorted_i, TyJ: candidate_j / sorted_j));
define_tuple_membership_flags_kernel!(tuple11_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e, TyF: candidate_f / sorted_f, TyG: candidate_g / sorted_g, TyH: candidate_h / sorted_h, TyI: candidate_i / sorted_i, TyJ: candidate_j / sorted_j, TyK: candidate_k / sorted_k));
define_tuple_membership_flags_kernel!(tuple12_membership_flags_kernel, (TyA: candidate_a / sorted_a, TyB: candidate_b / sorted_b, TyC: candidate_c / sorted_c, TyD: candidate_d / sorted_d, TyE: candidate_e / sorted_e, TyF: candidate_f / sorted_f, TyG: candidate_g / sorted_g, TyH: candidate_h / sorted_h, TyI: candidate_i / sorted_i, TyJ: candidate_j / sorted_j, TyK: candidate_k / sorted_k, TyL: candidate_l / sorted_l));

macro_rules! define_tuple_minmax_kernels {
    (
        $element_fn:ident,
        $index_fn:ident,
        ($( $ty:ident : $input:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $element_fn<$( $ty: CubePrimitive, )+ Less: BinaryPredicateOp<($( $ty, )+)>>(
            $( $input: &Array<$ty>, )+
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

        #[cube(launch_unchecked)]
        pub(crate) fn $index_fn<$( $ty: CubePrimitive, )+ Less: BinaryPredicateOp<($( $ty, )+)>>(
            $( $input: &Array<$ty>, )+
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
define_tuple_minmax_kernels!(tuple4_minmax_element_partials_kernel, tuple4_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d));
define_tuple_minmax_kernels!(tuple5_minmax_element_partials_kernel, tuple5_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e));
define_tuple_minmax_kernels!(tuple6_minmax_element_partials_kernel, tuple6_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f));
define_tuple_minmax_kernels!(tuple7_minmax_element_partials_kernel, tuple7_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g));
define_tuple_minmax_kernels!(tuple8_minmax_element_partials_kernel, tuple8_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h));
define_tuple_minmax_kernels!(tuple9_minmax_element_partials_kernel, tuple9_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i));
define_tuple_minmax_kernels!(tuple10_minmax_element_partials_kernel, tuple10_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j));
define_tuple_minmax_kernels!(tuple11_minmax_element_partials_kernel, tuple11_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k));
define_tuple_minmax_kernels!(tuple12_minmax_element_partials_kernel, tuple12_minmax_index_partials_kernel, (TyA: input_a, TyB: input_b, TyC: input_c, TyD: input_d, TyE: input_e, TyF: input_f, TyG: input_g, TyH: input_h, TyI: input_i, TyJ: input_j, TyK: input_k, TyL: input_l));

macro_rules! define_tuple_includes_missing_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_left:ident / $first_right:ident $(, $ty:ident : $left:ident / $right:ident )*)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Less: BinaryPredicateOp<($first_ty, $( $ty, )*)>>(
            $first_left: &Array<$first_ty>,
            $( $left: &Array<$ty>, )*
            $first_right: &Array<$first_ty>,
            $( $right: &Array<$ty>, )*
            flags: &mut Array<u32>,
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < flags.len() {
                let value = ($first_right[global], $( $right[global], )*);

                let left_lower = RuntimeCell::<usize>::new(0usize);
                let count = RuntimeCell::<usize>::new($first_left.len());
                while count.read() > 0usize {
                    let step = count.read() / 2usize;
                    let mid = left_lower.read() + step;
                    if Less::apply(($first_left[mid], $( $left[mid], )*), value) {
                        left_lower.store(mid + 1usize);
                        count.store(count.read() - step - 1usize);
                    } else {
                        count.store(step);
                    }
                }

                let left_upper = RuntimeCell::<usize>::new(0usize);
                let count = RuntimeCell::<usize>::new($first_left.len());
                while count.read() > 0usize {
                    let step = count.read() / 2usize;
                    let mid = left_upper.read() + step;
                    if Less::apply(value, ($first_left[mid], $( $left[mid], )*)) {
                        count.store(step);
                    } else {
                        left_upper.store(mid + 1usize);
                        count.store(count.read() - step - 1usize);
                    }
                }

                let right_lower = RuntimeCell::<usize>::new(0usize);
                let count = RuntimeCell::<usize>::new($first_right.len());
                while count.read() > 0usize {
                    let step = count.read() / 2usize;
                    let mid = right_lower.read() + step;
                    if Less::apply(($first_right[mid], $( $right[mid], )*), value) {
                        right_lower.store(mid + 1usize);
                        count.store(count.read() - step - 1usize);
                    } else {
                        count.store(step);
                    }
                }

                let right_upper = RuntimeCell::<usize>::new(0usize);
                let count = RuntimeCell::<usize>::new($first_right.len());
                while count.read() > 0usize {
                    let step = count.read() / 2usize;
                    let mid = right_upper.read() + step;
                    if Less::apply(value, ($first_right[mid], $( $right[mid], )*)) {
                        count.store(step);
                    } else {
                        right_upper.store(mid + 1usize);
                        count.store(count.read() - step - 1usize);
                    }
                }

                if (left_upper.read() - left_lower.read()) < (right_upper.read() - right_lower.read()) {
                    flags[global] = 1u32;
                } else {
                    flags[global] = 0u32;
                }
            }
        }
    };
}

define_tuple_includes_missing_flags_kernel!(tuple2_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b));
define_tuple_includes_missing_flags_kernel!(tuple3_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c));
define_tuple_includes_missing_flags_kernel!(tuple4_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d));
define_tuple_includes_missing_flags_kernel!(tuple5_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e));
define_tuple_includes_missing_flags_kernel!(tuple6_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f));
define_tuple_includes_missing_flags_kernel!(tuple7_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g));
define_tuple_includes_missing_flags_kernel!(tuple8_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h));
define_tuple_includes_missing_flags_kernel!(tuple9_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i));
define_tuple_includes_missing_flags_kernel!(tuple10_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j));
define_tuple_includes_missing_flags_kernel!(tuple11_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k));
define_tuple_includes_missing_flags_kernel!(tuple12_includes_missing_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k, TyL: left_l / right_l));

macro_rules! define_tuple_find_first_of_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_input:ident / $first_needle:ident $(, $ty:ident : $input:ident / $needle:ident )*)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Eq: BinaryPredicateOp<($first_ty, $( $ty, )*)>>(
            $first_input: &Array<$first_ty>,
            $( $input: &Array<$ty>, )*
            $first_needle: &Array<$first_ty>,
            $( $needle: &Array<$ty>, )*
            flags: &mut Array<u32>,
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
define_tuple_find_first_of_flags_kernel!(tuple4_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d));
define_tuple_find_first_of_flags_kernel!(tuple5_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e));
define_tuple_find_first_of_flags_kernel!(tuple6_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e, TyF: input_f / needle_f));
define_tuple_find_first_of_flags_kernel!(tuple7_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e, TyF: input_f / needle_f, TyG: input_g / needle_g));
define_tuple_find_first_of_flags_kernel!(tuple8_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e, TyF: input_f / needle_f, TyG: input_g / needle_g, TyH: input_h / needle_h));
define_tuple_find_first_of_flags_kernel!(tuple9_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e, TyF: input_f / needle_f, TyG: input_g / needle_g, TyH: input_h / needle_h, TyI: input_i / needle_i));
define_tuple_find_first_of_flags_kernel!(tuple10_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e, TyF: input_f / needle_f, TyG: input_g / needle_g, TyH: input_h / needle_h, TyI: input_i / needle_i, TyJ: input_j / needle_j));
define_tuple_find_first_of_flags_kernel!(tuple11_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e, TyF: input_f / needle_f, TyG: input_g / needle_g, TyH: input_h / needle_h, TyI: input_i / needle_i, TyJ: input_j / needle_j, TyK: input_k / needle_k));
define_tuple_find_first_of_flags_kernel!(tuple12_find_first_of_flags_kernel, (TyA: input_a / needle_a, TyB: input_b / needle_b, TyC: input_c / needle_c, TyD: input_d / needle_d, TyE: input_e / needle_e, TyF: input_f / needle_f, TyG: input_g / needle_g, TyH: input_h / needle_h, TyI: input_i / needle_i, TyJ: input_j / needle_j, TyK: input_k / needle_k, TyL: input_l / needle_l));

macro_rules! define_tuple_subrange_match_flags_kernel {
    (
        $fn_name:ident,
        ($first_ty:ident : $first_input:ident / $first_pattern:ident $(, $ty:ident : $input:ident / $pattern:ident )*)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$first_ty: CubePrimitive, $( $ty: CubePrimitive, )* Eq: BinaryPredicateOp<($first_ty, $( $ty, )*)>>(
            $first_input: &Array<$first_ty>,
            $( $input: &Array<$ty>, )*
            $first_pattern: &Array<$first_ty>,
            $( $pattern: &Array<$ty>, )*
            flags: &mut Array<u32>,
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let start = (CUBE_POS as usize) * cube_dim + unit;
            if start < flags.len() {
                let offset = RuntimeCell::<usize>::new(0usize);
                let matched = RuntimeCell::<u32>::new(1u32);
                while offset.read() < $first_pattern.len() {
                    if !Eq::apply(
                        ($first_input[start + offset.read()], $( $input[start + offset.read()], )*),
                        ($first_pattern[offset.read()], $( $pattern[offset.read()], )*),
                    ) {
                        matched.store(0u32);
                        offset.store($first_pattern.len());
                    } else {
                        offset.store(offset.read() + 1usize);
                    }
                }
                flags[start] = matched.read();
            }
        }
    };
}

define_tuple_subrange_match_flags_kernel!(tuple2_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b));
define_tuple_subrange_match_flags_kernel!(tuple3_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c));
define_tuple_subrange_match_flags_kernel!(tuple4_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d));
define_tuple_subrange_match_flags_kernel!(tuple5_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e));
define_tuple_subrange_match_flags_kernel!(tuple6_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e, TyF: input_f / pattern_f));
define_tuple_subrange_match_flags_kernel!(tuple7_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e, TyF: input_f / pattern_f, TyG: input_g / pattern_g));
define_tuple_subrange_match_flags_kernel!(tuple8_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e, TyF: input_f / pattern_f, TyG: input_g / pattern_g, TyH: input_h / pattern_h));
define_tuple_subrange_match_flags_kernel!(tuple9_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e, TyF: input_f / pattern_f, TyG: input_g / pattern_g, TyH: input_h / pattern_h, TyI: input_i / pattern_i));
define_tuple_subrange_match_flags_kernel!(tuple10_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e, TyF: input_f / pattern_f, TyG: input_g / pattern_g, TyH: input_h / pattern_h, TyI: input_i / pattern_i, TyJ: input_j / pattern_j));
define_tuple_subrange_match_flags_kernel!(tuple11_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e, TyF: input_f / pattern_f, TyG: input_g / pattern_g, TyH: input_h / pattern_h, TyI: input_i / pattern_i, TyJ: input_j / pattern_j, TyK: input_k / pattern_k));
define_tuple_subrange_match_flags_kernel!(tuple12_subrange_match_flags_kernel, (TyA: input_a / pattern_a, TyB: input_b / pattern_b, TyC: input_c / pattern_c, TyD: input_d / pattern_d, TyE: input_e / pattern_e, TyF: input_f / pattern_f, TyG: input_g / pattern_g, TyH: input_h / pattern_h, TyI: input_i / pattern_i, TyJ: input_j / pattern_j, TyK: input_k / pattern_k, TyL: input_l / pattern_l));

macro_rules! define_tuple_lexicographical_diff_flags_kernel {
    (
        $fn_name:ident,
        ($( $ty:ident : $left:ident / $right:ident ),+)
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Less: BinaryPredicateOp<($( $ty, )+)>>(
            $( $left: &Array<$ty>, )+
            $( $right: &Array<$ty>, )+
            flags: &mut Array<u32>,
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
        #[cube(launch_unchecked)]
        pub(crate) fn $fn_name<$( $ty: CubePrimitive, )+ Less: BinaryPredicateOp<($( $ty, )+)>>(
            $( $left: &Array<$ty>, )+
            $( $right: &Array<$ty>, )+
            index: &Array<u32>,
            output: &mut Array<u32>,
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
define_tuple_lexicographical_diff_flags_kernel!(tuple4_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d));
define_tuple_lexicographical_diff_flags_kernel!(tuple5_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e));
define_tuple_lexicographical_diff_flags_kernel!(tuple6_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f));
define_tuple_lexicographical_diff_flags_kernel!(tuple7_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g));
define_tuple_lexicographical_diff_flags_kernel!(tuple8_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h));
define_tuple_lexicographical_diff_flags_kernel!(tuple9_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i));
define_tuple_lexicographical_diff_flags_kernel!(tuple10_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j));
define_tuple_lexicographical_diff_flags_kernel!(tuple11_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k));
define_tuple_lexicographical_diff_flags_kernel!(tuple12_lexicographical_diff_flags_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k, TyL: left_l / right_l));

define_tuple_lexicographical_compare_at_kernel!(tuple2_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b));
define_tuple_lexicographical_compare_at_kernel!(tuple3_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c));
define_tuple_lexicographical_compare_at_kernel!(tuple4_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d));
define_tuple_lexicographical_compare_at_kernel!(tuple5_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e));
define_tuple_lexicographical_compare_at_kernel!(tuple6_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f));
define_tuple_lexicographical_compare_at_kernel!(tuple7_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g));
define_tuple_lexicographical_compare_at_kernel!(tuple8_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h));
define_tuple_lexicographical_compare_at_kernel!(tuple9_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i));
define_tuple_lexicographical_compare_at_kernel!(tuple10_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j));
define_tuple_lexicographical_compare_at_kernel!(tuple11_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k));
define_tuple_lexicographical_compare_at_kernel!(tuple12_lexicographical_compare_at_kernel, (TyA: left_a / right_a, TyB: left_b / right_b, TyC: left_c / right_c, TyD: left_d / right_d, TyE: left_e / right_e, TyF: left_f / right_f, TyG: left_g / right_g, TyH: left_h / right_h, TyI: left_i / right_i, TyJ: left_j / right_j, TyK: left_k / right_k, TyL: left_l / right_l));

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
    ValueExpr: GpuExpr<T>,
    IndexExpr: GpuExpr<u32>,
    Pred: PredicateOp<T>,
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
pub(crate) fn compact_rejected_scatter_kernel<T: CubePrimitive>(
    flags: &Array<u32>,
    positions: &Array<u32>,
    values: &Array<T>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn compact_scatter_pair_kernel<A: CubePrimitive, B: CubePrimitive>(
    flags: &Array<u32>,
    positions: &Array<u32>,
    values_a: &Array<A>,
    values_b: &Array<B>,
    output_a: &mut Array<A>,
    output_b: &mut Array<B>,
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
