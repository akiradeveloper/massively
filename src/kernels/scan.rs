use crate::op::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn scan_by_key_pass_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &Array<K>,
    input: &Array<T>,
    offset: &Array<u32>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn scan_tuple2_by_key_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<(A, B)>,
    Op: BinaryOp<T>,
>(
    key_a: &Array<A>,
    key_b: &Array<B>,
    input: &Array<T>,
    offset: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        let step = offset[0] as usize;
        if global >= step
            && KeyEq::apply(
                (key_a[global - step], key_b[global - step]),
                (key_a[global], key_b[global]),
            )
        {
            output[global] = Op::apply(input[global - step], input[global]);
        } else {
            output[global] = input[global];
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scan_tuple3_by_key_pass_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<(A, B, C)>,
    Op: BinaryOp<T>,
>(
    key_a: &Array<A>,
    key_b: &Array<B>,
    key_c: &Array<C>,
    input: &Array<T>,
    offset: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < input.len() {
        let step = offset[0] as usize;
        if global >= step
            && KeyEq::apply(
                (
                    key_a[global - step],
                    key_b[global - step],
                    key_c[global - step],
                ),
                (key_a[global], key_b[global], key_c[global]),
            )
        {
            output[global] = Op::apply(input[global - step], input[global]);
        } else {
            output[global] = input[global];
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scan_by_key_make_exclusive_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &Array<K>,
    inclusive: &Array<T>,
    init: &Array<T>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn scan_tuple2_by_key_make_exclusive_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<(A, B)>,
    Op: BinaryOp<T>,
>(
    key_a: &Array<A>,
    key_b: &Array<B>,
    inclusive: &Array<T>,
    init: &Array<T>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output.len() {
        if global == 0usize
            || !KeyEq::apply(
                (key_a[global - 1usize], key_b[global - 1usize]),
                (key_a[global], key_b[global]),
            )
        {
            output[global] = init[0];
        } else {
            output[global] = Op::apply(init[0], inclusive[global - 1usize]);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scan_tuple3_by_key_make_exclusive_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<(A, B, C)>,
    Op: BinaryOp<T>,
>(
    key_a: &Array<A>,
    key_b: &Array<B>,
    key_c: &Array<C>,
    inclusive: &Array<T>,
    init: &Array<T>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < output.len() {
        if global == 0usize
            || !KeyEq::apply(
                (
                    key_a[global - 1usize],
                    key_b[global - 1usize],
                    key_c[global - 1usize],
                ),
                (key_a[global], key_b[global], key_c[global]),
            )
        {
            output[global] = init[0];
        } else {
            output[global] = Op::apply(init[0], inclusive[global - 1usize]);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_by_key_end_flags_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &Array<K>,
    inclusive: &Array<T>,
    init: &Array<T>,
    flags: &mut Array<u32>,
    values: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn reduce_tuple2_by_key_end_flags_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<(A, B)>,
    Op: BinaryOp<T>,
>(
    key_a: &Array<A>,
    key_b: &Array<B>,
    inclusive: &Array<T>,
    init: &Array<T>,
    flags: &mut Array<u32>,
    values: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < key_a.len() {
        if global + 1usize == key_a.len()
            || !KeyEq::apply(
                (key_a[global], key_b[global]),
                (key_a[global + 1usize], key_b[global + 1usize]),
            )
        {
            flags[global] = 1u32;
            values[global] = Op::apply(init[0], inclusive[global]);
        } else {
            flags[global] = 0u32;
            values[global] = inclusive[global];
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_tuple3_by_key_end_flags_kernel<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<(A, B, C)>,
    Op: BinaryOp<T>,
>(
    key_a: &Array<A>,
    key_b: &Array<B>,
    key_c: &Array<C>,
    inclusive: &Array<T>,
    init: &Array<T>,
    flags: &mut Array<u32>,
    values: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    if global < key_a.len() {
        if global + 1usize == key_a.len()
            || !KeyEq::apply(
                (key_a[global], key_b[global], key_c[global]),
                (
                    key_a[global + 1usize],
                    key_b[global + 1usize],
                    key_c[global + 1usize],
                ),
            )
        {
            flags[global] = 1u32;
            values[global] = Op::apply(init[0], inclusive[global]);
        } else {
            flags[global] = 0u32;
            values[global] = inclusive[global];
        }
    }
}

macro_rules! define_tuple_by_key_scan_kernels {
    (
        $pass_name:ident,
        $exclusive_name:ident,
        $reduce_name:ident,
        ( $( $ty:ident: $key:ident ),+ )
    ) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $pass_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &Array<$ty>, )+
            input: &Array<T>,
            offset: &Array<u32>,
            output: &mut Array<T>,
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < input.len() {
                let step = offset[0] as usize;
                if global >= step
                    && KeyEq::apply(
                        ($( $key[global - step] ),+),
                        ($( $key[global] ),+),
                    )
                {
                    output[global] = Op::apply(input[global - step], input[global]);
                } else {
                    output[global] = input[global];
                }
            }
        }

        #[cube(launch_unchecked)]
        pub(crate) fn $exclusive_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &Array<$ty>, )+
            inclusive: &Array<T>,
            init: &Array<T>,
            output: &mut Array<T>,
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

        #[cube(launch_unchecked)]
        pub(crate) fn $reduce_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &Array<$ty>, )+
            inclusive: &Array<T>,
            init: &Array<T>,
            flags: &mut Array<u32>,
            values: &mut Array<T>,
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let global = (CUBE_POS as usize) * cube_dim + unit;
            if global < inclusive.len() {
                if global + 1usize == inclusive.len()
                    || !KeyEq::apply(
                        ($( $key[global] ),+),
                        ($( $key[global + 1usize] ),+),
                    )
                {
                    flags[global] = 1u32;
                    values[global] = Op::apply(init[0], inclusive[global]);
                } else {
                    flags[global] = 0u32;
                    values[global] = inclusive[global];
                }
            }
        }
    };
}

define_tuple_by_key_scan_kernels!(
    scan_tuple4_by_key_pass_kernel,
    scan_tuple4_by_key_make_exclusive_kernel,
    reduce_tuple4_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple5_by_key_pass_kernel,
    scan_tuple5_by_key_make_exclusive_kernel,
    reduce_tuple5_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple6_by_key_pass_kernel,
    scan_tuple6_by_key_make_exclusive_kernel,
    reduce_tuple6_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e, F: key_f)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple7_by_key_pass_kernel,
    scan_tuple7_by_key_make_exclusive_kernel,
    reduce_tuple7_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e, F: key_f, G: key_g)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple8_by_key_pass_kernel,
    scan_tuple8_by_key_make_exclusive_kernel,
    reduce_tuple8_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e, F: key_f, G: key_g, I: key_h)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple9_by_key_pass_kernel,
    scan_tuple9_by_key_make_exclusive_kernel,
    reduce_tuple9_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e, F: key_f, G: key_g, I: key_h, J: key_i)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple10_by_key_pass_kernel,
    scan_tuple10_by_key_make_exclusive_kernel,
    reduce_tuple10_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e, F: key_f, G: key_g, I: key_h, J: key_i, K: key_j)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple11_by_key_pass_kernel,
    scan_tuple11_by_key_make_exclusive_kernel,
    reduce_tuple11_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e, F: key_f, G: key_g, I: key_h, J: key_i, K: key_j, L: key_k)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple12_by_key_pass_kernel,
    scan_tuple12_by_key_make_exclusive_kernel,
    reduce_tuple12_by_key_end_flags_kernel,
    (A: key_a, B: key_b, C: key_c, D: key_d, E: key_e, F: key_f, G: key_g, I: key_h, J: key_i, K: key_j, L: key_k, M: key_l)
);

#[cube(launch_unchecked)]
pub(crate) fn reduce_by_key_values_at_ends_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &Array<K>,
    inclusive: &Array<T>,
    init: &Array<T>,
    values: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn adjacent_difference_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
    input: &Array<T>,
    output: &mut Array<T>,
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

#[cube(launch_unchecked)]
pub(crate) fn transform_if_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    Op: UnaryOp<T, Output = T>,
    Pred: PredicateOp<S>,
>(
    input: &Array<T>,
    stencil: &Array<S>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    if unit < output.len() && Pred::apply(stencil[unit]) {
        output[unit] = Op::apply(input[unit]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn transform_binary_if_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    Op: BinaryOp<T>,
    Pred: PredicateOp<S>,
>(
    lhs: &Array<T>,
    rhs: &Array<T>,
    stencil: &Array<S>,
    output: &mut Array<T>,
) {
    let unit = UNIT_POS as usize;
    if unit < output.len() && Pred::apply(stencil[unit]) {
        output[unit] = Op::apply(lhs[unit], rhs[unit]);
    }
}
