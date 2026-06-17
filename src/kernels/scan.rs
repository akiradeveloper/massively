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
