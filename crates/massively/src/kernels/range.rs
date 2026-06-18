use crate::op::{PredicateOp, UnaryOp};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn fill_kernel<T: CubePrimitive>(
    value: &Array<T>,
    len: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < (len[0] as usize) {
        output[unit] = value[0];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn copy_kernel<T: CubePrimitive>(input: &Array<T>, output: &mut Array<T>) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = input[unit];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn concat_kernel<T: CubePrimitive>(
    left: &Array<T>,
    right: &Array<T>,
    output: &mut Array<T>,
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        if unit < left.len() {
            output[unit] = left[unit];
        } else {
            output[unit] = right[unit - left.len()];
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn indices_u32_kernel(output: &mut Array<u32>) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = unit as u32;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn device_map_kernel<T: CubePrimitive, Op: UnaryOp<T, Output = T>>(
    input: &Array<T>,
    output: &mut Array<T>,
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = Op::apply(input[unit]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn random_u32_kernel(output: &mut Array<u32>, seed: &Array<u32>) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        let state = RuntimeCell::<u32>::new(seed[0] + unit as u32);
        let round = RuntimeCell::<u32>::new(0u32);

        while round.read() < 4u32 {
            state.store(state.read() * 1664525u32 + 1013904223u32);
            round.store(round.read() + 1u32);
        }

        output[unit] = state.read();
    }
}

#[cube(launch_unchecked)]
pub(crate) fn gather_kernel<T: CubePrimitive>(
    output: &mut Array<T>,
    indices: &Array<u32>,
    input: &Array<T>,
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = input[indices[unit] as usize];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scatter_kernel<T: CubePrimitive>(
    input: &Array<T>,
    indices: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < input.len() {
        let index = indices[unit] as usize;
        output[index] = input[unit];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scatter_if_kernel<T: CubePrimitive, Pred: PredicateOp<T>>(
    input: &Array<T>,
    indices: &Array<u32>,
    output: &mut Array<T>,
) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < input.len() {
        let value = input[unit];
        if Pred::apply(value) {
            let index = indices[unit] as usize;
            output[index] = value;
        }
    }
}
