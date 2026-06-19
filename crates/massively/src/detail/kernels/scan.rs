use crate::op::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp};
use cubecl::prelude::*;

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_pass_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    input: &[T],
    offset: &[u32],
    output: &mut [T],
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

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_block_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    input: &[T],
    len: &[u32],
    output: &mut [T],
    block_tail_keys: &mut [K],
    block_tail_values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut values = Shared::<[T]>::new_slice(cube_dim);
    let mut heads = Shared::<[u32]>::new_slice(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        values[unit] = input[global];
        valid[unit] = 1u32;
        if unit == 0usize || !KeyEq::apply(keys[global - 1usize], keys[global]) {
            heads[unit] = 1u32;
        } else {
            heads[unit] = 0u32;
        }
    } else {
        valid[unit] = 0u32;
        heads[unit] = 1u32;
        if logical_len > 0usize {
            values[unit] = input[0];
        }
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let addend = RuntimeCell::<T>::new(values[unit]);
        let addend_head = RuntimeCell::<u32>::new(0u32);
        let addend_valid = RuntimeCell::<u32>::new(0u32);
        if unit >= stride.read() {
            addend.store(values[unit - stride.read()]);
            addend_head.store(heads[unit - stride.read()]);
            addend_valid.store(valid[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
            if heads[unit] == 0u32 {
                values[unit] = Op::apply(addend.read(), values[unit]);
            }
            heads[unit] = heads[unit] | addend_head.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        output[global] = values[unit];
        if unit == cube_dim - 1usize || global == logical_len - 1usize {
            block_tail_keys[block] = keys[global];
            block_tail_values[block] = values[unit];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_add_block_prefix_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    block_tail_keys: &[K],
    block_prefixes: &[T],
    len: &[u32],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize || KeyEq::apply(keys[global - 1usize], keys[global]) {
            first_segment[unit] = 1u32;
        } else {
            first_segment[unit] = 0u32;
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if block > 0usize
        && global < logical_len
        && first_segment[unit] != 0u32
        && KeyEq::apply(block_tail_keys[block - 1usize], keys[block * cube_dim])
    {
        output[global] = Op::apply(block_prefixes[block - 1usize], output[global]);
    }
}

macro_rules! define_tuple_by_key_block_scan_kernels {
    (
        $block_name:ident,
        $add_prefix_name:ident,
        ( $( $ty:ident: $key:ident: $tail:ident ),+ )
    ) => {
        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $block_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &[$ty], )+
            input: &[T],
            len: &[u32],
            output: &mut [T],
            $( $tail: &mut [$ty], )+
            block_tail_values: &mut [T],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let block = CUBE_POS as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            let mut values = Shared::<[T]>::new_slice(cube_dim);
            let mut heads = Shared::<[u32]>::new_slice(cube_dim);
            let mut valid = Shared::<[u32]>::new_slice(cube_dim);

            if global < logical_len {
                values[unit] = input[global];
                valid[unit] = 1u32;
                if unit == 0usize || !KeyEq::apply(
                    ($( $key[global - 1usize] ),+),
                    ($( $key[global] ),+),
                ) {
                    heads[unit] = 1u32;
                } else {
                    heads[unit] = 0u32;
                }
            } else {
                valid[unit] = 0u32;
                heads[unit] = 1u32;
                if logical_len > 0usize {
                    values[unit] = input[0];
                }
            }
            sync_cube();

            let stride = RuntimeCell::<usize>::new(1usize);
            while stride.read() < cube_dim {
                let addend = RuntimeCell::<T>::new(values[unit]);
                let addend_head = RuntimeCell::<u32>::new(0u32);
                let addend_valid = RuntimeCell::<u32>::new(0u32);
                if unit >= stride.read() {
                    addend.store(values[unit - stride.read()]);
                    addend_head.store(heads[unit - stride.read()]);
                    addend_valid.store(valid[unit - stride.read()]);
                }
                sync_cube();
                if unit >= stride.read() && valid[unit] != 0u32 && addend_valid.read() != 0u32 {
                    if heads[unit] == 0u32 {
                        values[unit] = Op::apply(addend.read(), values[unit]);
                    }
                    heads[unit] = heads[unit] | addend_head.read();
                }
                sync_cube();
                stride.store(stride.read() * 2usize);
            }

            if global < logical_len {
                output[global] = values[unit];
                if unit == cube_dim - 1usize || global == logical_len - 1usize {
                    $(
                        $tail[block] = $key[global];
                    )+
                    block_tail_values[block] = values[unit];
                }
            }
        }

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $add_prefix_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &[$ty], )+
            $( $tail: &[$ty], )+
            block_prefixes: &[T],
            len: &[u32],
            output: &mut [T],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = 256usize;
            let block = CUBE_POS as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

            if global < logical_len {
                if unit == 0usize || KeyEq::apply(
                    ($( $key[global - 1usize] ),+),
                    ($( $key[global] ),+),
                ) {
                    first_segment[unit] = 1u32;
                } else {
                    first_segment[unit] = 0u32;
                }
            } else {
                first_segment[unit] = 0u32;
            }
            sync_cube();

            let stride = RuntimeCell::<usize>::new(1usize);
            while stride.read() < cube_dim {
                let previous = RuntimeCell::<u32>::new(first_segment[unit]);
                if unit >= stride.read() {
                    previous.store(first_segment[unit - stride.read()]);
                }
                sync_cube();
                if unit >= stride.read() {
                    first_segment[unit] = first_segment[unit] & previous.read();
                }
                sync_cube();
                stride.store(stride.read() * 2usize);
            }

            if block > 0usize
                && global < logical_len
                && first_segment[unit] != 0u32
                && KeyEq::apply(
                    ($( $tail[block - 1usize] ),+),
                    ($( $key[block * cube_dim] ),+),
                )
            {
                output[global] = Op::apply(block_prefixes[block - 1usize], output[global]);
            }
        }
    };
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn scan_by_key_make_exclusive_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    inclusive: &[T],
    init: &[T],
    output: &mut [T],
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

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_end_flags_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    inclusive: &[T],
    init: &[T],
    flags: &mut [u32],
    values: &mut [T],
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

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_end_flags_with_block_prefix_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    local_inclusive: &[T],
    block_tail_keys: &[K],
    block_prefixes: &[T],
    init: &[T],
    len: &[u32],
    flags: &mut [u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize || KeyEq::apply(keys[global - 1usize], keys[global]) {
            first_segment[unit] = 1u32;
        } else {
            first_segment[unit] = 0u32;
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        let is_end =
            global + 1usize == logical_len || !KeyEq::apply(keys[global], keys[global + 1usize]);
        if is_end {
            flags[global] = 1u32;
            let carry = block > 0usize
                && first_segment[unit] != 0u32
                && KeyEq::apply(block_tail_keys[block - 1usize], keys[block * cube_dim]);
            if carry {
                values[global] = Op::apply(
                    init[0],
                    Op::apply(block_prefixes[block - 1usize], local_inclusive[global]),
                );
            } else {
                values[global] = Op::apply(init[0], local_inclusive[global]);
            }
        } else {
            flags[global] = 0u32;
            values[global] = local_inclusive[global];
        }
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_values_at_ends_with_block_prefix_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    local_inclusive: &[T],
    block_tail_keys: &[K],
    block_prefixes: &[T],
    init: &[T],
    len: &[u32],
    values: &mut [T],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = 256usize;
    let block = CUBE_POS as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let mut first_segment = Shared::<[u32]>::new_slice(cube_dim);

    if global < logical_len {
        if unit == 0usize || KeyEq::apply(keys[global - 1usize], keys[global]) {
            first_segment[unit] = 1u32;
        } else {
            first_segment[unit] = 0u32;
        }
    } else {
        first_segment[unit] = 0u32;
    }
    sync_cube();

    let stride = RuntimeCell::<usize>::new(1usize);
    while stride.read() < cube_dim {
        let previous = RuntimeCell::<u32>::new(first_segment[unit]);
        if unit >= stride.read() {
            previous.store(first_segment[unit - stride.read()]);
        }
        sync_cube();
        if unit >= stride.read() {
            first_segment[unit] = first_segment[unit] & previous.read();
        }
        sync_cube();
        stride.store(stride.read() * 2usize);
    }

    if global < logical_len {
        let is_end =
            global + 1usize == logical_len || !KeyEq::apply(keys[global], keys[global + 1usize]);
        if is_end {
            let carry = block > 0usize
                && first_segment[unit] != 0u32
                && KeyEq::apply(block_tail_keys[block - 1usize], keys[block * cube_dim]);
            if carry {
                values[global] = Op::apply(
                    init[0],
                    Op::apply(block_prefixes[block - 1usize], local_inclusive[global]),
                );
            } else {
                values[global] = Op::apply(init[0], local_inclusive[global]);
            }
        } else {
            values[global] = local_inclusive[global];
        }
    }
}

macro_rules! define_tuple_by_key_scan_kernels {
    (
        $block_name:ident,
        $add_prefix_name:ident,
        $exclusive_name:ident,
        $reduce_name:ident,
        ( $( $ty:ident: $key:ident: $tail:ident ),+ )
    ) => {
        define_tuple_by_key_block_scan_kernels!(
            $block_name,
            $add_prefix_name,
            ($( $ty: $key: $tail ),+)
        );

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $exclusive_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &[$ty], )+
            inclusive: &[T],
            init: &[T],
            output: &mut [T],
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

        #[cube(launch_unchecked, explicit_define)]
        pub(crate) fn $reduce_name<
            $( $ty: CubePrimitive, )+
            T: CubePrimitive,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        >(
            $( $key: &[$ty], )+
            inclusive: &[T],
            init: &[T],
            flags: &mut [u32],
            values: &mut [T],
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
    scan_tuple2_by_key_block_kernel,
    scan_tuple2_by_key_add_block_prefix_kernel,
    scan_tuple2_by_key_make_exclusive_kernel,
    reduce_tuple2_by_key_end_flags_kernel,
    (A: key_a: block_tail_a, B: key_b: block_tail_b)
);
define_tuple_by_key_scan_kernels!(
    scan_tuple3_by_key_block_kernel,
    scan_tuple3_by_key_add_block_prefix_kernel,
    scan_tuple3_by_key_make_exclusive_kernel,
    reduce_tuple3_by_key_end_flags_kernel,
    (A: key_a: block_tail_a, B: key_b: block_tail_b, C: key_c: block_tail_c)
);

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn reduce_by_key_values_at_ends_kernel<
    K: CubePrimitive,
    T: CubePrimitive,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
>(
    keys: &[K],
    inclusive: &[T],
    init: &[T],
    values: &mut [T],
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

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn adjacent_difference_kernel<T: CubePrimitive, Op: BinaryOp<T>>(
    input: &[T],
    output: &mut [T],
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

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_if_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    Op: UnaryOp<T, Output = T>,
    Pred: PredicateOp<S>,
>(
    input: &[T],
    stencil: &[S],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < output.len() && Pred::apply(stencil[unit]) {
        output[unit] = Op::apply(input[unit]);
    }
}

#[cube(launch_unchecked, explicit_define)]
pub(crate) fn transform_binary_if_kernel<
    T: CubePrimitive,
    S: CubePrimitive,
    Op: BinaryOp<T>,
    Pred: PredicateOp<S>,
>(
    lhs: &[T],
    rhs: &[T],
    stencil: &[S],
    output: &mut [T],
) {
    let unit = UNIT_POS as usize;
    if unit < output.len() && Pred::apply(stencil[unit]) {
        output[unit] = Op::apply(lhs[unit], rhs[unit]);
    }
}
