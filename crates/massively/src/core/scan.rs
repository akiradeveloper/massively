//! Reusable prefix-scan control primitives.

use cubecl::prelude::*;

use crate::{
    A13, CanonicalAlloc, CanonicalStorage, Counting, DeviceVec, Dispatch, Error, Executor,
    MStorageElement, ReadExpression, S12, StorageLayout, WritableFrom,
    allocation::PrependInput,
    eval::Eval13,
    indexed::GatherInput,
    launch::cube_count_1d,
    output::{
        LowerOutputExpression, OutputBindings, OutputExpression, PaddedOutputSlots, StageOutput,
    },
    read::{Adjacent, Env0, Env12, Env13, KernelReadSlots, LowerReadExpression, PaddedReadSlots},
    reduce::{ReductionOp, StageRead, StagedBindings},
    storage::{
        Decompose, LoadMutPadded12, LoadPadded12, MutableLeaves, PlaneShuffleLeaves, Recompose,
        SharedLeaves, StorePadded12, StorePadded12Expand,
    },
    transform::{MaterializeDispatch, materialize},
};

const BLOCK_SIZE: u32 = 256;

type FixedScanStorage<R, Item> = <Item as crate::CanonicalAlloc<R>>::CanonicalStorage;
type FixedScanRead<R, Item> =
    crate::read::FixedReassociate<<FixedScanStorage<R, Item> as CanonicalStorage<R>>::Read, Item>;
type FixedScanSlots<Item> =
    <<Item as StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots;
type FixedScanOutput<R, Item> = crate::output::ReassociatedOutput<
    <FixedScanStorage<R, Item> as CanonicalStorage<R>>::Write,
    Item,
    FixedScanSlots<Item>,
>;

#[cubecl::cube(launch_unchecked, explicit_define)]
fn u32_block_inclusive_scan_kernel(
    input: &[u32],
    len: &[u32],
    output: &mut [u32],
    block_sums: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = BLOCK_SIZE as usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    let value = RuntimeCell::<u32>::new(if global < logical_len {
        input[global]
    } else {
        0u32
    });
    let valid = RuntimeCell::<u32>::new(if global < logical_len { 1u32 } else { 0u32 });

    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left = plane_shuffle_up(value.read(), offset.read());
        let left_valid = plane_shuffle_up(valid.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
            value.store(left + value.read());
            valid.store(1u32);
        }
        offset.store(offset.read() * 2u32);
    }

    let mut plane_values = Shared::<[u32]>::new_slice(cube_dim);
    let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        plane_values[PLANE_POS as usize] = value.read();
        plane_valid[PLANE_POS as usize] = valid.read();
    }
    sync_cube();

    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let prefix = RuntimeCell::<u32>::new(plane_values[0]);
        let prefix_valid = RuntimeCell::<u32>::new(plane_valid[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            if plane_valid[index] != 0u32 {
                if prefix_valid.read() != 0u32 {
                    prefix.store(prefix.read() + plane_values[index]);
                } else {
                    prefix.store(plane_values[index]);
                    prefix_valid.store(1u32);
                }
            }
            plane_values[index] = prefix.read();
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();

    if PLANE_POS > 0u32 && valid.read() != 0u32 {
        value.store(plane_values[PLANE_POS as usize - 1usize] + value.read());
    }

    if global < logical_len {
        output[global] = value.read();
    }
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        block_sums[CUBE_POS as usize] = plane_values[plane_count as usize - 1usize];
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn u32_add_block_prefix_kernel(block_prefixes: &[u32], len: &[u32], output: &mut [u32]) {
    let block = CUBE_POS as usize;
    let global = block * BLOCK_SIZE as usize + UNIT_POS as usize;
    if block > 0usize && global < len[0] as usize {
        output[global] += block_prefixes[block - 1usize];
    }
}

#[cubecl::cube(launch_unchecked)]
fn copy_last_kernel(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = input[input.len() - 1usize];
    }
}

#[cfg(any())]
mod unused_scalar_scan_kernels {
    use super::*;

    macro_rules! define_scalar_scan_kernel {
    ($name:ident,$eval:ident,$method:ident; $( $leaf:ident:$slot:ident ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubePrimitive,
            $( $leaf: CubePrimitive, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            output_offset: &[u32],
            output: &mut [Item],
            block_sums: &mut [Item],
        ) {
            let unit = UNIT_POS as usize;
            let block = CUBE_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            let safe_global = if global < logical_len { global } else { 0usize };
            let value = RuntimeCell::<Item>::new(
                Expr::$method($( $slot, )+ read_offsets, safe_global),
            );
            let valid = RuntimeCell::<u32>::new(
                if global < logical_len { 1u32 } else { 0u32 },
            );

            let offset = RuntimeCell::<u32>::new(1u32);
            while offset.read() < PLANE_DIM {
                let left = plane_shuffle_up(value.read(), offset.read());
                let left_valid = plane_shuffle_up(valid.read(), offset.read());
                if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
                    if valid.read() != 0u32 {
                        value.store(Op::apply(left, value.read()));
                    } else {
                        value.store(left);
                        valid.store(1u32);
                    }
                }
                offset.store(offset.read() * 2u32);
            }

            let mut plane_values = Shared::<[Item]>::new_slice(cube_dim);
            let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
            if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
                plane_values[PLANE_POS as usize] = value.read();
                plane_valid[PLANE_POS as usize] = valid.read();
            }
            sync_cube();

            if unit == 0usize {
                let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                let prefix = RuntimeCell::<Item>::new(plane_values[0]);
                let prefix_valid = RuntimeCell::<u32>::new(plane_valid[0]);
                let plane = RuntimeCell::<u32>::new(1u32);
                while plane.read() < plane_count {
                    let index = plane.read() as usize;
                    if plane_valid[index] != 0u32 {
                        if prefix_valid.read() != 0u32 {
                            prefix.store(Op::apply(prefix.read(), plane_values[index]));
                        } else {
                            prefix.store(plane_values[index]);
                            prefix_valid.store(1u32);
                        }
                    }
                    plane_values[index] = prefix.read();
                    plane.store(plane.read() + 1u32);
                }
            }
            sync_cube();

            if PLANE_POS > 0u32 && valid.read() != 0u32 {
                value.store(Op::apply(
                    plane_values[PLANE_POS as usize - 1usize],
                    value.read(),
                ));
            }

            if global < logical_len {
                output[output_offset[0] as usize + global] = value.read();
            }
            if unit == 0usize {
                let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                block_sums[block] = plane_values[plane_count as usize - 1usize];
            }
        }
    };
}

    define_scalar_scan_kernel!(scalar_scan_a1,Eval1,eval1; L0:slot0);
    define_scalar_scan_kernel!(scalar_scan_a2,Eval2,eval2; L0:slot0,L1:slot1);
    define_scalar_scan_kernel!(scalar_scan_a3,Eval3,eval3; L0:slot0,L1:slot1,L2:slot2);
    define_scalar_scan_kernel!(scalar_scan_a4,Eval4,eval4; L0:slot0,L1:slot1,L2:slot2,L3:slot3);
    define_scalar_scan_kernel!(scalar_scan_a5,Eval5,eval5; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4);
    define_scalar_scan_kernel!(scalar_scan_a6,Eval6,eval6; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5);
    define_scalar_scan_kernel!(scalar_scan_a7,Eval7,eval7; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6);
    define_scalar_scan_kernel!(scalar_scan_a8,Eval8,eval8; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7);
    define_scalar_scan_kernel!(scalar_scan_a9,Eval9,eval9; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8);
    define_scalar_scan_kernel!(scalar_scan_a10,Eval10,eval10; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9);
    define_scalar_scan_kernel!(scalar_scan_a11,Eval11,eval11; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10);
    define_scalar_scan_kernel!(scalar_scan_a12,Eval12,eval12; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11);
    define_scalar_scan_kernel!(scalar_scan_a13,Eval13,eval13; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12);
}

#[cubecl::cube]
#[allow(clippy::too_many_arguments)]
fn scan_value_padded12<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
    value: Item,
    valid_value: u32,
    logical_len: usize,
    global: usize,
    unit: usize,
    block: usize,
    zero_offsets: &[u32],
    output_offsets: &[u32],
    out0: &mut [O0],
    out1: &mut [O1],
    out2: &mut [O2],
    out3: &mut [O3],
    out4: &mut [O4],
    out5: &mut [O5],
    out6: &mut [O6],
    out7: &mut [O7],
    out8: &mut [O8],
    out9: &mut [O9],
    out10: &mut [O10],
    out11: &mut [O11],
    sum0: &mut [O0],
    sum1: &mut [O1],
    sum2: &mut [O2],
    sum3: &mut [O3],
    sum4: &mut [O4],
    sum5: &mut [O5],
    sum6: &mut [O6],
    sum7: &mut [O7],
    sum8: &mut [O8],
    sum9: &mut [O9],
    sum10: &mut [O10],
    sum11: &mut [O11],
) where
    Item: CubeType + Send + Sync + 'static,
    O0: CubePrimitive,
    O1: CubePrimitive,
    O2: CubePrimitive,
    O3: CubePrimitive,
    O4: CubePrimitive,
    O5: CubePrimitive,
    O6: CubePrimitive,
    O7: CubePrimitive,
    O8: CubePrimitive,
    O9: CubePrimitive,
    O10: CubePrimitive,
    O11: CubePrimitive,
    Leaves: SharedLeaves
        + MutableLeaves
        + PlaneShuffleLeaves
        + StorePadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cube_dim = BLOCK_SIZE as usize;
    let mut shared = Leaves::new_shared(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);
    let cells = Leaves::into_cells(Layout::decompose(value));
    let is_valid = RuntimeCell::<u32>::new(valid_value);
    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left_cells = Leaves::into_cells(Leaves::shuffle_leaves_up(
            Leaves::read(&cells),
            offset.read(),
        ));
        let left_valid = plane_shuffle_up(is_valid.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
            if is_valid.read() != 0u32 {
                let combined = Layout::decompose(Op::apply(
                    Layout::recompose(Leaves::read(&left_cells)),
                    Layout::recompose(Leaves::read(&cells)),
                ));
                Leaves::store(&cells, combined);
            } else {
                Leaves::store(&cells, Leaves::read(&left_cells));
                is_valid.store(1u32);
            }
        }
        offset.store(offset.read() * 2u32);
    }
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        Leaves::store_shared(Leaves::read(&cells), &mut shared, PLANE_POS as usize);
        valid[PLANE_POS as usize] = is_valid.read();
    }
    sync_cube();
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let plane_cells = Leaves::into_cells(Leaves::load_shared(&shared, 0usize));
        let plane_is_valid = RuntimeCell::<u32>::new(valid[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            if valid[index] != 0u32 {
                if plane_is_valid.read() != 0u32 {
                    let combined = Layout::decompose(Op::apply(
                        Layout::recompose(Leaves::read(&plane_cells)),
                        Layout::recompose(Leaves::load_shared(&shared, index)),
                    ));
                    Leaves::store(&plane_cells, combined);
                } else {
                    Leaves::store(&plane_cells, Leaves::load_shared(&shared, index));
                    plane_is_valid.store(1u32);
                }
            }
            Leaves::store_shared(Leaves::read(&plane_cells), &mut shared, index);
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();
    if PLANE_POS > 0u32 && is_valid.read() != 0u32 {
        let prefix = Leaves::load_shared(&shared, PLANE_POS as usize - 1usize);
        let combined = Layout::decompose(Op::apply(
            Layout::recompose(prefix),
            Layout::recompose(Leaves::read(&cells)),
        ));
        Leaves::store(&cells, combined);
    }
    if global < logical_len {
        Leaves::read(&cells).store_padded(
            out0,
            out1,
            out2,
            out3,
            out4,
            out5,
            out6,
            out7,
            out8,
            out9,
            out10,
            out11,
            output_offsets,
            global,
        );
    }
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        Leaves::load_shared(&shared, plane_count as usize - 1usize).store_padded(
            sum0,
            sum1,
            sum2,
            sum3,
            sum4,
            sum5,
            sum6,
            sum7,
            sum8,
            sum9,
            sum10,
            sum11,
            zero_offsets,
            block,
        );
    }
}

macro_rules! define_padded_scan_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            O0: CubePrimitive, O1: CubePrimitive, O2: CubePrimitive, O3: CubePrimitive,
            O4: CubePrimitive, O5: CubePrimitive, O6: CubePrimitive, O7: CubePrimitive,
            O8: CubePrimitive, O9: CubePrimitive, O10: CubePrimitive, O11: CubePrimitive,
            Leaves: SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + Send + Sync + 'static,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            zero_offsets: &[u32],
            output_offsets: &[u32],
            out0: &mut [O0], out1: &mut [O1], out2: &mut [O2], out3: &mut [O3],
            out4: &mut [O4], out5: &mut [O5], out6: &mut [O6], out7: &mut [O7],
            out8: &mut [O8], out9: &mut [O9], out10: &mut [O10], out11: &mut [O11],
            sum0: &mut [O0], sum1: &mut [O1], sum2: &mut [O2], sum3: &mut [O3],
            sum4: &mut [O4], sum5: &mut [O5], sum6: &mut [O6], sum7: &mut [O7],
            sum8: &mut [O8], sum9: &mut [O9], sum10: &mut [O10], sum11: &mut [O11],
        ) {
            let unit = UNIT_POS as usize;
            let block = CUBE_POS as usize;
            let global = block * BLOCK_SIZE as usize + unit;
            let logical_len = len[0] as usize;
            let safe_global = if global < logical_len { global } else { 0usize };
            scan_value_padded12::<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
                Expr::$method($( $slot, )+ read_offsets, safe_global),
                if global < logical_len { 1u32 } else { 0u32 },
                logical_len, global, unit, block, zero_offsets, output_offsets,
                out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
                sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11,
            );
        }
    };
}

define_padded_scan_kernel!(padded_scan_a13,Eval13,eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

#[cfg(any())]
mod legacy_storage_scan {
    use super::*;

    macro_rules! define_storage_scan_kernel {
    (
        $name:ident,$load_trait:ident,$store_trait:ident,$select_trait:ident;
        [$( $out_ty:ident:$input:ident:$output:ident:$shared:ident:$block_sum:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $out_ty: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>
                + $select_trait<$( $out_ty ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $input: &[$out_ty], )+
            len: &[u32],
            offsets: &[u32],
            $( $output: &mut [$out_ty], )+
            $( $block_sum: &mut [$out_ty], )+
        ) {
            let unit = UNIT_POS as usize;
            let block = CUBE_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            $( let mut $shared = Shared::<[$out_ty]>::new_slice(cube_dim); )+
            let mut valid = Shared::<[u32]>::new_slice(cube_dim);

            let initial = Leaves::load($( $input, )+ offsets, 0);
            initial.store($( &mut $shared, )+ offsets, unit);
            valid[unit] = 0u32;
            if global < logical_len {
                let item = Leaves::load($( $input, )+ offsets, global);
                item.store($( &mut $shared, )+ offsets, unit);
                valid[unit] = 1u32;
            }
            sync_cube();

            let stride = RuntimeCell::<usize>::new(1usize);
            while stride.read() < cube_dim {
                let current = Layout::recompose(Leaves::load(
                    $( &$shared, )+ offsets, unit,
                ));
                let left_index = if unit >= stride.read() { unit - stride.read() } else { unit };
                let left = Layout::recompose(Leaves::load(
                    $( &$shared, )+ offsets, left_index,
                ));
                let left_valid = if unit >= stride.read() { valid[left_index] } else { 0u32 };
                let combined = Layout::decompose(Op::apply(left, current));
                sync_cube();
                combined.select_store(
                    left_valid, $( $shared[unit], )+ $( &mut $shared, )+ offsets, unit,
                );
                sync_cube();
                stride.store(stride.read() * 2usize);
            }

            if global < logical_len {
                Leaves::load($( &$shared, )+ offsets, unit).store(
                    $( $output, )+ offsets, global,
                );
            }
            if unit == 0usize {
                let block_start = block * cube_dim;
                if block_start < logical_len {
                    let remaining = logical_len - block_start;
                    let last = if remaining < cube_dim { remaining - 1usize } else { cube_dim - 1usize };
                    Leaves::load($( &$shared, )+ offsets, last).store(
                        $( $block_sum, )+ offsets, block,
                    );
                }
            }
        }
    };
}

    define_storage_scan_kernel!(storage_scan_s2,LoadLeaves2,StoreLeaves2,SelectStoreLeaves2; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1]);
    define_storage_scan_kernel!(storage_scan_s3,LoadLeaves3,StoreLeaves3,SelectStoreLeaves3; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2]);
    define_storage_scan_kernel!(storage_scan_s4,LoadLeaves4,StoreLeaves4,SelectStoreLeaves4; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3]);
    define_storage_scan_kernel!(storage_scan_s5,LoadLeaves5,StoreLeaves5,SelectStoreLeaves5; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4]);
    define_storage_scan_kernel!(storage_scan_s6,LoadLeaves6,StoreLeaves6,SelectStoreLeaves6; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5]);
    define_storage_scan_kernel!(storage_scan_s7,LoadLeaves7,StoreLeaves7,SelectStoreLeaves7; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5,O6:in6:out6:shared6:sum6]);
    define_storage_scan_kernel!(storage_scan_s8,LoadLeaves8,StoreLeaves8,SelectStoreLeaves8; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5,O6:in6:out6:shared6:sum6,O7:in7:out7:shared7:sum7]);
    define_storage_scan_kernel!(storage_scan_s9,LoadLeaves9,StoreLeaves9,SelectStoreLeaves9; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5,O6:in6:out6:shared6:sum6,O7:in7:out7:shared7:sum7,O8:in8:out8:shared8:sum8]);
    define_storage_scan_kernel!(storage_scan_s10,LoadLeaves10,StoreLeaves10,SelectStoreLeaves10; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5,O6:in6:out6:shared6:sum6,O7:in7:out7:shared7:sum7,O8:in8:out8:shared8:sum8,O9:in9:out9:shared9:sum9]);
    define_storage_scan_kernel!(storage_scan_s11,LoadLeaves11,StoreLeaves11,SelectStoreLeaves11; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5,O6:in6:out6:shared6:sum6,O7:in7:out7:shared7:sum7,O8:in8:out8:shared8:sum8,O9:in9:out9:shared9:sum9,O10:in10:out10:shared10:sum10]);
    define_storage_scan_kernel!(storage_scan_s12,LoadLeaves12,StoreLeaves12,SelectStoreLeaves12; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5,O6:in6:out6:shared6:sum6,O7:in7:out7:shared7:sum7,O8:in8:out8:shared8:sum8,O9:in9:out9:shared9:sum9,O10:in10:out10:shared10:sum10,O11:in11:out11:shared11:sum11]);

    macro_rules! define_multi_add_prefix_kernel {
    (
        $name:ident,$load_trait:ident,$load_mut_trait:ident,$store_trait:ident;
        [$( $out_ty:ident:$prefix:ident:$output:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $out_ty: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$( $out_ty ),+>
                + $load_mut_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $prefix: &[$out_ty], )+
            len: &[u32],
            prefix_offsets: &[u32],
            output_offsets: &[u32],
            $( $output: &mut [$out_ty], )+
        ) {
            let block = CUBE_POS as usize;
            let global = block * BLOCK_SIZE as usize + UNIT_POS as usize;
            if block > 0usize && global < len[0] as usize {
                let prefix = Layout::recompose(Leaves::load(
                    $( $prefix, )+ prefix_offsets, block - 1usize,
                ));
                let current = Layout::recompose(Leaves::load_mut(
                    $( $output, )+ output_offsets, global,
                ));
                Layout::decompose(Op::apply(prefix, current)).store(
                    $( $output, )+ output_offsets, global,
                );
            }
        }
    };
}

    define_multi_add_prefix_kernel!(multi_add_prefix_s2,LoadLeaves2,LoadMutLeaves2,StoreLeaves2; [O0:prefix0:out0,O1:prefix1:out1]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s3,LoadLeaves3,LoadMutLeaves3,StoreLeaves3; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s4,LoadLeaves4,LoadMutLeaves4,StoreLeaves4; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s5,LoadLeaves5,LoadMutLeaves5,StoreLeaves5; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s6,LoadLeaves6,LoadMutLeaves6,StoreLeaves6; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s7,LoadLeaves7,LoadMutLeaves7,StoreLeaves7; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5,O6:prefix6:out6]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s8,LoadLeaves8,LoadMutLeaves8,StoreLeaves8; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5,O6:prefix6:out6,O7:prefix7:out7]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s9,LoadLeaves9,LoadMutLeaves9,StoreLeaves9; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5,O6:prefix6:out6,O7:prefix7:out7,O8:prefix8:out8]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s10,LoadLeaves10,LoadMutLeaves10,StoreLeaves10; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5,O6:prefix6:out6,O7:prefix7:out7,O8:prefix8:out8,O9:prefix9:out9]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s11,LoadLeaves11,LoadMutLeaves11,StoreLeaves11; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5,O6:prefix6:out6,O7:prefix7:out7,O8:prefix8:out8,O9:prefix9:out9,O10:prefix10:out10]);
    define_multi_add_prefix_kernel!(multi_add_prefix_s12,LoadLeaves12,LoadMutLeaves12,StoreLeaves12; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5,O6:prefix6:out6,O7:prefix7:out7,O8:prefix8:out8,O9:prefix9:out9,O10:prefix10:out10,O11:prefix11:out11]);

    #[cubecl::cube(launch_unchecked, explicit_define)]
    fn scalar_add_block_prefix_kernel<Item: CubePrimitive, Op: ReductionOp<Item>>(
        block_prefixes: &[Item],
        len: &[u32],
        output_offset: &[u32],
        output: &mut [Item],
    ) {
        let block = CUBE_POS as usize;
        let global = block * BLOCK_SIZE as usize + UNIT_POS as usize;
        if block > 0usize && global < len[0] as usize {
            let index = output_offset[0] as usize + global;
            output[index] = Op::apply(block_prefixes[block - 1usize], output[index]);
        }
    }

    macro_rules! define_storage_scan_host {
    (
        $name:ident,$arity:ty,$leaves:ty,$scan_kernel:ident,$prefix_kernel:ident,
        $load_trait:ident,$load_mut_trait:ident,$store_trait:ident,$select_trait:ident;
        [$( $out_ty:ident:$input:ident:$output:ident:$block_sum:ident:$prefix:ident ),+]
    ) => {
        fn $name<R, Item, Op, $( $out_ty ),+>(
            exec: &Executor<R>,
            $( $input: &DeviceVec<R, $out_ty>, )+
            $( $output: &DeviceVec<R, $out_ty>, )+
        ) -> Result<(), Error>
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $arity, StorageLeaves = $leaves>
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout: Recompose<Item, Leaves = $leaves>,
            Op: ReductionOp<Item>,
            $( $out_ty: MStorageElement, )+
            $leaves: $load_trait<$( $out_ty ),+>
                + $load_mut_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>
                + $select_trait<$( $out_ty ),+>
                + Send
                + Sync
                + 'static,
        {
            let len = define_storage_scan_host!(@first_len $( $input ),+);
            if len == 0 {
                return Ok(());
            }
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let blocks = len.div_ceil(BLOCK_SIZE as usize);
            $( let $block_sum = exec.alloc_column::<$out_ty>(blocks); )+
            let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
            let offsets = vec![$( { let _ = stringify!($input); 0u32 } ),+];
            let offsets_handle = exec.client().create_from_slice(u32::as_bytes(&offsets));
            let count = cube_count_1d(blocks)?;
            unsafe {
                $scan_kernel::launch_unchecked::<
                    Item,
                    $( $out_ty, )+
                    $leaves,
                    Item::DeviceLayout,
                    Op,
                    R,
                >(
                    exec.client(),
                    count.clone(),
                    CubeDim::new_1d(BLOCK_SIZE),
                    $( BufferArg::from_raw_parts($input.handle.clone(), len), )+
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(offsets_handle.clone(), offsets.len()),
                    $( BufferArg::from_raw_parts($output.handle.clone(), len), )+
                    $( BufferArg::from_raw_parts($block_sum.handle.clone(), blocks), )+
                );
            }
            if blocks > 1 {
                $( let $prefix = exec.alloc_column::<$out_ty>(blocks); )+
                $name::<R, Item, Op, $( $out_ty ),+>(
                    exec,
                    $( &$block_sum, )+
                    $( &$prefix, )+
                )?;
                unsafe {
                    $prefix_kernel::launch_unchecked::<
                        Item,
                        $( $out_ty, )+
                        $leaves,
                        Item::DeviceLayout,
                        Op,
                        R,
                    >(
                        exec.client(),
                        count,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts($prefix.handle.clone(), blocks), )+
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(offsets_handle.clone(), offsets.len()),
                        BufferArg::from_raw_parts(offsets_handle, offsets.len()),
                        $( BufferArg::from_raw_parts($output.handle.clone(), len), )+
                    );
                }
            }
            Ok(())
        }
    };
    (@first_len $first:ident $(, $rest:ident)*) => { $first.len() };
}

    define_storage_scan_host!(scan_storage2,S2,More<O0,Last<O1>>,storage_scan_s2,multi_add_prefix_s2,LoadLeaves2,LoadMutLeaves2,StoreLeaves2,SelectStoreLeaves2; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1]);
    define_storage_scan_host!(scan_storage3,S3,More<O0,More<O1,Last<O2>>>,storage_scan_s3,multi_add_prefix_s3,LoadLeaves3,LoadMutLeaves3,StoreLeaves3,SelectStoreLeaves3; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2]);
    define_storage_scan_host!(scan_storage4,S4,More<O0,More<O1,More<O2,Last<O3>>>>,storage_scan_s4,multi_add_prefix_s4,LoadLeaves4,LoadMutLeaves4,StoreLeaves4,SelectStoreLeaves4; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3]);
    define_storage_scan_host!(scan_storage5,S5,More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>,storage_scan_s5,multi_add_prefix_s5,LoadLeaves5,LoadMutLeaves5,StoreLeaves5,SelectStoreLeaves5; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4]);
    define_storage_scan_host!(scan_storage6,S6,More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>,storage_scan_s6,multi_add_prefix_s6,LoadLeaves6,LoadMutLeaves6,StoreLeaves6,SelectStoreLeaves6; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5]);
    define_storage_scan_host!(scan_storage7,S7,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>,storage_scan_s7,multi_add_prefix_s7,LoadLeaves7,LoadMutLeaves7,StoreLeaves7,SelectStoreLeaves7; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5,O6:input6:output6:sum6:prefix6]);
    define_storage_scan_host!(scan_storage8,S8,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,Last<O7>>>>>>>>,storage_scan_s8,multi_add_prefix_s8,LoadLeaves8,LoadMutLeaves8,StoreLeaves8,SelectStoreLeaves8; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5,O6:input6:output6:sum6:prefix6,O7:input7:output7:sum7:prefix7]);
    define_storage_scan_host!(scan_storage9,S9,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,Last<O8>>>>>>>>>,storage_scan_s9,multi_add_prefix_s9,LoadLeaves9,LoadMutLeaves9,StoreLeaves9,SelectStoreLeaves9; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5,O6:input6:output6:sum6:prefix6,O7:input7:output7:sum7:prefix7,O8:input8:output8:sum8:prefix8]);
    define_storage_scan_host!(scan_storage10,S10,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,Last<O9>>>>>>>>>>,storage_scan_s10,multi_add_prefix_s10,LoadLeaves10,LoadMutLeaves10,StoreLeaves10,SelectStoreLeaves10; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5,O6:input6:output6:sum6:prefix6,O7:input7:output7:sum7:prefix7,O8:input8:output8:sum8:prefix8,O9:input9:output9:sum9:prefix9]);
    define_storage_scan_host!(scan_storage11,S11,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,Last<O10>>>>>>>>>>>,storage_scan_s11,multi_add_prefix_s11,LoadLeaves11,LoadMutLeaves11,StoreLeaves11,SelectStoreLeaves11; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5,O6:input6:output6:sum6:prefix6,O7:input7:output7:sum7:prefix7,O8:input8:output8:sum8:prefix8,O9:input9:output9:sum9:prefix9,O10:input10:output10:sum10:prefix10]);
    define_storage_scan_host!(scan_storage12,S12,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,More<O10,Last<O11>>>>>>>>>>>>,storage_scan_s12,multi_add_prefix_s12,LoadLeaves12,LoadMutLeaves12,StoreLeaves12,SelectStoreLeaves12; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5,O6:input6:output6:sum6:prefix6,O7:input7:output7:sum7:prefix7,O8:input8:output8:sum8:prefix8,O9:input9:output9:sum9:prefix9,O10:input10:output10:sum10:prefix10,O11:input11:output11:sum11:prefix11]);
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn add_block_prefix_padded12<
    Item: CubeType + Send + Sync + 'static,
    O0: CubePrimitive,
    O1: CubePrimitive,
    O2: CubePrimitive,
    O3: CubePrimitive,
    O4: CubePrimitive,
    O5: CubePrimitive,
    O6: CubePrimitive,
    O7: CubePrimitive,
    O8: CubePrimitive,
    O9: CubePrimitive,
    O10: CubePrimitive,
    O11: CubePrimitive,
    Leaves: LoadPadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + LoadMutPadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    prefix0: &[O0],
    prefix1: &[O1],
    prefix2: &[O2],
    prefix3: &[O3],
    prefix4: &[O4],
    prefix5: &[O5],
    prefix6: &[O6],
    prefix7: &[O7],
    prefix8: &[O8],
    prefix9: &[O9],
    prefix10: &[O10],
    prefix11: &[O11],
    len: &[u32],
    prefix_offsets: &[u32],
    output_offsets: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
) {
    let block = CUBE_POS as usize;
    let index = block * BLOCK_SIZE as usize + UNIT_POS as usize;
    if block > 0usize && index < len[0] as usize {
        let prefix = Layout::recompose(Leaves::load_padded(
            prefix0,
            prefix1,
            prefix2,
            prefix3,
            prefix4,
            prefix5,
            prefix6,
            prefix7,
            prefix8,
            prefix9,
            prefix10,
            prefix11,
            prefix_offsets,
            block - 1usize,
        ));
        let value = Layout::recompose(Leaves::load_mut_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            index,
        ));
        Layout::decompose(Op::apply(prefix, value)).store_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            index,
        );
    }
}

#[doc(hidden)]
pub trait InclusiveScanDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Op>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, input: &Input, op: Op, output: &Output) -> Result<(), Error>;
}

#[doc(hidden)]
pub trait InclusiveScanPassDispatch<R, Input, Output, Partials, Item, ReadSlots, WriteSlots, Op>
where
    R: Runtime,
{
    fn run_pass(
        exec: &Executor<R>,
        input: &Input,
        output: &Output,
        partials: &Partials,
    ) -> Result<(), Error>;
}

#[cfg(any())]
mod legacy_output_host {
    use super::*;

    pub trait ScanOutputHost<R: Runtime, Item: StorageLayout>: PaddedOutputSlots {
        type Partials;

        fn allocate(exec: &Executor<R>, blocks: usize) -> (Self::Partials, OutputBindings);

        fn finish<Op: ReductionOp<Item>>(
            exec: &Executor<R>,
            partials: &Self::Partials,
            op: Op,
            blocks: usize,
            count: CubeCount,
            len_handle: cubecl::server::Handle,
            zero_offsets: cubecl::server::Handle,
            output_offsets: cubecl::server::Handle,
            output: &OutputBindings,
        ) -> Result<(), Error>;
    }

    #[cfg(any())]
    impl<R, Item> ScanOutputHost<R, Item> for Env1<Item>
    where
        R: Runtime,
        Item: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<Item>>,
    {
        type Partials = DeviceVec<R, Item>;

        fn allocate(exec: &Executor<R>, blocks: usize) -> (Self::Partials, OutputBindings) {
            let partials = exec.alloc_column::<Item>(blocks);
            let mut bindings = OutputBindings::new();
            bindings.push(partials.handle.clone(), blocks, 0);
            bindings.pad_to_twelve(exec.client());
            (partials, bindings)
        }

        fn finish<Op: ReductionOp<Item>>(
            exec: &Executor<R>,
            partials: &Self::Partials,
            op: Op,
            blocks: usize,
            count: CubeCount,
            len_handle: cubecl::server::Handle,
            _zero_offsets: cubecl::server::Handle,
            output_offsets: cubecl::server::Handle,
            output: &OutputBindings,
        ) -> Result<(), Error> {
            if blocks > 1 {
                let prefixes = exec.alloc_column::<Item>(blocks);
                scan_scalar_column::<R, Item, Op>(exec, partials, op, &prefixes)?;
                unsafe {
                    scalar_add_block_prefix_kernel::launch_unchecked::<Item, Op, R>(
                        exec.client(),
                        count,
                        CubeDim::new_1d(BLOCK_SIZE),
                        BufferArg::from_raw_parts(prefixes.handle.clone(), blocks),
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(output_offsets, output.offsets.len()),
                        BufferArg::from_raw_parts(output.slots[0].0.clone(), output.slots[0].1),
                    );
                }
            }
            Ok(())
        }
    }

    macro_rules! impl_scan_output_host {
    (
        $env:ty, $arity:ty, $leaves:ty, $host_scan:ident, $prefix_kernel:ident;
        [$( $out:ident:$index:tt:$partial:ident:$prefix:ident ),+]
    ) => {
        impl<R, Item, $( $out ),+> ScanOutputHost<R, Item> for $env
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $arity, StorageLeaves = $leaves>
                + Send + Sync + 'static,
            Item::DeviceLayout: Recompose<Item, Leaves = $leaves>,
            $( $out: MStorageElement, )+
            $leaves: Send + Sync + 'static,
        {
            type Partials = ($( DeviceVec<R, $out>, )+);

            fn allocate(exec: &Executor<R>, blocks: usize) -> (Self::Partials, OutputBindings) {
                $( let $partial = exec.alloc_column::<$out>(blocks); )+
                let mut bindings = OutputBindings::new();
                $( bindings.push($partial.handle.clone(), blocks, 0); )+
                bindings.pad_to_twelve(exec.client());
                (($( $partial, )+), bindings)
            }

            fn finish<Op: ReductionOp<Item>>(
                exec: &Executor<R>,
                partials: &Self::Partials,
                _op: Op,
                blocks: usize,
                count: CubeCount,
                len_handle: cubecl::server::Handle,
                zero_offsets: cubecl::server::Handle,
                output_offsets: cubecl::server::Handle,
                output: &OutputBindings,
            ) -> Result<(), Error> {
                if blocks > 1 {
                    $( let $prefix = exec.alloc_column::<$out>(blocks); )+
                    $host_scan::<R, Item, Op, $( $out ),+>(
                        exec,
                        $( &partials.$index, )+
                        $( &$prefix, )+
                    )?;
                    unsafe {
                        $prefix_kernel::launch_unchecked::<
                            Item, $( $out, )+ $leaves, Item::DeviceLayout, Op, R,
                        >(
                            exec.client(),
                            count,
                            CubeDim::new_1d(BLOCK_SIZE),
                            $( BufferArg::from_raw_parts($prefix.handle.clone(), blocks), )+
                            BufferArg::from_raw_parts(len_handle, 1),
                            BufferArg::from_raw_parts(zero_offsets, output.offsets.len()),
                            BufferArg::from_raw_parts(output_offsets, output.offsets.len()),
                            $( BufferArg::from_raw_parts(output.slots[$index].0.clone(), output.slots[$index].1), )+
                        );
                    }
                }
                Ok(())
            }
        }
    };
}

    #[cfg(any())]
    impl_scan_output_host!(Env2<O0,O1>,S2,More<O0,Last<O1>>,scan_storage2,multi_add_prefix_s2; [O0:0:p0:q0,O1:1:p1:q1]);
    #[cfg(any())]
    impl_scan_output_host!(Env3<O0,O1,O2>,S3,More<O0,More<O1,Last<O2>>>,scan_storage3,multi_add_prefix_s3; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2]);
    #[cfg(any())]
    impl_scan_output_host!(Env4<O0,O1,O2,O3>,S4,More<O0,More<O1,More<O2,Last<O3>>>>,scan_storage4,multi_add_prefix_s4; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3]);
    #[cfg(any())]
    impl_scan_output_host!(Env5<O0,O1,O2,O3,O4>,S5,More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>,scan_storage5,multi_add_prefix_s5; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4]);
    #[cfg(any())]
    impl_scan_output_host!(Env6<O0,O1,O2,O3,O4,O5>,S6,More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>,scan_storage6,multi_add_prefix_s6; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4,O5:5:p5:q5]);
    #[cfg(any())]
    impl_scan_output_host!(Env7<O0,O1,O2,O3,O4,O5,O6>,S7,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>,scan_storage7,multi_add_prefix_s7; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4,O5:5:p5:q5,O6:6:p6:q6]);
    #[cfg(any())]
    impl_scan_output_host!(Env8<O0,O1,O2,O3,O4,O5,O6,O7>,S8,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,Last<O7>>>>>>>>,scan_storage8,multi_add_prefix_s8; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4,O5:5:p5:q5,O6:6:p6:q6,O7:7:p7:q7]);
    #[cfg(any())]
    impl_scan_output_host!(Env9<O0,O1,O2,O3,O4,O5,O6,O7,O8>,S9,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,Last<O8>>>>>>>>>,scan_storage9,multi_add_prefix_s9; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4,O5:5:p5:q5,O6:6:p6:q6,O7:7:p7:q7,O8:8:p8:q8]);
    #[cfg(any())]
    impl_scan_output_host!(Env10<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9>,S10,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,Last<O9>>>>>>>>>>,scan_storage10,multi_add_prefix_s10; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4,O5:5:p5:q5,O6:6:p6:q6,O7:7:p7:q7,O8:8:p8:q8,O9:9:p9:q9]);
    #[cfg(any())]
    impl_scan_output_host!(Env11<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10>,S11,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,Last<O10>>>>>>>>>>>,scan_storage11,multi_add_prefix_s11; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4,O5:5:p5:q5,O6:6:p6:q6,O7:7:p7:q7,O8:8:p8:q8,O9:9:p9:q9,O10:10:p10:q10]);
    #[cfg(any())]
    impl_scan_output_host!(Env12<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11>,S12,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,More<O10,Last<O11>>>>>>>>>>>>,scan_storage12,multi_add_prefix_s12; [O0:0:p0:q0,O1:1:p1:q1,O2:2:p2:q2,O3:3:p3:q3,O4:4:p4:q4,O5:5:p5:q5,O6:6:p6:q6,O7:7:p7:q7,O8:8:p8:q8,O9:9:p9:q9,O10:10:p10:q10,O11:11:p11:q11]);
}

macro_rules! impl_padded_scan_dispatch {
    (
        $arity:ty, $eval:ident, $kernel:ident, $env:ty;
        [$( $leaf:ident:$index:literal ),+]
    ) => {
        impl<
            R, Input, Output, Partials, Item, Op,
            O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
            $( $leaf ),+
        > InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            Partials,
            Item,
            $env,
            Env12<O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11>,
            Op,
        >
            for Dispatch<$arity, S12>
        where
            R: Runtime,
            Item: StorageLayout + Send + Sync + 'static,
            Item::DeviceLayout: Recompose<Item, Leaves = Item::StorageLeaves>,
            Op: ReductionOp<Item>,
            $( $leaf: MStorageElement, )+
            O0: MStorageElement,
            O1: MStorageElement,
            O2: MStorageElement,
            O3: MStorageElement,
            O4: MStorageElement,
            O5: MStorageElement,
            O6: MStorageElement,
            O7: MStorageElement,
            O8: MStorageElement,
            O9: MStorageElement,
            O10: MStorageElement,
            O11: MStorageElement,
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
            Output::Item: WritableFrom<Item>,
            Output::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Partials: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
            Partials::Item: WritableFrom<Item>,
            Partials::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Item::StorageLeaves: SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + Send + Sync + 'static,
        {
            fn run_pass(
                exec: &Executor<R>,
                input: &Input,
                output: &Output,
                partials: &Partials,
            ) -> Result<(), Error> {
                let len = input.logical_len()?;
                let output_len = output.logical_len()?;
                if output_len != len {
                    return Err(Error::LengthMismatch { left: len, right: output_len });
                }
                if len == 0 {
                    return Ok(());
                }
                let blocks = len.div_ceil(BLOCK_SIZE as usize);
                let partial_len = partials.logical_len()?;
                if partial_len != blocks {
                    return Err(Error::LengthMismatch { left: blocks, right: partial_len });
                }
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                reads.pad_to_thirteen(exec.client());
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                writes.pad_to_twelve(exec.client());
                let mut partial_bindings = OutputBindings::new();
                partials.stage_output(exec.id(), &mut partial_bindings)?;
                partial_bindings.pad_to_twelve(exec.client());
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let zero_values = [0u32; 12];
                let zero_offsets = exec.client().create_from_slice(u32::as_bytes(&zero_values));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[
                    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?
                ]));
                unsafe {
                    $kernel::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
                        Item::StorageLeaves,
                        Item::DeviceLayout,
                        Input::DeviceExpr,
                        Op,
                        R,
                    >(
                        exec.client(),
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.clone(), 1),
                        BufferArg::from_raw_parts(zero_offsets.clone(), 12),
                        BufferArg::from_raw_parts(write_offsets.clone(), writes.offsets.len()),
                        BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
                        BufferArg::from_raw_parts(writes.slots[1].0.clone(), writes.slots[1].1),
                        BufferArg::from_raw_parts(writes.slots[2].0.clone(), writes.slots[2].1),
                        BufferArg::from_raw_parts(writes.slots[3].0.clone(), writes.slots[3].1),
                        BufferArg::from_raw_parts(writes.slots[4].0.clone(), writes.slots[4].1),
                        BufferArg::from_raw_parts(writes.slots[5].0.clone(), writes.slots[5].1),
                        BufferArg::from_raw_parts(writes.slots[6].0.clone(), writes.slots[6].1),
                        BufferArg::from_raw_parts(writes.slots[7].0.clone(), writes.slots[7].1),
                        BufferArg::from_raw_parts(writes.slots[8].0.clone(), writes.slots[8].1),
                        BufferArg::from_raw_parts(writes.slots[9].0.clone(), writes.slots[9].1),
                        BufferArg::from_raw_parts(writes.slots[10].0.clone(), writes.slots[10].1),
                        BufferArg::from_raw_parts(writes.slots[11].0.clone(), writes.slots[11].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[0].0.clone(), partial_bindings.slots[0].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[1].0.clone(), partial_bindings.slots[1].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[2].0.clone(), partial_bindings.slots[2].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[3].0.clone(), partial_bindings.slots[3].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[4].0.clone(), partial_bindings.slots[4].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[5].0.clone(), partial_bindings.slots[5].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[6].0.clone(), partial_bindings.slots[6].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[7].0.clone(), partial_bindings.slots[7].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[8].0.clone(), partial_bindings.slots[8].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[9].0.clone(), partial_bindings.slots[9].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[10].0.clone(), partial_bindings.slots[10].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[11].0.clone(), partial_bindings.slots[11].1),
                    );
                }
                Ok(())
            }
        }
    };
}

impl_padded_scan_dispatch!(A13,Eval13,padded_scan_a13,Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12]);

fn scan_pass<R, Input, Output, Partials, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    output: &Output,
    partials: &Partials,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: PaddedReadSlots>
        + StageRead<R, Env0>,
    Output:
        OutputExpression + LowerOutputExpression<Slots: PaddedOutputSlots> + StageOutput<R, Env0>,
    Output::Item: WritableFrom<Item>,
    Partials:
        OutputExpression + LowerOutputExpression<Slots: PaddedOutputSlots> + StageOutput<R, Env0>,
    Partials::Item: WritableFrom<Item>,
    Item: StorageLayout,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            Partials,
            Item,
            KernelReadSlots<Input::Slots>,
            crate::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    <Dispatch<A13, S12> as InclusiveScanPassDispatch<
        R,
        Input,
        Output,
        Partials,
        Item,
        KernelReadSlots<Input::Slots>,
        crate::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run_pass(exec, input, output, partials)
}

fn add_fixed_prefixes<R, Output, Item, Op>(
    exec: &Executor<R>,
    prefixes: &FixedScanStorage<R, Item>,
    output: &Output,
    len: usize,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::api::iter::MItem<R> + crate::CanonicalAlloc<R>,
    Op: ReductionOp<Item>,
    Output: OutputExpression
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
    Output::Item: WritableFrom<Item>,
{
    if len == 0 {
        return Ok(());
    }
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let prefix_len = prefixes.len()?;
    if prefix_len != blocks {
        return Err(Error::LengthMismatch {
            left: blocks,
            right: prefix_len,
        });
    }

    let prefix_read = prefixes.read();
    let mut prefix_bindings = StagedBindings::new();
    prefix_read.stage_at(exec.client(), exec.id(), &mut prefix_bindings)?;
    prefix_bindings.pad_to_thirteen(exec.client());
    let mut output_bindings = OutputBindings::new();
    output.stage_output(exec.id(), &mut output_bindings)?;
    output_bindings.pad_to_twelve(exec.client());

    let prefix_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&prefix_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    let len_handle = exec.client().create_from_slice(u32::as_bytes(&[
        u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?
    ]));

    unsafe {
        add_block_prefix_padded12::launch_unchecked::<
            Item,
            <Item::StorageLeaves as StorePadded12>::O0,
            <Item::StorageLeaves as StorePadded12>::O1,
            <Item::StorageLeaves as StorePadded12>::O2,
            <Item::StorageLeaves as StorePadded12>::O3,
            <Item::StorageLeaves as StorePadded12>::O4,
            <Item::StorageLeaves as StorePadded12>::O5,
            <Item::StorageLeaves as StorePadded12>::O6,
            <Item::StorageLeaves as StorePadded12>::O7,
            <Item::StorageLeaves as StorePadded12>::O8,
            <Item::StorageLeaves as StorePadded12>::O9,
            <Item::StorageLeaves as StorePadded12>::O10,
            <Item::StorageLeaves as StorePadded12>::O11,
            Item::StorageLeaves,
            Item::DeviceLayout,
            Op,
            R,
        >(
            exec.client(),
            cube_count_1d(blocks)?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[0].0.clone(),
                prefix_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[1].0.clone(),
                prefix_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[2].0.clone(),
                prefix_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[3].0.clone(),
                prefix_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[4].0.clone(),
                prefix_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[5].0.clone(),
                prefix_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[6].0.clone(),
                prefix_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[7].0.clone(),
                prefix_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[8].0.clone(),
                prefix_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[9].0.clone(),
                prefix_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[10].0.clone(),
                prefix_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[11].0.clone(),
                prefix_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(len_handle, 1),
            BufferArg::from_raw_parts(prefix_offsets, prefix_bindings.offsets.len()),
            BufferArg::from_raw_parts(output_offsets, output_bindings.offsets.len()),
            BufferArg::from_raw_parts(
                output_bindings.slots[0].0.clone(),
                output_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[1].0.clone(),
                output_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[2].0.clone(),
                output_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[3].0.clone(),
                output_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[4].0.clone(),
                output_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[5].0.clone(),
                output_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[6].0.clone(),
                output_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[7].0.clone(),
                output_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[8].0.clone(),
                output_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[9].0.clone(),
                output_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[10].0.clone(),
                output_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[11].0.clone(),
                output_bindings.slots[11].1,
            ),
        );
    }
    Ok(())
}

fn scan_fixed_storage<R, Item, Op>(
    exec: &Executor<R>,
    input: &FixedScanStorage<R, Item>,
    output: &FixedScanStorage<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::api::iter::MItem<R> + crate::CanonicalAlloc<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: InclusiveScanPassDispatch<
            R,
            FixedScanRead<R, Item>,
            FixedScanOutput<R, Item>,
            FixedScanOutput<R, Item>,
            Item,
            KernelReadSlots<<FixedScanRead<R, Item> as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<
                <FixedScanOutput<R, Item> as LowerOutputExpression>::Slots,
            >,
            Op,
        >,
{
    let len = input.len()?;
    let output_len = output.len()?;
    if output_len != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: output_len,
        });
    }
    if len == 0 {
        return Ok(());
    }

    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let partials = exec.alloc_canonical::<Item>(blocks);
    let input_read = FixedScanRead::<R, Item>::new(input.read());
    let output_write = FixedScanOutput::<R, Item>::new(output.write());
    let partial_write = FixedScanOutput::<R, Item>::new(partials.write());
    scan_pass::<R, _, _, _, Item, Op>(exec, &input_read, &output_write, &partial_write)?;

    if blocks > 1 {
        let prefixes = exec.alloc_canonical::<Item>(blocks);
        scan_fixed_storage::<R, Item, Op>(exec, &partials, &prefixes)?;
        add_fixed_prefixes::<R, _, Item, Op>(exec, &prefixes, &output_write, len)?;
    }
    Ok(())
}

impl<R, Input, Output, Item, Op, ReadSlots, WriteSlots>
    InclusiveScanDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Op> for Dispatch<A13, S12>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
    Output::Item: WritableFrom<Item>,
    Item: crate::api::iter::MItem<R> + crate::CanonicalAlloc<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            FixedScanOutput<R, Item>,
            Item,
            ReadSlots,
            WriteSlots,
            Op,
        > + InclusiveScanPassDispatch<
            R,
            FixedScanRead<R, Item>,
            FixedScanOutput<R, Item>,
            FixedScanOutput<R, Item>,
            Item,
            KernelReadSlots<<FixedScanRead<R, Item> as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<
                <FixedScanOutput<R, Item> as LowerOutputExpression>::Slots,
            >,
            Op,
        >,
{
    fn run(exec: &Executor<R>, input: &Input, _op: Op, output: &Output) -> Result<(), Error> {
        let len = input.logical_len()?;
        let output_len = output.logical_len()?;
        if output_len != len {
            return Err(Error::LengthMismatch {
                left: len,
                right: output_len,
            });
        }
        if len == 0 {
            return Ok(());
        }

        let blocks = len.div_ceil(BLOCK_SIZE as usize);
        let partials = exec.alloc_canonical::<Item>(blocks);
        let partial_write = FixedScanOutput::<R, Item>::new(partials.write());
        <Dispatch<A13, S12> as InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            FixedScanOutput<R, Item>,
            Item,
            ReadSlots,
            WriteSlots,
            Op,
        >>::run_pass(exec, input, output, &partial_write)?;

        if blocks > 1 {
            let prefixes = exec.alloc_canonical::<Item>(blocks);
            scan_fixed_storage::<R, Item, Op>(exec, &partials, &prefixes)?;
            add_fixed_prefixes::<R, _, Item, Op>(exec, &prefixes, output, len)?;
        }
        Ok(())
    }
}

#[cfg(any())]
fn scan_scalar_column<R, Item, Op>(
    exec: &Executor<R>,
    input: &DeviceVec<R, Item>,
    op: Op,
    output: &DeviceVec<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: MStorageElement + StorageLayout<StorageArity = S1>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: InclusiveScanDispatch<
            R,
            crate::Column<Item>,
            DeviceSliceMut<Item>,
            Item,
            KernelReadSlots<Env1<Item>>,
            crate::output::KernelOutputSlots<Env1<Item>>,
            Op,
        >,
{
    <Dispatch<A13, S12> as InclusiveScanDispatch<
        R,
        crate::Column<Item>,
        DeviceSliceMut<Item>,
        Item,
        KernelReadSlots<Env1<Item>>,
        crate::output::KernelOutputSlots<Env1<Item>>,
        Op,
    >>::run(exec, &input.column(), op, &output.slice_mut(..))
}

/// Computes an inclusive scan into preallocated output storage.
pub(crate) fn inclusive_scan<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Op: ReductionOp<Input::Item>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<A13, S12>: InclusiveScanDispatch<
            R,
            Input,
            Output,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            crate::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    <Dispatch<A13, S12> as InclusiveScanDispatch<
        R,
        Input,
        Output,
        Input::Item,
        KernelReadSlots<Input::Slots>,
        crate::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run(exec, &input, op, &output)
}

/// Computes adjacent reductions while preserving the first input item.
pub(crate) fn adjacent_difference<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression,
    Op: ReductionOp<Input::Item>,
    Adjacent<Input, Op>:
        ReadExpression<Item = Input::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: WritableFrom<Input::Item>,
    Dispatch<crate::A13, crate::S12>: MaterializeDispatch<
            R,
            Adjacent<Input, Op>,
            Output,
            crate::read::KernelReadSlots<<Adjacent<Input, Op> as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<Output::Slots>,
        >,
    Output::Slots: PaddedOutputSlots,
{
    materialize(exec, Adjacent::new(input, op), output)
}

/// Internal public-API capability for a fully generic exclusive scan.
#[doc(hidden)]
pub trait ExclusiveScanInput<R: Runtime, Output, Op>: PrependInput<R> {
    fn exclusive_scan_into(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Input, Output, Op> ExclusiveScanInput<R, Output, Op> for Input
where
    R: Runtime,
    Input: PrependInput<R>,
    Input::Item: crate::api::iter::MItem<R>
        + CanonicalAlloc<R, CanonicalStorage = Input::Storage>,
    <Input::Item as StorageLayout>::StorageLeaves:
        StorePadded12 + crate::core::facade::KernelValue,
    Input::Storage: CanonicalStorage<R, Item = <Input::Item as CanonicalAlloc<R>>::CanonicalItem>,
    Input::SemanticRead: LowerReadExpression + StageRead<R, Env0>,
    <Input::Storage as CanonicalStorage<R>>::Write: LowerOutputExpression + StageOutput<R, Env0>,
    <<Input::Storage as CanonicalStorage<R>>::Write as OutputExpression>::Item:
        WritableFrom<Input::Item>,
    Dispatch<A13, S12>: InclusiveScanDispatch<
            R,
            Input::SemanticRead,
            crate::output::ReassociatedOutput<
                <Input::Storage as CanonicalStorage<R>>::Write,
                Input::Item,
                <<Input::Item as StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots,
            >,
            Input::Item,
            KernelReadSlots<<Input::SemanticRead as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<
                <<Input::Item as StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots,
            >,
            Op,
        >,
    <Input::Storage as CanonicalStorage<R>>::Read: GatherInput<R, Counting, Output>,
    Op: ReductionOp<Input::Item>,
{
    fn exclusive_scan_into(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error> {
        let prefixed = self.prepend(exec, init)?;
        let prefixed_len = prefixed.len()?;
        let scanned = exec.alloc_canonical::<Input::Item>(prefixed_len);
        let scanned_output = crate::output::ReassociatedOutput::<
            _,
            Input::Item,
            <<Input::Item as StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots,
        >::new(scanned.write());
        inclusive_scan(exec, Input::semantic_read(&prefixed), op, scanned_output)?;
        crate::indexed::gather_direct(
            exec,
            scanned.read(),
            Counting::new(0, prefixed_len.saturating_sub(1)),
            output,
        )
    }
}

/// Computes an exclusive scan into preallocated output storage.
pub(crate) fn exclusive_scan<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Input::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ExclusiveScanInput<R, Output, Op>,
{
    input.exclusive_scan_into(exec, init, op, output)
}

pub(crate) fn inclusive_scan_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error> {
    if input.is_empty() {
        return Ok(exec.alloc_canonical::<u32>(0));
    }
    let len = input.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let output = exec.alloc_canonical::<u32>(len);
    let block_sums = exec.alloc_canonical::<u32>(blocks);
    let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
    let count = cube_count_1d(blocks)?;
    unsafe {
        u32_block_inclusive_scan_kernel::launch_unchecked::<R>(
            exec.client(),
            count.clone(),
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(input.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output.handle.clone(), len),
            BufferArg::from_raw_parts(block_sums.handle.clone(), blocks),
        );
    }
    if blocks > 1 {
        let prefixes = inclusive_scan_u32(exec, &block_sums)?;
        unsafe {
            u32_add_block_prefix_kernel::launch_unchecked::<R>(
                exec.client(),
                count,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(prefixes.handle.clone(), blocks),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(output.handle.clone(), len),
            );
        }
    }
    Ok(output)
}

pub(crate) fn last_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<u32, Error> {
    if input.is_empty() {
        return Ok(0);
    }
    let output = exec.alloc_canonical::<u32>(1);
    unsafe {
        copy_last_kernel::launch_unchecked::<R>(
            exec.client(),
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input.handle.clone(), input.len()),
            BufferArg::from_raw_parts(output.handle.clone(), 1),
        );
    }
    Ok(exec.to_host(&output)?[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalStorage, Counting, Permute, Transform, Zip, op::UnaryOp};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn inclusive_u32_scan_crosses_block_and_recursive_boundaries() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&vec![1_u32; 70_001]);
        let output = inclusive_scan_u32(&exec, &input).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_eq!(actual[0], 1);
        assert_eq!(actual[255], 256);
        assert_eq!(actual[256], 257);
        assert_eq!(actual[65_535], 65_536);
        assert_eq!(actual[70_000], 70_001);
        assert_eq!(last_u32(&exec, &output).unwrap(), 70_001);
    }

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    struct SumPair;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for SumPair {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1)
        }
    }

    #[test]
    fn inclusive_pair_scan_crosses_recursive_block_boundary() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 70_001;
        let left = exec.to_device(&vec![1_u32; len]);
        let right = exec.to_device(&vec![2_u32; len]);
        let output = exec.alloc_canonical::<(u32, u32)>(len);

        inclusive_scan(
            &exec,
            Zip::new(left.column(), right.column()),
            SumPair,
            output.write(),
        )
        .unwrap();

        let actual_left = exec.to_host(&output.0).unwrap();
        let actual_right = exec.to_host(&output.1).unwrap();
        for &index in &[0, 255, 256, 65_535, 70_000] {
            assert_eq!(actual_left[index], index as u32 + 1);
            assert_eq!(actual_right[index], 2 * (index as u32 + 1));
        }
    }

    type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));
    struct SumSeven;

    #[cubecl::cube]
    impl UnaryOp<Seven> for SumSeven {
        type Output = u32;
        fn apply(input: Seven) -> u32 {
            input.0
                + input.1.0
                + input.1.1.0
                + input.1.1.1.0
                + input.1.1.1.1.0
                + input.1.1.1.1.1.0
                + input.1.1.1.1.1.1
        }
    }

    struct SumSevenItems;

    #[cubecl::cube]
    impl ReductionOp<Seven> for SumSevenItems {
        fn apply(lhs: Seven, rhs: Seven) -> Seven {
            (
                lhs.0 + rhs.0,
                (
                    lhs.1.0 + rhs.1.0,
                    (
                        lhs.1.1.0 + rhs.1.1.0,
                        (
                            lhs.1.1.1.0 + rhs.1.1.1.0,
                            (
                                lhs.1.1.1.1.0 + rhs.1.1.1.1.0,
                                (
                                    lhs.1.1.1.1.1.0 + rhs.1.1.1.1.1.0,
                                    lhs.1.1.1.1.1.1 + rhs.1.1.1.1.1.1,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        }
    }

    #[test]
    fn inclusive_s7_scan_dispatches_eval8_and_normalizes_output_shape() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; 600]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 600));
        let output = exec.alloc_canonical::<Seven>(600);

        inclusive_scan(&exec, input, SumSevenItems, output.write()).unwrap();

        assert_eq!(exec.to_host(&output.0.0.0.0.0.0).unwrap()[599], 600);
        assert_eq!(exec.to_host(&output.0.0.0.0.0.1).unwrap()[599], 1_200);
        assert_eq!(exec.to_host(&output.0.0.0.0.1).unwrap()[599], 1_800);
        assert_eq!(exec.to_host(&output.0.0.0.1).unwrap()[599], 2_400);
        assert_eq!(exec.to_host(&output.0.0.1).unwrap()[599], 3_000);
        assert_eq!(exec.to_host(&output.0.1).unwrap()[599], 3_600);
        assert_eq!(exec.to_host(&output.1).unwrap()[599], 4_200);
    }

    #[test]
    fn inclusive_scalar_scan_dispatches_eval8() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (0..7).map(|_| exec.to_device(&[1_u32; 600])).collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Transform::new(Permute::new(seven, Counting::new(0, 600)), SumSeven);
        let output = exec.to_device(&[0_u32; 600]);
        inclusive_scan(&exec, input, Sum, output.slice_mut(..)).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_eq!(actual[0], 7);
        assert_eq!(actual[255], 7 * 256);
        assert_eq!(actual[256], 7 * 257);
        assert_eq!(actual[599], 7 * 600);
    }

    #[test]
    fn exclusive_scalar_scan_applies_init_once() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 2, 3, 4]);
        let output = exec.to_device(&[99_u32; 6]);
        exclusive_scan(&exec, input.column(), 10, Sum, output.slice_mut(1..5)).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![99, 10, 11, 13, 16, 99]);
    }

    #[test]
    fn adjacent_difference_is_a_regular_fused_read_expression() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 3, 6, 10]);
        let output = exec.to_device(&[0_u32; 4]);
        adjacent_difference(&exec, input.column(), Sum, output.slice_mut(..)).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![1, 4, 9, 16]);
    }

    #[test]
    fn exclusive_storage7_accepts_eval8_and_preserves_semantic_init() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&[value; 4]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 4));
        let output = exec.alloc_canonical::<Seven>(4);
        let init: Seven = (10, (20, (30, (40, (50, (60, 70))))));
        exclusive_scan(&exec, input, init, SumSevenItems, output.write()).unwrap();

        assert_eq!(
            exec.to_host(&output.0.0.0.0.0.0).unwrap(),
            vec![10, 11, 12, 13]
        );
        assert_eq!(exec.to_host(&output.1).unwrap(), vec![70, 77, 84, 91]);
    }

    #[test]
    fn scan_rejects_mismatched_output_tree_and_foreign_storage() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 3]);
        let right = exec.to_device(&[4_u32, 5, 6]);
        let out_left = exec.to_device(&[0_u32; 3]);
        let out_right = exec.to_device(&[0_u32; 2]);
        assert_eq!(
            inclusive_scan(
                &exec,
                Zip::new(left.column(), right.column()),
                SumPair,
                Zip::new(out_left.slice_mut(..), out_right.slice_mut(..)),
            ),
            Err(Error::LengthMismatch { left: 3, right: 2 })
        );

        let other = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let foreign_output = other.to_device(&[0_u32; 3]);
        assert_eq!(
            inclusive_scan(&exec, left.column(), Sum, foreign_output.slice_mut(..)),
            Err(Error::ForeignExecutor)
        );
    }
}
