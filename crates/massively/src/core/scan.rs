//! Reusable prefix-scan control primitives.

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, A8, CanonicalAlloc, CanonicalStorage, Counting, DeviceSliceMut,
    DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression, S1, S2, S3, S4, S5, S6,
    S7, StorageLayout, WriteFrom,
    allocation::PrependInput,
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7, Eval8},
    indexed::GatherInput,
    launch::cube_count_1d,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{Adjacent, Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, Env8, LowerReadExpression},
    reduce::{ReductionOp, StageRead, StagedBindings},
    storage::{
        Decompose, Last, LoadLeaves2, LoadLeaves3, LoadLeaves4, LoadLeaves5, LoadLeaves6,
        LoadLeaves7, LoadMutLeaves2, LoadMutLeaves3, LoadMutLeaves4, LoadMutLeaves5,
        LoadMutLeaves6, LoadMutLeaves7, More, MutableLeaves, PlaneShuffleLeaves, Recompose,
        SelectStoreLeaves2, SelectStoreLeaves2Expand, SelectStoreLeaves3, SelectStoreLeaves3Expand,
        SelectStoreLeaves4, SelectStoreLeaves4Expand, SelectStoreLeaves5, SelectStoreLeaves5Expand,
        SelectStoreLeaves6, SelectStoreLeaves6Expand, SelectStoreLeaves7, SelectStoreLeaves7Expand,
        StoreLeaves2, StoreLeaves2Expand, StoreLeaves3, StoreLeaves3Expand, StoreLeaves4,
        StoreLeaves4Expand, StoreLeaves5, StoreLeaves5Expand, StoreLeaves6, StoreLeaves6Expand,
        StoreLeaves7, StoreLeaves7Expand,
    },
    transform::{MaterializeDispatch, materialize},
};

const BLOCK_SIZE: u32 = 256;

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

macro_rules! define_multi_scan_kernel {
    (
        $name:ident,$eval:ident,$method:ident,$load_trait:ident,$store_trait:ident,$select_trait:ident;
        [$( $leaf:ident:$slot:ident ),+];
        [$( $out_ty:ident:$output:ident:$shared:ident:$block_sum:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            $( $out_ty: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>
                + $select_trait<$( $out_ty ),+>
                + MutableLeaves
                + PlaneShuffleLeaves,
            Expr: $eval<Item, $( $leaf ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            zero_offsets: &[u32],
            output_offsets: &[u32],
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

            let safe_global = if global < logical_len { global } else { 0usize };
            let cells = <Leaves as MutableLeaves>::into_cells(Layout::decompose(
                Expr::$method($( $slot, )+ read_offsets, safe_global),
            ));
            let is_valid = RuntimeCell::<u32>::new(
                if global < logical_len { 1u32 } else { 0u32 },
            );

            let offset = RuntimeCell::<u32>::new(1u32);
            while offset.read() < PLANE_DIM {
                let left_cells = <Leaves as MutableLeaves>::into_cells(
                    Leaves::shuffle_leaves_up(
                        <Leaves as MutableLeaves>::read(&cells),
                        offset.read(),
                    ),
                );
                let left_valid = plane_shuffle_up(is_valid.read(), offset.read());
                if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
                    if is_valid.read() != 0u32 {
                        let combined = Layout::decompose(Op::apply(
                            Layout::recompose(<Leaves as MutableLeaves>::read(&left_cells)),
                            Layout::recompose(<Leaves as MutableLeaves>::read(&cells)),
                        ));
                        <Leaves as MutableLeaves>::store(&cells, combined);
                    } else {
                        <Leaves as MutableLeaves>::store(
                            &cells,
                            <Leaves as MutableLeaves>::read(&left_cells),
                        );
                        is_valid.store(1u32);
                    }
                }
                offset.store(offset.read() * 2u32);
            }

            if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
                <Leaves as MutableLeaves>::read(&cells).store(
                    $( &mut $shared, )+ zero_offsets, PLANE_POS as usize,
                );
                valid[PLANE_POS as usize] = is_valid.read();
            }
            sync_cube();

            if unit == 0usize {
                let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                let plane_cells = <Leaves as MutableLeaves>::into_cells(Leaves::load(
                    $( &$shared, )+ zero_offsets, 0usize,
                ));
                let plane_is_valid = RuntimeCell::<u32>::new(valid[0]);
                let plane = RuntimeCell::<u32>::new(1u32);
                while plane.read() < plane_count {
                    let index = plane.read() as usize;
                    if valid[index] != 0u32 {
                        if plane_is_valid.read() != 0u32 {
                            let combined = Layout::decompose(Op::apply(
                                Layout::recompose(<Leaves as MutableLeaves>::read(&plane_cells)),
                                Layout::recompose(Leaves::load(
                                    $( &$shared, )+ zero_offsets, index,
                                )),
                            ));
                            <Leaves as MutableLeaves>::store(&plane_cells, combined);
                        } else {
                            <Leaves as MutableLeaves>::store(
                                &plane_cells,
                                Leaves::load($( &$shared, )+ zero_offsets, index),
                            );
                            plane_is_valid.store(1u32);
                        }
                    }
                    <Leaves as MutableLeaves>::read(&plane_cells).store(
                        $( &mut $shared, )+ zero_offsets, index,
                    );
                    plane.store(plane.read() + 1u32);
                }
            }
            sync_cube();

            if PLANE_POS > 0u32 && is_valid.read() != 0u32 {
                let prefix = Leaves::load(
                    $( &$shared, )+ zero_offsets, PLANE_POS as usize - 1usize,
                );
                let combined = Layout::decompose(Op::apply(
                    Layout::recompose(prefix),
                    Layout::recompose(<Leaves as MutableLeaves>::read(&cells)),
                ));
                <Leaves as MutableLeaves>::store(&cells, combined);
            }

            if global < logical_len {
                <Leaves as MutableLeaves>::read(&cells).store(
                    $( $output, )+ output_offsets, global,
                );
            }
            if unit == 0usize {
                let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                Leaves::load(
                    $( &$shared, )+ zero_offsets, plane_count as usize - 1usize,
                ).store($( $block_sum, )+ zero_offsets, block);
            }
        }
    };
}

macro_rules! define_multi_scan_kernels_for_eval {
    ($eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]; [$k2:ident,$k3:ident,$k4:ident,$k5:ident,$k6:ident,$k7:ident]) => {
        define_multi_scan_kernel!($k2,$eval,$method,LoadLeaves2,StoreLeaves2,SelectStoreLeaves2; [$($leaf:$slot),+]; [O0:out0:shared0:sum0,O1:out1:shared1:sum1]);
        define_multi_scan_kernel!($k3,$eval,$method,LoadLeaves3,StoreLeaves3,SelectStoreLeaves3; [$($leaf:$slot),+]; [O0:out0:shared0:sum0,O1:out1:shared1:sum1,O2:out2:shared2:sum2]);
        define_multi_scan_kernel!($k4,$eval,$method,LoadLeaves4,StoreLeaves4,SelectStoreLeaves4; [$($leaf:$slot),+]; [O0:out0:shared0:sum0,O1:out1:shared1:sum1,O2:out2:shared2:sum2,O3:out3:shared3:sum3]);
        define_multi_scan_kernel!($k5,$eval,$method,LoadLeaves5,StoreLeaves5,SelectStoreLeaves5; [$($leaf:$slot),+]; [O0:out0:shared0:sum0,O1:out1:shared1:sum1,O2:out2:shared2:sum2,O3:out3:shared3:sum3,O4:out4:shared4:sum4]);
        define_multi_scan_kernel!($k6,$eval,$method,LoadLeaves6,StoreLeaves6,SelectStoreLeaves6; [$($leaf:$slot),+]; [O0:out0:shared0:sum0,O1:out1:shared1:sum1,O2:out2:shared2:sum2,O3:out3:shared3:sum3,O4:out4:shared4:sum4,O5:out5:shared5:sum5]);
        define_multi_scan_kernel!($k7,$eval,$method,LoadLeaves7,StoreLeaves7,SelectStoreLeaves7; [$($leaf:$slot),+]; [O0:out0:shared0:sum0,O1:out1:shared1:sum1,O2:out2:shared2:sum2,O3:out3:shared3:sum3,O4:out4:shared4:sum4,O5:out5:shared5:sum5,O6:out6:shared6:sum6]);
    };
}

define_multi_scan_kernels_for_eval!(Eval1,eval1; [L0:slot0]; [multi_scan_a1_s2,multi_scan_a1_s3,multi_scan_a1_s4,multi_scan_a1_s5,multi_scan_a1_s6,multi_scan_a1_s7]);
define_multi_scan_kernels_for_eval!(Eval2,eval2; [L0:slot0,L1:slot1]; [multi_scan_a2_s2,multi_scan_a2_s3,multi_scan_a2_s4,multi_scan_a2_s5,multi_scan_a2_s6,multi_scan_a2_s7]);
define_multi_scan_kernels_for_eval!(Eval3,eval3; [L0:slot0,L1:slot1,L2:slot2]; [multi_scan_a3_s2,multi_scan_a3_s3,multi_scan_a3_s4,multi_scan_a3_s5,multi_scan_a3_s6,multi_scan_a3_s7]);
define_multi_scan_kernels_for_eval!(Eval4,eval4; [L0:slot0,L1:slot1,L2:slot2,L3:slot3]; [multi_scan_a4_s2,multi_scan_a4_s3,multi_scan_a4_s4,multi_scan_a4_s5,multi_scan_a4_s6,multi_scan_a4_s7]);
define_multi_scan_kernels_for_eval!(Eval5,eval5; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4]; [multi_scan_a5_s2,multi_scan_a5_s3,multi_scan_a5_s4,multi_scan_a5_s5,multi_scan_a5_s6,multi_scan_a5_s7]);
define_multi_scan_kernels_for_eval!(Eval6,eval6; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5]; [multi_scan_a6_s2,multi_scan_a6_s3,multi_scan_a6_s4,multi_scan_a6_s5,multi_scan_a6_s6,multi_scan_a6_s7]);
define_multi_scan_kernels_for_eval!(Eval7,eval7; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6]; [multi_scan_a7_s2,multi_scan_a7_s3,multi_scan_a7_s4,multi_scan_a7_s5,multi_scan_a7_s6,multi_scan_a7_s7]);
define_multi_scan_kernels_for_eval!(Eval8,eval8; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7]; [multi_scan_a8_s2,multi_scan_a8_s3,multi_scan_a8_s4,multi_scan_a8_s5,multi_scan_a8_s6,multi_scan_a8_s7]);

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

define_storage_scan_kernel!(storage_scan_s2,LoadLeaves2,StoreLeaves2,SelectStoreLeaves2; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1]);
define_storage_scan_kernel!(storage_scan_s3,LoadLeaves3,StoreLeaves3,SelectStoreLeaves3; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2]);
define_storage_scan_kernel!(storage_scan_s4,LoadLeaves4,StoreLeaves4,SelectStoreLeaves4; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3]);
define_storage_scan_kernel!(storage_scan_s5,LoadLeaves5,StoreLeaves5,SelectStoreLeaves5; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4]);
define_storage_scan_kernel!(storage_scan_s6,LoadLeaves6,StoreLeaves6,SelectStoreLeaves6; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5]);
define_storage_scan_kernel!(storage_scan_s7,LoadLeaves7,StoreLeaves7,SelectStoreLeaves7; [O0:in0:out0:shared0:sum0,O1:in1:out1:shared1:sum1,O2:in2:out2:shared2:sum2,O3:in3:out3:shared3:sum3,O4:in4:out4:shared4:sum4,O5:in5:out5:shared5:sum5,O6:in6:out6:shared6:sum6]);

define_multi_add_prefix_kernel!(multi_add_prefix_s2,LoadLeaves2,LoadMutLeaves2,StoreLeaves2; [O0:prefix0:out0,O1:prefix1:out1]);
define_multi_add_prefix_kernel!(multi_add_prefix_s3,LoadLeaves3,LoadMutLeaves3,StoreLeaves3; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2]);
define_multi_add_prefix_kernel!(multi_add_prefix_s4,LoadLeaves4,LoadMutLeaves4,StoreLeaves4; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3]);
define_multi_add_prefix_kernel!(multi_add_prefix_s5,LoadLeaves5,LoadMutLeaves5,StoreLeaves5; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4]);
define_multi_add_prefix_kernel!(multi_add_prefix_s6,LoadLeaves6,LoadMutLeaves6,StoreLeaves6; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5]);
define_multi_add_prefix_kernel!(multi_add_prefix_s7,LoadLeaves7,LoadMutLeaves7,StoreLeaves7; [O0:prefix0:out0,O1:prefix1:out1,O2:prefix2:out2,O3:prefix3:out3,O4:prefix4:out4,O5:prefix5:out5,O6:prefix6:out6]);

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

#[doc(hidden)]
pub trait InclusiveScanDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Op>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, input: &Input, op: Op, output: &Output) -> Result<(), Error>;
}

macro_rules! impl_scalar_scan_dispatch {
    ($arity:ty,$eval:ident,$kernel:ident; [$( $leaf:ident:$index:literal ),+],$env:ty) => {
        impl<R, Input, Output, Item, Op, $( $leaf ),+>
            InclusiveScanDispatch<R, Input, Output, Item, $env, Env1<Item>, Op>
            for Dispatch<$arity, S1>
        where
            R: Runtime,
            Item: MStorageElement + StorageLayout<StorageArity = S1>,
            Op: ReductionOp<Item>,
            $( $leaf: MStorageElement, )+
            Input: ReadExpression<Item = Item>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: OutputExpression<Item = Item, StorageArity = S1>
                + LowerOutputExpression<Slots = Env1<Item>>
                + StageOutput<R, Env0>,
        {
            fn run(
                exec: &Executor<R>,
                input: &Input,
                op: Op,
                output: &Output,
            ) -> Result<(), Error> {
                let len = input.logical_len()?;
                let output_len = output.logical_len()?;
                if output_len != len {
                    return Err(Error::LengthMismatch { left: len, right: output_len });
                }
                if len == 0 { return Ok(()); }
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let blocks = len.div_ceil(BLOCK_SIZE as usize);
                let block_sums = exec.alloc_column::<Item>(blocks);
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                let count = cube_count_1d(blocks)?;
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Op, R>(
                        exec.client(), count.clone(), CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.clone(), 1),
                        BufferArg::from_raw_parts(write_offsets.clone(), writes.offsets.len()),
                        BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
                        BufferArg::from_raw_parts(block_sums.handle.clone(), blocks),
                    );
                }
                if blocks > 1 {
                    let block_prefixes = exec.alloc_column::<Item>(blocks);
                    scan_scalar_column::<R, Item, Op>(exec, &block_sums, op, &block_prefixes)?;
                    unsafe {
                        scalar_add_block_prefix_kernel::launch_unchecked::<Item, Op, R>(
                            exec.client(), count, CubeDim::new_1d(BLOCK_SIZE),
                            BufferArg::from_raw_parts(block_prefixes.handle.clone(), blocks),
                            BufferArg::from_raw_parts(len_handle, 1),
                            BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
                            BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
                        );
                    }
                }
                Ok(())
            }
        }
    };
}

impl_scalar_scan_dispatch!(A1,Eval1,scalar_scan_a1; [L0:0],Env1<L0>);
impl_scalar_scan_dispatch!(A2,Eval2,scalar_scan_a2; [L0:0,L1:1],Env2<L0,L1>);
impl_scalar_scan_dispatch!(A3,Eval3,scalar_scan_a3; [L0:0,L1:1,L2:2],Env3<L0,L1,L2>);
impl_scalar_scan_dispatch!(A4,Eval4,scalar_scan_a4; [L0:0,L1:1,L2:2,L3:3],Env4<L0,L1,L2,L3>);
impl_scalar_scan_dispatch!(A5,Eval5,scalar_scan_a5; [L0:0,L1:1,L2:2,L3:3,L4:4],Env5<L0,L1,L2,L3,L4>);
impl_scalar_scan_dispatch!(A6,Eval6,scalar_scan_a6; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5],Env6<L0,L1,L2,L3,L4,L5>);
impl_scalar_scan_dispatch!(A7,Eval7,scalar_scan_a7; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6],Env7<L0,L1,L2,L3,L4,L5,L6>);
impl_scalar_scan_dispatch!(A8,Eval8,scalar_scan_a8; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7],Env8<L0,L1,L2,L3,L4,L5,L6,L7>);

macro_rules! impl_multi_scan_dispatch {
    (
        $arity:ty,$eval:ident,$kernel:ident,$storage:ty,$read_env:ty,$out_env:ty,
        $leaves:ty,$host_scan:ident,$prefix_kernel:ident;
        [$( $leaf:ident:$read_index:literal ),+];
        [$( $out_ty:ident:$out_index:literal:$block_sum:ident:$prefix:ident ),+]
    ) => {
        impl<R, Input, Output, Item, Op, $( $leaf, )+ $( $out_ty ),+>
            InclusiveScanDispatch<R, Input, Output, Item, $read_env, $out_env, Op>
            for Dispatch<$arity, $storage>
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $storage, StorageLeaves = $leaves>
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout: Recompose<Item, Leaves = $leaves>,
            Op: ReductionOp<Item>,
            $( $leaf: MStorageElement, )+
            $( $out_ty: MStorageElement, )+
            Input: ReadExpression<Item = Item>
                + LowerReadExpression<Slots = $read_env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: OutputExpression<StorageArity = $storage>
                + LowerOutputExpression<Slots = $out_env>
                + StageOutput<R, Env0>,
            Output::Item: WriteFrom<Item>,
            $leaves: Send + Sync + 'static,
        {
            fn run(
                exec: &Executor<R>,
                input: &Input,
                op: Op,
                output: &Output,
            ) -> Result<(), Error> {
                let len = input.logical_len()?;
                let output_len = output.logical_len()?;
                if output_len != len {
                    return Err(Error::LengthMismatch { left: len, right: output_len });
                }
                if len == 0 {
                    return Ok(());
                }
                let _ = op;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let blocks = len.div_ceil(BLOCK_SIZE as usize);
                $( let $block_sum = exec.alloc_column::<$out_ty>(blocks); )+
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let zero_offsets = vec![$( { let _ = stringify!($out_ty); 0u32 } ),+];
                let zero_offsets_handle = exec.client().create_from_slice(u32::as_bytes(&zero_offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                let count = cube_count_1d(blocks)?;
                unsafe {
                    $kernel::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        $( $out_ty, )+
                        $leaves,
                        Input::DeviceExpr,
                        Item::DeviceLayout,
                        Op,
                        R,
                    >(
                        exec.client(),
                        count.clone(),
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$read_index].0.clone(), reads.slots[$read_index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.clone(), 1),
                        BufferArg::from_raw_parts(zero_offsets_handle.clone(), zero_offsets.len()),
                        BufferArg::from_raw_parts(write_offsets.clone(), writes.offsets.len()),
                        $( BufferArg::from_raw_parts(writes.slots[$out_index].0.clone(), writes.slots[$out_index].1), )+
                        $( BufferArg::from_raw_parts($block_sum.handle.clone(), blocks), )+
                    );
                }
                if blocks > 1 {
                    $( let $prefix = exec.alloc_column::<$out_ty>(blocks); )+
                    $host_scan::<R, Item, Op, $( $out_ty ),+>(
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
                            BufferArg::from_raw_parts(zero_offsets_handle, zero_offsets.len()),
                            BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
                            $( BufferArg::from_raw_parts(writes.slots[$out_index].0.clone(), writes.slots[$out_index].1), )+
                        );
                    }
                }
                Ok(())
            }
        }
    };
}

macro_rules! impl_multi_scan_dispatches_for_eval {
    (
        $arity:ty,$eval:ident,$read_env:ty;
        [$( $leaf:ident:$read_index:literal ),+];
        [$k2:ident,$k3:ident,$k4:ident,$k5:ident,$k6:ident,$k7:ident]
    ) => {
        impl_multi_scan_dispatch!($arity,$eval,$k2,S2,$read_env,Env2<O0,O1>,More<O0,Last<O1>>,scan_storage2,multi_add_prefix_s2; [$($leaf:$read_index),+]; [O0:0:sum0:prefix0,O1:1:sum1:prefix1]);
        impl_multi_scan_dispatch!($arity,$eval,$k3,S3,$read_env,Env3<O0,O1,O2>,More<O0,More<O1,Last<O2>>>,scan_storage3,multi_add_prefix_s3; [$($leaf:$read_index),+]; [O0:0:sum0:prefix0,O1:1:sum1:prefix1,O2:2:sum2:prefix2]);
        impl_multi_scan_dispatch!($arity,$eval,$k4,S4,$read_env,Env4<O0,O1,O2,O3>,More<O0,More<O1,More<O2,Last<O3>>>>,scan_storage4,multi_add_prefix_s4; [$($leaf:$read_index),+]; [O0:0:sum0:prefix0,O1:1:sum1:prefix1,O2:2:sum2:prefix2,O3:3:sum3:prefix3]);
        impl_multi_scan_dispatch!($arity,$eval,$k5,S5,$read_env,Env5<O0,O1,O2,O3,O4>,More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>,scan_storage5,multi_add_prefix_s5; [$($leaf:$read_index),+]; [O0:0:sum0:prefix0,O1:1:sum1:prefix1,O2:2:sum2:prefix2,O3:3:sum3:prefix3,O4:4:sum4:prefix4]);
        impl_multi_scan_dispatch!($arity,$eval,$k6,S6,$read_env,Env6<O0,O1,O2,O3,O4,O5>,More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>,scan_storage6,multi_add_prefix_s6; [$($leaf:$read_index),+]; [O0:0:sum0:prefix0,O1:1:sum1:prefix1,O2:2:sum2:prefix2,O3:3:sum3:prefix3,O4:4:sum4:prefix4,O5:5:sum5:prefix5]);
        impl_multi_scan_dispatch!($arity,$eval,$k7,S7,$read_env,Env7<O0,O1,O2,O3,O4,O5,O6>,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>,scan_storage7,multi_add_prefix_s7; [$($leaf:$read_index),+]; [O0:0:sum0:prefix0,O1:1:sum1:prefix1,O2:2:sum2:prefix2,O3:3:sum3:prefix3,O4:4:sum4:prefix4,O5:5:sum5:prefix5,O6:6:sum6:prefix6]);
    };
}

impl_multi_scan_dispatches_for_eval!(A1,Eval1,Env1<L0>; [L0:0]; [multi_scan_a1_s2,multi_scan_a1_s3,multi_scan_a1_s4,multi_scan_a1_s5,multi_scan_a1_s6,multi_scan_a1_s7]);
impl_multi_scan_dispatches_for_eval!(A2,Eval2,Env2<L0,L1>; [L0:0,L1:1]; [multi_scan_a2_s2,multi_scan_a2_s3,multi_scan_a2_s4,multi_scan_a2_s5,multi_scan_a2_s6,multi_scan_a2_s7]);
impl_multi_scan_dispatches_for_eval!(A3,Eval3,Env3<L0,L1,L2>; [L0:0,L1:1,L2:2]; [multi_scan_a3_s2,multi_scan_a3_s3,multi_scan_a3_s4,multi_scan_a3_s5,multi_scan_a3_s6,multi_scan_a3_s7]);
impl_multi_scan_dispatches_for_eval!(A4,Eval4,Env4<L0,L1,L2,L3>; [L0:0,L1:1,L2:2,L3:3]; [multi_scan_a4_s2,multi_scan_a4_s3,multi_scan_a4_s4,multi_scan_a4_s5,multi_scan_a4_s6,multi_scan_a4_s7]);
impl_multi_scan_dispatches_for_eval!(A5,Eval5,Env5<L0,L1,L2,L3,L4>; [L0:0,L1:1,L2:2,L3:3,L4:4]; [multi_scan_a5_s2,multi_scan_a5_s3,multi_scan_a5_s4,multi_scan_a5_s5,multi_scan_a5_s6,multi_scan_a5_s7]);
impl_multi_scan_dispatches_for_eval!(A6,Eval6,Env6<L0,L1,L2,L3,L4,L5>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5]; [multi_scan_a6_s2,multi_scan_a6_s3,multi_scan_a6_s4,multi_scan_a6_s5,multi_scan_a6_s6,multi_scan_a6_s7]);
impl_multi_scan_dispatches_for_eval!(A7,Eval7,Env7<L0,L1,L2,L3,L4,L5,L6>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6]; [multi_scan_a7_s2,multi_scan_a7_s3,multi_scan_a7_s4,multi_scan_a7_s5,multi_scan_a7_s6,multi_scan_a7_s7]);
impl_multi_scan_dispatches_for_eval!(A8,Eval8,Env8<L0,L1,L2,L3,L4,L5,L6,L7>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7]; [multi_scan_a8_s2,multi_scan_a8_s3,multi_scan_a8_s4,multi_scan_a8_s5,multi_scan_a8_s6,multi_scan_a8_s7]);

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
    Dispatch<A1, S1>: InclusiveScanDispatch<
            R,
            crate::Column<Item>,
            DeviceSliceMut<Item>,
            Item,
            Env1<Item>,
            Env1<Item>,
            Op,
        >,
{
    <Dispatch<A1, S1> as InclusiveScanDispatch<
        R,
        crate::Column<Item>,
        DeviceSliceMut<Item>,
        Item,
        Env1<Item>,
        Env1<Item>,
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
    Dispatch<Input::ReadArity, Output::StorageArity>:
        InclusiveScanDispatch<R, Input, Output, Input::Item, Input::Slots, Output::Slots, Op>,
{
    <Dispatch<Input::ReadArity, Output::StorageArity> as InclusiveScanDispatch<
        R,
        Input,
        Output,
        Input::Item,
        Input::Slots,
        Output::Slots,
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
    Input::Item: StorageLayout,
    Op: ReductionOp<Input::Item>,
    Adjacent<Input, Op>:
        ReadExpression<Item = Input::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: WriteFrom<Input::Item>,
    Dispatch<<Adjacent<Input, Op> as ReadExpression>::ReadArity, Output::StorageArity>:
        MaterializeDispatch<
                R,
                Adjacent<Input, Op>,
                Output,
                <Adjacent<Input, Op> as LowerReadExpression>::Slots,
                Output::Slots,
            >,
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
    Input::Item: CanonicalAlloc<R, CanonicalStorage = Input::Storage>,
    Input::Storage: CanonicalStorage<R>,
    Input::SemanticRead: LowerReadExpression + StageRead<R, Env0>,
    <Input::Storage as CanonicalStorage<R>>::Write: LowerOutputExpression + StageOutput<R, Env0>,
    <<Input::Storage as CanonicalStorage<R>>::Write as OutputExpression>::Item:
        WriteFrom<Input::Item>,
    Dispatch<
        <Input::SemanticRead as ReadExpression>::ReadArity,
        <<Input::Storage as CanonicalStorage<R>>::Write as OutputExpression>::StorageArity,
    >: InclusiveScanDispatch<
            R,
            Input::SemanticRead,
            <Input::Storage as CanonicalStorage<R>>::Write,
            Input::Item,
            <Input::SemanticRead as LowerReadExpression>::Slots,
            <<Input::Storage as CanonicalStorage<R>>::Write as LowerOutputExpression>::Slots,
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
        inclusive_scan(exec, Input::semantic_read(&prefixed), op, scanned.write())?;
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
    use crate::{CanonicalStorage, Counting, Permute, Transform, UnaryOp, Zip};
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
