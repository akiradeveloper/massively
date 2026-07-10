//! Segmented scan over canonical storage leaves and a separate head-flag control.

use cubecl::prelude::*;

use crate::{
    DeviceVec, Error, Executor, MStorageElement, ReductionOp, StorageLayout,
    storage::{
        Decompose, Last, LoadLeaves2, LoadLeaves3, LoadLeaves4, LoadLeaves5, LoadLeaves6,
        LoadLeaves7, LoadMutLeaves2, LoadMutLeaves3, LoadMutLeaves4, LoadMutLeaves5,
        LoadMutLeaves6, LoadMutLeaves7, More, MutableLeaves, PlaneShuffleLeaves, Recompose,
        StoreLeaves2, StoreLeaves2Expand, StoreLeaves3, StoreLeaves3Expand, StoreLeaves4,
        StoreLeaves4Expand, StoreLeaves5, StoreLeaves5Expand, StoreLeaves6, StoreLeaves6Expand,
        StoreLeaves7, StoreLeaves7Expand,
    },
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked, explicit_define)]
fn segmented_scan_s1_kernel<Item: CubePrimitive, Op: ReductionOp<Item>>(
    input: &[Item],
    flags: &[u32],
    len: &[u32],
    output: &mut [Item],
    local_flags: &mut [u32],
    block_sums: &mut [Item],
    block_flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let block = CUBE_POS as usize;
    let cube_dim = BLOCK_SIZE as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let value = RuntimeCell::<Item>::new(input[0]);
    let segment = RuntimeCell::<u32>::new(0u32);
    let valid = RuntimeCell::<u32>::new(0u32);
    if global < logical_len {
        value.store(input[global]);
        segment.store(flags[global]);
        valid.store(1u32);
    }

    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left = plane_shuffle_up(value.read(), offset.read());
        let left_segment = plane_shuffle_up(segment.read(), offset.read());
        let left_valid = plane_shuffle_up(valid.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
            if valid.read() != 0u32 {
                if segment.read() == 0u32 {
                    value.store(Op::apply(left, value.read()));
                }
                segment.store(left_segment | segment.read());
            } else {
                value.store(left);
                segment.store(left_segment);
                valid.store(1u32);
            }
        }
        offset.store(offset.read() * 2u32);
    }

    let mut plane_values = Shared::<[Item]>::new_slice(cube_dim);
    let mut plane_segments = Shared::<[u32]>::new_slice(cube_dim);
    let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        plane_values[PLANE_POS as usize] = value.read();
        plane_segments[PLANE_POS as usize] = segment.read();
        plane_valid[PLANE_POS as usize] = valid.read();
    }
    sync_cube();

    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let prefix = RuntimeCell::<Item>::new(plane_values[0]);
        let prefix_segment = RuntimeCell::<u32>::new(plane_segments[0]);
        let prefix_valid = RuntimeCell::<u32>::new(plane_valid[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            if plane_valid[index] != 0u32 {
                if prefix_valid.read() != 0u32 {
                    if plane_segments[index] == 0u32 {
                        prefix.store(Op::apply(prefix.read(), plane_values[index]));
                    } else {
                        prefix.store(plane_values[index]);
                    }
                    prefix_segment.store(prefix_segment.read() | plane_segments[index]);
                } else {
                    prefix.store(plane_values[index]);
                    prefix_segment.store(plane_segments[index]);
                    prefix_valid.store(1u32);
                }
            }
            plane_values[index] = prefix.read();
            plane_segments[index] = prefix_segment.read();
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();

    if PLANE_POS > 0u32 && valid.read() != 0u32 {
        let prefix_index = PLANE_POS as usize - 1usize;
        if segment.read() == 0u32 {
            value.store(Op::apply(plane_values[prefix_index], value.read()));
        }
        segment.store(plane_segments[prefix_index] | segment.read());
    }

    if global < logical_len {
        output[global] = value.read();
        local_flags[global] = segment.read();
    }
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let last_plane = plane_count as usize - 1usize;
        block_sums[block] = plane_values[last_plane];
        block_flags[block] = plane_segments[last_plane];
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn segmented_prefix_s1_kernel<Item: CubePrimitive, Op: ReductionOp<Item>>(
    prefixes: &[Item],
    local_flags: &[u32],
    len: &[u32],
    output: &mut [Item],
) {
    let block = CUBE_POS as usize;
    let global = block * BLOCK_SIZE as usize + UNIT_POS as usize;
    if block > 0usize && global < len[0] as usize && local_flags[global] == 0u32 {
        output[global] = Op::apply(prefixes[block - 1usize], output[global]);
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn segmented_exclusive_s1_kernel<Item: CubePrimitive, Op: ReductionOp<Item>>(
    inclusive: &[Item],
    flags: &[u32],
    init: &[Item],
    len: &[u32],
    output: &mut [Item],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        output[index] = if flags[index] != 0u32 {
            init[0]
        } else {
            Op::apply(init[0], inclusive[index - 1usize])
        };
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn apply_init_s1_kernel<Item: CubePrimitive, Op: ReductionOp<Item>>(
    inclusive: &[Item],
    init: &[Item],
    len: &[u32],
    output: &mut [Item],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        output[index] = Op::apply(init[0], inclusive[index]);
    }
}

macro_rules! define_segmented_multi_kernel {
    ($name:ident,$load_trait:ident,$store_trait:ident; [$( $out_ty:ident:$input:ident:$output:ident:$shared:ident:$sum:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $out_ty: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>
                + MutableLeaves
                + PlaneShuffleLeaves,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $input: &[$out_ty], )+
            flags: &[u32],
            len: &[u32],
            zero_offsets: &[u32],
            $( $output: &mut [$out_ty], )+
            local_flags: &mut [u32],
            $( $sum: &mut [$out_ty], )+
            block_flags: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let block = CUBE_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let global = block * cube_dim + unit;
            let logical_len = len[0] as usize;
            $( let mut $shared = Shared::<[$out_ty]>::new_slice(cube_dim); )+
            let mut segments = Shared::<[u32]>::new_slice(cube_dim);
            let mut valid = Shared::<[u32]>::new_slice(cube_dim);
            let cells = <Leaves as MutableLeaves>::into_cells(Leaves::load(
                $( $input, )+ zero_offsets, 0,
            ));
            let segment = RuntimeCell::<u32>::new(0u32);
            let is_valid = RuntimeCell::<u32>::new(0u32);
            if global < logical_len {
                <Leaves as MutableLeaves>::store(
                    &cells,
                    Leaves::load($( $input, )+ zero_offsets, global),
                );
                segment.store(flags[global]);
                is_valid.store(1u32);
            }

            let offset = RuntimeCell::<u32>::new(1u32);
            while offset.read() < PLANE_DIM {
                let left_cells = <Leaves as MutableLeaves>::into_cells(
                    Leaves::shuffle_leaves_up(
                        <Leaves as MutableLeaves>::read(&cells),
                        offset.read(),
                    ),
                );
                let left_segment = plane_shuffle_up(segment.read(), offset.read());
                let left_valid = plane_shuffle_up(is_valid.read(), offset.read());
                if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
                    if is_valid.read() != 0u32 {
                        if segment.read() == 0u32 {
                            let combined = Layout::decompose(Op::apply(
                                Layout::recompose(<Leaves as MutableLeaves>::read(&left_cells)),
                                Layout::recompose(<Leaves as MutableLeaves>::read(&cells)),
                            ));
                            <Leaves as MutableLeaves>::store(&cells, combined);
                        }
                        segment.store(left_segment | segment.read());
                    } else {
                        <Leaves as MutableLeaves>::store(
                            &cells,
                            <Leaves as MutableLeaves>::read(&left_cells),
                        );
                        segment.store(left_segment);
                        is_valid.store(1u32);
                    }
                }
                offset.store(offset.read() * 2u32);
            }

            if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
                <Leaves as MutableLeaves>::read(&cells).store(
                    $( &mut $shared, )+ zero_offsets, PLANE_POS as usize,
                );
                segments[PLANE_POS as usize] = segment.read();
                valid[PLANE_POS as usize] = is_valid.read();
            }
            sync_cube();

            if unit == 0usize {
                let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                let plane_cells = <Leaves as MutableLeaves>::into_cells(Leaves::load(
                    $( &$shared, )+ zero_offsets, 0usize,
                ));
                let plane_segment = RuntimeCell::<u32>::new(segments[0]);
                let plane_is_valid = RuntimeCell::<u32>::new(valid[0]);
                let plane = RuntimeCell::<u32>::new(1u32);
                while plane.read() < plane_count {
                    let index = plane.read() as usize;
                    if valid[index] != 0u32 {
                        if plane_is_valid.read() != 0u32 {
                            if segments[index] == 0u32 {
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
                            }
                            plane_segment.store(plane_segment.read() | segments[index]);
                        } else {
                            <Leaves as MutableLeaves>::store(
                                &plane_cells,
                                Leaves::load($( &$shared, )+ zero_offsets, index),
                            );
                            plane_segment.store(segments[index]);
                            plane_is_valid.store(1u32);
                        }
                    }
                    <Leaves as MutableLeaves>::read(&plane_cells).store(
                        $( &mut $shared, )+ zero_offsets, index,
                    );
                    segments[index] = plane_segment.read();
                    plane.store(plane.read() + 1u32);
                }
            }
            sync_cube();

            if PLANE_POS > 0u32 && is_valid.read() != 0u32 {
                let prefix_index = PLANE_POS as usize - 1usize;
                if segment.read() == 0u32 {
                    let combined = Layout::decompose(Op::apply(
                        Layout::recompose(Leaves::load(
                            $( &$shared, )+ zero_offsets, prefix_index,
                        )),
                        Layout::recompose(<Leaves as MutableLeaves>::read(&cells)),
                    ));
                    <Leaves as MutableLeaves>::store(&cells, combined);
                }
                segment.store(segments[prefix_index] | segment.read());
            }

            if global < logical_len {
                <Leaves as MutableLeaves>::read(&cells).store(
                    $( $output, )+ zero_offsets, global,
                );
                local_flags[global] = segment.read();
            }
            if unit == 0usize {
                let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                let last_plane = plane_count as usize - 1usize;
                Leaves::load($( &$shared, )+ zero_offsets, last_plane).store(
                    $( $sum, )+ zero_offsets, block,
                );
                block_flags[block] = segments[last_plane];
            }
        }
    };
}

macro_rules! define_segmented_multi_apply_kernels {
    ($prefix_name:ident,$exclusive_name:ident,$init_name:ident,$load_trait:ident,$load_mut_trait:ident,$store_trait:ident; [$( $out_ty:ident:$prefix:ident:$inclusive:ident:$init:ident:$output:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $prefix_name<
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
            local_flags: &[u32],
            len: &[u32],
            zero_offsets: &[u32],
            $( $output: &mut [$out_ty], )+
        ) {
            let block = CUBE_POS as usize;
            let global = block * BLOCK_SIZE as usize + UNIT_POS as usize;
            if block > 0usize && global < len[0] as usize && local_flags[global] == 0u32 {
                let prefix = Layout::recompose(Leaves::load(
                    $( $prefix, )+ zero_offsets, block - 1usize,
                ));
                let current = Layout::recompose(Leaves::load_mut(
                    $( $output, )+ zero_offsets, global,
                ));
                Layout::decompose(Op::apply(prefix, current)).store(
                    $( $output, )+ zero_offsets, global,
                );
            }
        }

        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $exclusive_name<
            Item: CubeType + Send + Sync + 'static,
            $( $out_ty: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $inclusive: &[$out_ty], )+
            flags: &[u32],
            $( $init: &[$out_ty], )+
            len: &[u32],
            zero_offsets: &[u32],
            $( $output: &mut [$out_ty], )+
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                if flags[index] != 0u32 {
                    Leaves::load($( $init, )+ zero_offsets, 0).store(
                        $( $output, )+ zero_offsets, index,
                    );
                } else {
                    let initial = Layout::recompose(Leaves::load(
                        $( $init, )+ zero_offsets, 0,
                    ));
                    let previous = Layout::recompose(Leaves::load(
                        $( $inclusive, )+ zero_offsets, index - 1usize,
                    ));
                    Layout::decompose(Op::apply(initial, previous)).store(
                        $( $output, )+ zero_offsets, index,
                    );
                }
            }
        }

        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $init_name<
            Item: CubeType + Send + Sync + 'static,
            $( $out_ty: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $inclusive: &[$out_ty], )+
            $( $init: &[$out_ty], )+
            len: &[u32],
            zero_offsets: &[u32],
            $( $output: &mut [$out_ty], )+
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                let initial = Layout::recompose(Leaves::load(
                    $( $init, )+ zero_offsets, 0,
                ));
                let current = Layout::recompose(Leaves::load(
                    $( $inclusive, )+ zero_offsets, index,
                ));
                Layout::decompose(Op::apply(initial, current)).store(
                    $( $output, )+ zero_offsets, index,
                );
            }
        }
    };
}

define_segmented_multi_kernel!(segmented_scan_s2_kernel,LoadLeaves2,StoreLeaves2; [O0:input0:output0:shared0:sum0,O1:input1:output1:shared1:sum1]);
define_segmented_multi_kernel!(segmented_scan_s3_kernel,LoadLeaves3,StoreLeaves3; [O0:input0:output0:shared0:sum0,O1:input1:output1:shared1:sum1,O2:input2:output2:shared2:sum2]);
define_segmented_multi_kernel!(segmented_scan_s4_kernel,LoadLeaves4,StoreLeaves4; [O0:input0:output0:shared0:sum0,O1:input1:output1:shared1:sum1,O2:input2:output2:shared2:sum2,O3:input3:output3:shared3:sum3]);
define_segmented_multi_kernel!(segmented_scan_s5_kernel,LoadLeaves5,StoreLeaves5; [O0:input0:output0:shared0:sum0,O1:input1:output1:shared1:sum1,O2:input2:output2:shared2:sum2,O3:input3:output3:shared3:sum3,O4:input4:output4:shared4:sum4]);
define_segmented_multi_kernel!(segmented_scan_s6_kernel,LoadLeaves6,StoreLeaves6; [O0:input0:output0:shared0:sum0,O1:input1:output1:shared1:sum1,O2:input2:output2:shared2:sum2,O3:input3:output3:shared3:sum3,O4:input4:output4:shared4:sum4,O5:input5:output5:shared5:sum5]);
define_segmented_multi_kernel!(segmented_scan_s7_kernel,LoadLeaves7,StoreLeaves7; [O0:input0:output0:shared0:sum0,O1:input1:output1:shared1:sum1,O2:input2:output2:shared2:sum2,O3:input3:output3:shared3:sum3,O4:input4:output4:shared4:sum4,O5:input5:output5:shared5:sum5,O6:input6:output6:shared6:sum6]);

define_segmented_multi_apply_kernels!(segmented_prefix_s2_kernel,segmented_exclusive_s2_kernel,apply_init_s2_kernel,LoadLeaves2,LoadMutLeaves2,StoreLeaves2; [O0:prefix0:inclusive0:init0:output0,O1:prefix1:inclusive1:init1:output1]);
define_segmented_multi_apply_kernels!(segmented_prefix_s3_kernel,segmented_exclusive_s3_kernel,apply_init_s3_kernel,LoadLeaves3,LoadMutLeaves3,StoreLeaves3; [O0:prefix0:inclusive0:init0:output0,O1:prefix1:inclusive1:init1:output1,O2:prefix2:inclusive2:init2:output2]);
define_segmented_multi_apply_kernels!(segmented_prefix_s4_kernel,segmented_exclusive_s4_kernel,apply_init_s4_kernel,LoadLeaves4,LoadMutLeaves4,StoreLeaves4; [O0:prefix0:inclusive0:init0:output0,O1:prefix1:inclusive1:init1:output1,O2:prefix2:inclusive2:init2:output2,O3:prefix3:inclusive3:init3:output3]);
define_segmented_multi_apply_kernels!(segmented_prefix_s5_kernel,segmented_exclusive_s5_kernel,apply_init_s5_kernel,LoadLeaves5,LoadMutLeaves5,StoreLeaves5; [O0:prefix0:inclusive0:init0:output0,O1:prefix1:inclusive1:init1:output1,O2:prefix2:inclusive2:init2:output2,O3:prefix3:inclusive3:init3:output3,O4:prefix4:inclusive4:init4:output4]);
define_segmented_multi_apply_kernels!(segmented_prefix_s6_kernel,segmented_exclusive_s6_kernel,apply_init_s6_kernel,LoadLeaves6,LoadMutLeaves6,StoreLeaves6; [O0:prefix0:inclusive0:init0:output0,O1:prefix1:inclusive1:init1:output1,O2:prefix2:inclusive2:init2:output2,O3:prefix3:inclusive3:init3:output3,O4:prefix4:inclusive4:init4:output4,O5:prefix5:inclusive5:init5:output5]);
define_segmented_multi_apply_kernels!(segmented_prefix_s7_kernel,segmented_exclusive_s7_kernel,apply_init_s7_kernel,LoadLeaves7,LoadMutLeaves7,StoreLeaves7; [O0:prefix0:inclusive0:init0:output0,O1:prefix1:inclusive1:init1:output1,O2:prefix2:inclusive2:init2:output2,O3:prefix3:inclusive3:init3:output3,O4:prefix4:inclusive4:init4:output4,O5:prefix5:inclusive5:init5:output5,O6:prefix6:inclusive6:init6:output6]);

fn checked_u32(len: usize) -> Result<u32, Error> {
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })
}

fn segmented_storage1<R, Item, Op>(
    exec: &Executor<R>,
    input: &DeviceVec<R, Item>,
    flags: &DeviceVec<R, u32>,
    output: &DeviceVec<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: MStorageElement,
    Op: ReductionOp<Item>,
{
    let len = input.len();
    if flags.len() != len || output.len() != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: flags.len().min(output.len()),
        });
    }
    if len == 0 {
        return Ok(());
    }
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let local_flags = exec.alloc::<u32>(len);
    let block_sums = exec.alloc_column::<Item>(blocks);
    let block_flags = exec.alloc::<u32>(blocks);
    let len_handle = exec
        .client()
        .create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
    let count = crate::launch::cube_count_1d(blocks)?;
    unsafe {
        segmented_scan_s1_kernel::launch_unchecked::<Item, Op, R>(
            exec.client(),
            count.clone(),
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(input.handle.clone(), len),
            BufferArg::from_raw_parts(flags.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(output.handle.clone(), len),
            BufferArg::from_raw_parts(local_flags.handle.clone(), len),
            BufferArg::from_raw_parts(block_sums.handle.clone(), blocks),
            BufferArg::from_raw_parts(block_flags.handle.clone(), blocks),
        );
    }
    if blocks > 1 {
        let prefixes = exec.alloc_column::<Item>(blocks);
        segmented_storage1::<R, Item, Op>(exec, &block_sums, &block_flags, &prefixes)?;
        unsafe {
            segmented_prefix_s1_kernel::launch_unchecked::<Item, Op, R>(
                exec.client(),
                count,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(prefixes.handle.clone(), blocks),
                BufferArg::from_raw_parts(local_flags.handle.clone(), len),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(output.handle.clone(), len),
            );
        }
    }
    Ok(())
}

macro_rules! define_segmented_storage_host {
    (
        $name:ident,$arity:ty,$leaves:ty,$scan_kernel:ident,$prefix_kernel:ident,
        $load_trait:ident,$load_mut_trait:ident,$store_trait:ident;
        [$( $out_ty:ident:$input:ident:$output:ident:$sum:ident:$prefix:ident ),+]
    ) => {
        fn $name<R, Item, Op, $( $out_ty ),+>(
            exec: &Executor<R>,
            $( $input: &DeviceVec<R, $out_ty>, )+
            flags: &DeviceVec<R, u32>,
            $( $output: &DeviceVec<R, $out_ty>, )+
        ) -> Result<(), Error>
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $arity, StorageLeaves = $leaves>
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout: Decompose<Item, Leaves = $leaves>
                + Recompose<Item, Leaves = $leaves>,
            Op: ReductionOp<Item>,
            $( $out_ty: MStorageElement, )+
            $leaves: $load_trait<$( $out_ty ),+>
                + $load_mut_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>
                + Send
                + Sync
                + 'static,
        {
            let len = define_segmented_storage_host!(@first_len $( $input ),+);
            if flags.len() != len || define_segmented_storage_host!(@first_len $( $output ),+) != len {
                return Err(Error::LengthMismatch {
                    left: len,
                    right: flags.len().min(define_segmented_storage_host!(@first_len $( $output ),+)),
                });
            }
            if len == 0 {
                return Ok(());
            }
            let blocks = len.div_ceil(BLOCK_SIZE as usize);
            let local_flags = exec.alloc::<u32>(len);
            $( let $sum = exec.alloc_column::<$out_ty>(blocks); )+
            let block_flags = exec.alloc::<u32>(blocks);
            let len_handle = exec.client().create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
            let zero_offsets = vec![$({ let _ = stringify!($out_ty); 0u32 }),+];
            let zero_handle = exec.client().create_from_slice(u32::as_bytes(&zero_offsets));
            let count = crate::launch::cube_count_1d(blocks)?;
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
                    BufferArg::from_raw_parts(flags.handle.clone(), len),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(zero_handle.clone(), zero_offsets.len()),
                    $( BufferArg::from_raw_parts($output.handle.clone(), len), )+
                    BufferArg::from_raw_parts(local_flags.handle.clone(), len),
                    $( BufferArg::from_raw_parts($sum.handle.clone(), blocks), )+
                    BufferArg::from_raw_parts(block_flags.handle.clone(), blocks),
                );
            }
            if blocks > 1 {
                $( let $prefix = exec.alloc_column::<$out_ty>(blocks); )+
                $name::<R, Item, Op, $( $out_ty ),+>(
                    exec,
                    $( &$sum, )+
                    &block_flags,
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
                        BufferArg::from_raw_parts(local_flags.handle.clone(), len),
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(zero_handle, zero_offsets.len()),
                        $( BufferArg::from_raw_parts($output.handle.clone(), len), )+
                    );
                }
            }
            Ok(())
        }
    };
    (@first_len $first:ident $(, $rest:ident)*) => { $first.len() };
}

define_segmented_storage_host!(segmented_storage2,crate::S2,More<O0,Last<O1>>,segmented_scan_s2_kernel,segmented_prefix_s2_kernel,LoadLeaves2,LoadMutLeaves2,StoreLeaves2; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1]);
define_segmented_storage_host!(segmented_storage3,crate::S3,More<O0,More<O1,Last<O2>>>,segmented_scan_s3_kernel,segmented_prefix_s3_kernel,LoadLeaves3,LoadMutLeaves3,StoreLeaves3; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2]);
define_segmented_storage_host!(segmented_storage4,crate::S4,More<O0,More<O1,More<O2,Last<O3>>>>,segmented_scan_s4_kernel,segmented_prefix_s4_kernel,LoadLeaves4,LoadMutLeaves4,StoreLeaves4; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3]);
define_segmented_storage_host!(segmented_storage5,crate::S5,More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>,segmented_scan_s5_kernel,segmented_prefix_s5_kernel,LoadLeaves5,LoadMutLeaves5,StoreLeaves5; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4]);
define_segmented_storage_host!(segmented_storage6,crate::S6,More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>,segmented_scan_s6_kernel,segmented_prefix_s6_kernel,LoadLeaves6,LoadMutLeaves6,StoreLeaves6; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5]);
define_segmented_storage_host!(segmented_storage7,crate::S7,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>,segmented_scan_s7_kernel,segmented_prefix_s7_kernel,LoadLeaves7,LoadMutLeaves7,StoreLeaves7; [O0:input0:output0:sum0:prefix0,O1:input1:output1:sum1:prefix1,O2:input2:output2:sum2:prefix2,O3:input3:output3:sum3:prefix3,O4:input4:output4:sum4:prefix4,O5:input5:output5:sum5:prefix5,O6:input6:output6:sum6:prefix6]);

/// Canonical payload storage that supports segmented scan and its init
/// post-processing operations.
#[doc(hidden)]
pub trait SegmentedStorage<R: Runtime, Item: StorageLayout, Op: ReductionOp<Item>> {
    fn segmented_len(&self) -> usize;
    fn segmented_inclusive(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
        output: &Self,
    ) -> Result<(), Error>;
    fn segmented_exclusive(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
        init: &Self,
        output: &Self,
    ) -> Result<(), Error>;
    fn apply_init(&self, exec: &Executor<R>, init: &Self, output: &Self) -> Result<(), Error>;
}

impl<R, Item, Op> SegmentedStorage<R, Item, Op> for DeviceVec<R, Item>
where
    R: Runtime,
    Item: MStorageElement + StorageLayout<StorageArity = crate::S1>,
    Op: ReductionOp<Item>,
{
    fn segmented_len(&self) -> usize {
        self.len()
    }

    fn segmented_inclusive(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
        output: &Self,
    ) -> Result<(), Error> {
        segmented_storage1::<R, Item, Op>(exec, self, flags, output)
    }

    fn segmented_exclusive(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
        init: &Self,
        output: &Self,
    ) -> Result<(), Error> {
        let len = self.len();
        if flags.len() != len || output.len() != len || init.len() != 1 {
            return Err(Error::LengthMismatch {
                left: len,
                right: flags.len().min(output.len()),
            });
        }
        if len == 0 {
            return Ok(());
        }
        let len_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
        unsafe {
            segmented_exclusive_s1_kernel::launch_unchecked::<Item, Op, R>(
                exec.client(),
                crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.handle.clone(), len),
                BufferArg::from_raw_parts(flags.handle.clone(), len),
                BufferArg::from_raw_parts(init.handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(output.handle.clone(), len),
            );
        }
        Ok(())
    }

    fn apply_init(&self, exec: &Executor<R>, init: &Self, output: &Self) -> Result<(), Error> {
        let len = self.len();
        if output.len() != len || init.len() != 1 {
            return Err(Error::LengthMismatch {
                left: len,
                right: output.len(),
            });
        }
        if len == 0 {
            return Ok(());
        }
        let len_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
        unsafe {
            apply_init_s1_kernel::launch_unchecked::<Item, Op, R>(
                exec.client(),
                crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.handle.clone(), len),
                BufferArg::from_raw_parts(init.handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(output.handle.clone(), len),
            );
        }
        Ok(())
    }
}

macro_rules! nested_field {
    ($root:ident; $first:tt $( . $rest:tt )*) => {
        $root.$first $( .$rest )*
    };
}

macro_rules! impl_segmented_storage {
    (
        $storage:ty,$arity:ty,$leaves:ty,$host:ident,$exclusive_kernel:ident,$init_kernel:ident,
        $load_trait:ident,$load_mut_trait:ident,$store_trait:ident;
        [$( $out_ty:ident:$first_field:tt $(.$rest_field:tt)* ),+]
    ) => {
        impl<R, Item, Op, $( $out_ty ),+> SegmentedStorage<R, Item, Op> for $storage
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $arity, StorageLeaves = $leaves>
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout: Decompose<Item, Leaves = $leaves>
                + Recompose<Item, Leaves = $leaves>,
            Op: ReductionOp<Item>,
            $( $out_ty: MStorageElement, )+
            $leaves: $load_trait<$( $out_ty ),+>
                + $load_mut_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>
                + Send
                + Sync
                + 'static,
        {
            fn segmented_len(&self) -> usize {
                let storage = self;
                impl_segmented_storage!(@first_len storage; $( $first_field $(.$rest_field)* ),+)
            }

            fn segmented_inclusive(
                &self,
                exec: &Executor<R>,
                flags: &DeviceVec<R, u32>,
                output: &Self,
            ) -> Result<(), Error> {
                let input = self;
                $host::<R, Item, Op, $( $out_ty ),+>(
                    exec,
                    $( &nested_field!(input; $first_field $(.$rest_field)*), )+
                    flags,
                    $( &nested_field!(output; $first_field $(.$rest_field)*), )+
                )
            }

            fn segmented_exclusive(
                &self,
                exec: &Executor<R>,
                flags: &DeviceVec<R, u32>,
                init: &Self,
                output: &Self,
            ) -> Result<(), Error> {
                let inclusive = self;
                let len = impl_segmented_storage!(@first_len inclusive; $( $first_field $(.$rest_field)* ),+);
                let output_len = impl_segmented_storage!(@first_len output; $( $first_field $(.$rest_field)* ),+);
                let init_len = impl_segmented_storage!(@first_len init; $( $first_field $(.$rest_field)* ),+);
                if flags.len() != len || output_len != len || init_len != 1 {
                    return Err(Error::LengthMismatch { left: len, right: flags.len().min(output_len) });
                }
                if len == 0 {
                    return Ok(());
                }
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
                let zero_offsets = vec![$({ let _ = stringify!($out_ty); 0u32 }),+];
                let zero_handle = exec.client().create_from_slice(u32::as_bytes(&zero_offsets));
                unsafe {
                    $exclusive_kernel::launch_unchecked::<
                        Item,
                        $( $out_ty, )+
                        $leaves,
                        Item::DeviceLayout,
                        Op,
                        R,
                    >(
                        exec.client(),
                        crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(nested_field!(inclusive; $first_field $(.$rest_field)*).handle.clone(), len), )+
                        BufferArg::from_raw_parts(flags.handle.clone(), len),
                        $( BufferArg::from_raw_parts(nested_field!(init; $first_field $(.$rest_field)*).handle.clone(), 1), )+
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(zero_handle, zero_offsets.len()),
                        $( BufferArg::from_raw_parts(nested_field!(output; $first_field $(.$rest_field)*).handle.clone(), len), )+
                    );
                }
                Ok(())
            }

            fn apply_init(
                &self,
                exec: &Executor<R>,
                init: &Self,
                output: &Self,
            ) -> Result<(), Error> {
                let inclusive = self;
                let len = impl_segmented_storage!(@first_len inclusive; $( $first_field $(.$rest_field)* ),+);
                let output_len = impl_segmented_storage!(@first_len output; $( $first_field $(.$rest_field)* ),+);
                let init_len = impl_segmented_storage!(@first_len init; $( $first_field $(.$rest_field)* ),+);
                if output_len != len || init_len != 1 {
                    return Err(Error::LengthMismatch { left: len, right: output_len });
                }
                if len == 0 {
                    return Ok(());
                }
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
                let zero_offsets = vec![$({ let _ = stringify!($out_ty); 0u32 }),+];
                let zero_handle = exec.client().create_from_slice(u32::as_bytes(&zero_offsets));
                unsafe {
                    $init_kernel::launch_unchecked::<
                        Item,
                        $( $out_ty, )+
                        $leaves,
                        Item::DeviceLayout,
                        Op,
                        R,
                    >(
                        exec.client(),
                        crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(nested_field!(inclusive; $first_field $(.$rest_field)*).handle.clone(), len), )+
                        $( BufferArg::from_raw_parts(nested_field!(init; $first_field $(.$rest_field)*).handle.clone(), 1), )+
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(zero_handle, zero_offsets.len()),
                        $( BufferArg::from_raw_parts(nested_field!(output; $first_field $(.$rest_field)*).handle.clone(), len), )+
                    );
                }
                Ok(())
            }
        }
    };
    (@first_len $root:ident; $first:tt $(.$first_rest:tt)* $(, $rest:tt $(.$rest_tail:tt)*)*) => {
        nested_field!($root; $first $(.$first_rest)*).len()
    };
}

impl_segmented_storage!(crate::Zip<DeviceVec<R,O0>,DeviceVec<R,O1>>,crate::S2,More<O0,Last<O1>>,segmented_storage2,segmented_exclusive_s2_kernel,apply_init_s2_kernel,LoadLeaves2,LoadMutLeaves2,StoreLeaves2; [O0:0,O1:1]);
impl_segmented_storage!(crate::Zip<crate::Zip<DeviceVec<R,O0>,DeviceVec<R,O1>>,DeviceVec<R,O2>>,crate::S3,More<O0,More<O1,Last<O2>>>,segmented_storage3,segmented_exclusive_s3_kernel,apply_init_s3_kernel,LoadLeaves3,LoadMutLeaves3,StoreLeaves3; [O0:0.0,O1:0.1,O2:1]);
impl_segmented_storage!(crate::Zip<crate::Zip<crate::Zip<DeviceVec<R,O0>,DeviceVec<R,O1>>,DeviceVec<R,O2>>,DeviceVec<R,O3>>,crate::S4,More<O0,More<O1,More<O2,Last<O3>>>>,segmented_storage4,segmented_exclusive_s4_kernel,apply_init_s4_kernel,LoadLeaves4,LoadMutLeaves4,StoreLeaves4; [O0:0.0.0,O1:0.0.1,O2:0.1,O3:1]);
impl_segmented_storage!(crate::Zip<crate::Zip<crate::Zip<crate::Zip<DeviceVec<R,O0>,DeviceVec<R,O1>>,DeviceVec<R,O2>>,DeviceVec<R,O3>>,DeviceVec<R,O4>>,crate::S5,More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>,segmented_storage5,segmented_exclusive_s5_kernel,apply_init_s5_kernel,LoadLeaves5,LoadMutLeaves5,StoreLeaves5; [O0:0.0.0.0,O1:0.0.0.1,O2:0.0.1,O3:0.1,O4:1]);
impl_segmented_storage!(crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<DeviceVec<R,O0>,DeviceVec<R,O1>>,DeviceVec<R,O2>>,DeviceVec<R,O3>>,DeviceVec<R,O4>>,DeviceVec<R,O5>>,crate::S6,More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>,segmented_storage6,segmented_exclusive_s6_kernel,apply_init_s6_kernel,LoadLeaves6,LoadMutLeaves6,StoreLeaves6; [O0:0.0.0.0.0,O1:0.0.0.0.1,O2:0.0.0.1,O3:0.0.1,O4:0.1,O5:1]);
impl_segmented_storage!(crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<DeviceVec<R,O0>,DeviceVec<R,O1>>,DeviceVec<R,O2>>,DeviceVec<R,O3>>,DeviceVec<R,O4>>,DeviceVec<R,O5>>,DeviceVec<R,O6>>,crate::S7,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>,segmented_storage7,segmented_exclusive_s7_kernel,apply_init_s7_kernel,LoadLeaves7,LoadMutLeaves7,StoreLeaves7; [O0:0.0.0.0.0.0,O1:0.0.0.0.0.1,O2:0.0.0.0.1,O3:0.0.0.1,O4:0.0.1,O5:0.1,O6:1]);
