//! Fixed twelve-value-slot segmented scan implementation.

use cubecl::prelude::*;

use crate::output::{OutputBindings, StageOutput};
use crate::reduce::{ReductionOp, StageRead, StagedBindings};
use crate::storage::{
    Decompose, LoadMutPadded12, LoadPadded12, MutableLeaves, PlaneShuffleLeaves, Recompose,
    SharedLeaves, SharedLeavesExpand, StorePadded12, StorePadded12Expand,
};
use crate::{CanonicalStorage, DeviceVec, Error, Executor};

const BLOCK_SIZE: u32 = 256;

type FixedStorage<R, Item> = <Item as crate::CanonicalAlloc<R>>::CanonicalStorage;

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn segmented_scan_padded12<
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
        > + SharedLeaves
        + MutableLeaves
        + PlaneShuffleLeaves
        + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    input0: &[O0],
    input1: &[O1],
    input2: &[O2],
    input3: &[O3],
    input4: &[O4],
    input5: &[O5],
    input6: &[O6],
    input7: &[O7],
    input8: &[O8],
    input9: &[O9],
    input10: &[O10],
    input11: &[O11],
    flags: &[u32],
    len: &[u32],
    input_offsets: &[u32],
    output_offsets: &[u32],
    sum_offsets: &[u32],
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
    local_flags: &mut [u32],
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
    block_flags: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let block = CUBE_POS as usize;
    let cube_dim = BLOCK_SIZE as usize;
    let global = block * cube_dim + unit;
    let logical_len = len[0] as usize;
    let safe_global = if global < logical_len { global } else { 0usize };
    let cells = Leaves::into_cells(Leaves::load_padded(
        input0,
        input1,
        input2,
        input3,
        input4,
        input5,
        input6,
        input7,
        input8,
        input9,
        input10,
        input11,
        input_offsets,
        safe_global,
    ));
    let segment = RuntimeCell::<u32>::new(if global < logical_len {
        flags[global]
    } else {
        0u32
    });
    let valid = RuntimeCell::<u32>::new(if global < logical_len { 1u32 } else { 0u32 });

    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left_cells = Leaves::into_cells(Leaves::shuffle_leaves_up(
            Leaves::read(&cells),
            offset.read(),
        ));
        let left_segment = plane_shuffle_up(segment.read(), offset.read());
        let left_valid = plane_shuffle_up(valid.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
            if valid.read() != 0u32 {
                if segment.read() == 0u32 {
                    let combined = Layout::decompose(Op::apply(
                        Layout::recompose(Leaves::read(&left_cells)),
                        Layout::recompose(Leaves::read(&cells)),
                    ));
                    Leaves::store(&cells, combined);
                }
                segment.store(left_segment | segment.read());
            } else {
                Leaves::store(&cells, Leaves::read(&left_cells));
                segment.store(left_segment);
                valid.store(1u32);
            }
        }
        offset.store(offset.read() * 2u32);
    }

    let mut shared = Leaves::new_shared(cube_dim);
    let mut shared_segments = Shared::<[u32]>::new_slice(cube_dim);
    let mut shared_valid = Shared::<[u32]>::new_slice(cube_dim);
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        Leaves::read(&cells).store_shared(&mut shared, PLANE_POS as usize);
        shared_segments[PLANE_POS as usize] = segment.read();
        shared_valid[PLANE_POS as usize] = valid.read();
    }
    sync_cube();

    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let plane_cells = Leaves::into_cells(Leaves::load_shared(&shared, 0usize));
        let plane_segment = RuntimeCell::<u32>::new(shared_segments[0]);
        let plane_valid = RuntimeCell::<u32>::new(shared_valid[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            if shared_valid[index] != 0u32 {
                if plane_valid.read() != 0u32 {
                    if shared_segments[index] == 0u32 {
                        let combined = Layout::decompose(Op::apply(
                            Layout::recompose(Leaves::read(&plane_cells)),
                            Layout::recompose(Leaves::load_shared(&shared, index)),
                        ));
                        Leaves::store(&plane_cells, combined);
                    } else {
                        Leaves::store(&plane_cells, Leaves::load_shared(&shared, index));
                    }
                    plane_segment.store(plane_segment.read() | shared_segments[index]);
                } else {
                    Leaves::store(&plane_cells, Leaves::load_shared(&shared, index));
                    plane_segment.store(shared_segments[index]);
                    plane_valid.store(1u32);
                }
            }
            Leaves::read(&plane_cells).store_shared(&mut shared, index);
            shared_segments[index] = plane_segment.read();
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();

    if PLANE_POS > 0u32 && valid.read() != 0u32 {
        let prefix_index = PLANE_POS as usize - 1usize;
        if segment.read() == 0u32 {
            let combined = Layout::decompose(Op::apply(
                Layout::recompose(Leaves::load_shared(&shared, prefix_index)),
                Layout::recompose(Leaves::read(&cells)),
            ));
            Leaves::store(&cells, combined);
        }
        segment.store(shared_segments[prefix_index] | segment.read());
    }

    if global < logical_len {
        Leaves::read(&cells).store_padded(
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
            global,
        );
        local_flags[global] = segment.read();
    }
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let last_plane = plane_count as usize - 1usize;
        Leaves::load_shared(&shared, last_plane).store_padded(
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
            sum_offsets,
            block,
        );
        block_flags[block] = shared_segments[last_plane];
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn segmented_prefix_padded12<
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
        > + LoadMutPadded12
        + Send
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
    local_flags: &[u32],
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
    if block > 0usize && index < len[0] as usize && local_flags[index] == 0u32 {
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
        let current = Layout::recompose(Leaves::load_mut_padded(
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
        Layout::decompose(Op::apply(prefix, current)).store_padded(
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

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn segmented_exclusive_padded12<
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
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    inclusive0: &[O0],
    inclusive1: &[O1],
    inclusive2: &[O2],
    inclusive3: &[O3],
    inclusive4: &[O4],
    inclusive5: &[O5],
    inclusive6: &[O6],
    inclusive7: &[O7],
    inclusive8: &[O8],
    inclusive9: &[O9],
    inclusive10: &[O10],
    inclusive11: &[O11],
    flags: &[u32],
    init0: &[O0],
    init1: &[O1],
    init2: &[O2],
    init3: &[O3],
    init4: &[O4],
    init5: &[O5],
    init6: &[O6],
    init7: &[O7],
    init8: &[O8],
    init9: &[O9],
    init10: &[O10],
    init11: &[O11],
    len: &[u32],
    inclusive_offsets: &[u32],
    init_offsets: &[u32],
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
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        if index == 0usize || flags[index] != 0u32 {
            Leaves::load_padded(
                init0,
                init1,
                init2,
                init3,
                init4,
                init5,
                init6,
                init7,
                init8,
                init9,
                init10,
                init11,
                init_offsets,
                0usize,
            )
            .store_padded(
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
        } else {
            Layout::decompose(Op::apply(
                Layout::recompose(Leaves::load_padded(
                    init0,
                    init1,
                    init2,
                    init3,
                    init4,
                    init5,
                    init6,
                    init7,
                    init8,
                    init9,
                    init10,
                    init11,
                    init_offsets,
                    0usize,
                )),
                Layout::recompose(Leaves::load_padded(
                    inclusive0,
                    inclusive1,
                    inclusive2,
                    inclusive3,
                    inclusive4,
                    inclusive5,
                    inclusive6,
                    inclusive7,
                    inclusive8,
                    inclusive9,
                    inclusive10,
                    inclusive11,
                    inclusive_offsets,
                    index - 1usize,
                )),
            ))
            .store_padded(
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
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn apply_init_padded12<
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
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    inclusive0: &[O0],
    inclusive1: &[O1],
    inclusive2: &[O2],
    inclusive3: &[O3],
    inclusive4: &[O4],
    inclusive5: &[O5],
    inclusive6: &[O6],
    inclusive7: &[O7],
    inclusive8: &[O8],
    inclusive9: &[O9],
    inclusive10: &[O10],
    inclusive11: &[O11],
    init0: &[O0],
    init1: &[O1],
    init2: &[O2],
    init3: &[O3],
    init4: &[O4],
    init5: &[O5],
    init6: &[O6],
    init7: &[O7],
    init8: &[O8],
    init9: &[O9],
    init10: &[O10],
    init11: &[O11],
    len: &[u32],
    inclusive_offsets: &[u32],
    init_offsets: &[u32],
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
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let initial = Layout::recompose(Leaves::load_padded(
            init0,
            init1,
            init2,
            init3,
            init4,
            init5,
            init6,
            init7,
            init8,
            init9,
            init10,
            init11,
            init_offsets,
            0usize,
        ));
        let value = Layout::recompose(Leaves::load_padded(
            inclusive0,
            inclusive1,
            inclusive2,
            inclusive3,
            inclusive4,
            inclusive5,
            inclusive6,
            inclusive7,
            inclusive8,
            inclusive9,
            inclusive10,
            inclusive11,
            inclusive_offsets,
            index,
        ));
        Layout::decompose(Op::apply(initial, value)).store_padded(
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

fn checked_len_handle<R: Runtime>(
    exec: &Executor<R>,
    len: usize,
) -> Result<cubecl::server::Handle, Error> {
    let len = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    Ok(exec.client().create_from_slice(u32::as_bytes(&[len])))
}

fn stage_read<R, Read>(exec: &Executor<R>, read: &Read) -> Result<StagedBindings, Error>
where
    R: Runtime,
    Read: StageRead<R, crate::read::Env0>,
{
    let mut bindings = StagedBindings::new();
    read.stage_at(exec.client(), exec.id(), &mut bindings)?;
    bindings.pad_to_thirteen(exec.client());
    Ok(bindings)
}

fn stage_write<R, Write>(exec: &Executor<R>, write: &Write) -> Result<OutputBindings, Error>
where
    R: Runtime,
    Write: StageOutput<R, crate::read::Env0>,
{
    let mut bindings = OutputBindings::new();
    write.stage_output(exec.id(), &mut bindings)?;
    bindings.pad_to_twelve(exec.client());
    Ok(bindings)
}

fn segmented_inclusive_fixed<R, Item, Op>(
    exec: &Executor<R>,
    input: &FixedStorage<R, Item>,
    flags: &DeviceVec<R, u32>,
    output: &FixedStorage<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::api::iter::MItem<R>,
    Op: ReductionOp<Item>,
{
    let len = input.len()?;
    let output_len = output.len()?;
    if flags.len() != len || output_len != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: flags.len().min(output_len),
        });
    }
    if len == 0 {
        return Ok(());
    }

    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let local_flags = exec.alloc_column::<u32>(len);
    let block_sums = exec.alloc_canonical::<Item>(blocks);
    let block_flags = exec.alloc_column::<u32>(blocks);
    let input_read = input.read();
    let output_write = output.write();
    let sum_write = block_sums.write();
    let input_bindings = stage_read(exec, &input_read)?;
    let output_bindings = stage_write(exec, &output_write)?;
    let sum_bindings = stage_write(exec, &sum_write)?;
    let input_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&input_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    let sum_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&sum_bindings.offsets));
    let len_handle = checked_len_handle(exec, len)?;

    unsafe {
        segmented_scan_padded12::launch_unchecked::<
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
            crate::launch::cube_count_1d(blocks)?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(input_bindings.slots[0].0.clone(), input_bindings.slots[0].1),
            BufferArg::from_raw_parts(input_bindings.slots[1].0.clone(), input_bindings.slots[1].1),
            BufferArg::from_raw_parts(input_bindings.slots[2].0.clone(), input_bindings.slots[2].1),
            BufferArg::from_raw_parts(input_bindings.slots[3].0.clone(), input_bindings.slots[3].1),
            BufferArg::from_raw_parts(input_bindings.slots[4].0.clone(), input_bindings.slots[4].1),
            BufferArg::from_raw_parts(input_bindings.slots[5].0.clone(), input_bindings.slots[5].1),
            BufferArg::from_raw_parts(input_bindings.slots[6].0.clone(), input_bindings.slots[6].1),
            BufferArg::from_raw_parts(input_bindings.slots[7].0.clone(), input_bindings.slots[7].1),
            BufferArg::from_raw_parts(input_bindings.slots[8].0.clone(), input_bindings.slots[8].1),
            BufferArg::from_raw_parts(input_bindings.slots[9].0.clone(), input_bindings.slots[9].1),
            BufferArg::from_raw_parts(
                input_bindings.slots[10].0.clone(),
                input_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                input_bindings.slots[11].0.clone(),
                input_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(flags.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(input_offsets, input_bindings.offsets.len()),
            BufferArg::from_raw_parts(output_offsets.clone(), output_bindings.offsets.len()),
            BufferArg::from_raw_parts(sum_offsets, sum_bindings.offsets.len()),
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
            BufferArg::from_raw_parts(local_flags.handle.clone(), len),
            BufferArg::from_raw_parts(sum_bindings.slots[0].0.clone(), sum_bindings.slots[0].1),
            BufferArg::from_raw_parts(sum_bindings.slots[1].0.clone(), sum_bindings.slots[1].1),
            BufferArg::from_raw_parts(sum_bindings.slots[2].0.clone(), sum_bindings.slots[2].1),
            BufferArg::from_raw_parts(sum_bindings.slots[3].0.clone(), sum_bindings.slots[3].1),
            BufferArg::from_raw_parts(sum_bindings.slots[4].0.clone(), sum_bindings.slots[4].1),
            BufferArg::from_raw_parts(sum_bindings.slots[5].0.clone(), sum_bindings.slots[5].1),
            BufferArg::from_raw_parts(sum_bindings.slots[6].0.clone(), sum_bindings.slots[6].1),
            BufferArg::from_raw_parts(sum_bindings.slots[7].0.clone(), sum_bindings.slots[7].1),
            BufferArg::from_raw_parts(sum_bindings.slots[8].0.clone(), sum_bindings.slots[8].1),
            BufferArg::from_raw_parts(sum_bindings.slots[9].0.clone(), sum_bindings.slots[9].1),
            BufferArg::from_raw_parts(sum_bindings.slots[10].0.clone(), sum_bindings.slots[10].1),
            BufferArg::from_raw_parts(sum_bindings.slots[11].0.clone(), sum_bindings.slots[11].1),
            BufferArg::from_raw_parts(block_flags.handle.clone(), blocks),
        );
    }

    if blocks > 1 {
        let prefixes = exec.alloc_canonical::<Item>(blocks);
        segmented_inclusive_fixed::<R, Item, Op>(exec, &block_sums, &block_flags, &prefixes)?;
        let prefix_read = prefixes.read();
        let prefix_bindings = stage_read(exec, &prefix_read)?;
        let prefix_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&prefix_bindings.offsets));
        unsafe {
            segmented_prefix_padded12::launch_unchecked::<
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
                crate::launch::cube_count_1d(blocks)?,
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
                BufferArg::from_raw_parts(local_flags.handle.clone(), len),
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
    }
    Ok(())
}

fn launch_exclusive<R, Item, Op>(
    exec: &Executor<R>,
    inclusive: &FixedStorage<R, Item>,
    flags: &DeviceVec<R, u32>,
    init: &FixedStorage<R, Item>,
    output: &FixedStorage<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::api::iter::MItem<R>,
    Op: ReductionOp<Item>,
{
    let len = inclusive.len()?;
    let inclusive_read = inclusive.read();
    let init_read = init.read();
    let output_write = output.write();
    let inclusive_bindings = stage_read(exec, &inclusive_read)?;
    let init_bindings = stage_read(exec, &init_read)?;
    let output_bindings = stage_write(exec, &output_write)?;
    let inclusive_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&inclusive_bindings.offsets));
    let init_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&init_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    unsafe {
        segmented_exclusive_padded12::launch_unchecked::<
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
            crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[0].0.clone(),
                inclusive_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[1].0.clone(),
                inclusive_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[2].0.clone(),
                inclusive_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[3].0.clone(),
                inclusive_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[4].0.clone(),
                inclusive_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[5].0.clone(),
                inclusive_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[6].0.clone(),
                inclusive_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[7].0.clone(),
                inclusive_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[8].0.clone(),
                inclusive_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[9].0.clone(),
                inclusive_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[10].0.clone(),
                inclusive_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[11].0.clone(),
                inclusive_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(flags.handle.clone(), len),
            BufferArg::from_raw_parts(init_bindings.slots[0].0.clone(), init_bindings.slots[0].1),
            BufferArg::from_raw_parts(init_bindings.slots[1].0.clone(), init_bindings.slots[1].1),
            BufferArg::from_raw_parts(init_bindings.slots[2].0.clone(), init_bindings.slots[2].1),
            BufferArg::from_raw_parts(init_bindings.slots[3].0.clone(), init_bindings.slots[3].1),
            BufferArg::from_raw_parts(init_bindings.slots[4].0.clone(), init_bindings.slots[4].1),
            BufferArg::from_raw_parts(init_bindings.slots[5].0.clone(), init_bindings.slots[5].1),
            BufferArg::from_raw_parts(init_bindings.slots[6].0.clone(), init_bindings.slots[6].1),
            BufferArg::from_raw_parts(init_bindings.slots[7].0.clone(), init_bindings.slots[7].1),
            BufferArg::from_raw_parts(init_bindings.slots[8].0.clone(), init_bindings.slots[8].1),
            BufferArg::from_raw_parts(init_bindings.slots[9].0.clone(), init_bindings.slots[9].1),
            BufferArg::from_raw_parts(init_bindings.slots[10].0.clone(), init_bindings.slots[10].1),
            BufferArg::from_raw_parts(init_bindings.slots[11].0.clone(), init_bindings.slots[11].1),
            BufferArg::from_raw_parts(checked_len_handle(exec, len)?, 1),
            BufferArg::from_raw_parts(inclusive_offsets, inclusive_bindings.offsets.len()),
            BufferArg::from_raw_parts(init_offsets, init_bindings.offsets.len()),
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

fn launch_apply_init<R, Item, Op>(
    exec: &Executor<R>,
    inclusive: &FixedStorage<R, Item>,
    init: &FixedStorage<R, Item>,
    output: &FixedStorage<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::api::iter::MItem<R>,
    Op: ReductionOp<Item>,
{
    let len = inclusive.len()?;
    let inclusive_read = inclusive.read();
    let init_read = init.read();
    let output_write = output.write();
    let inclusive_bindings = stage_read(exec, &inclusive_read)?;
    let init_bindings = stage_read(exec, &init_read)?;
    let output_bindings = stage_write(exec, &output_write)?;
    let inclusive_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&inclusive_bindings.offsets));
    let init_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&init_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    unsafe {
        apply_init_padded12::launch_unchecked::<
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
            crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[0].0.clone(),
                inclusive_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[1].0.clone(),
                inclusive_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[2].0.clone(),
                inclusive_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[3].0.clone(),
                inclusive_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[4].0.clone(),
                inclusive_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[5].0.clone(),
                inclusive_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[6].0.clone(),
                inclusive_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[7].0.clone(),
                inclusive_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[8].0.clone(),
                inclusive_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[9].0.clone(),
                inclusive_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[10].0.clone(),
                inclusive_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                inclusive_bindings.slots[11].0.clone(),
                inclusive_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(init_bindings.slots[0].0.clone(), init_bindings.slots[0].1),
            BufferArg::from_raw_parts(init_bindings.slots[1].0.clone(), init_bindings.slots[1].1),
            BufferArg::from_raw_parts(init_bindings.slots[2].0.clone(), init_bindings.slots[2].1),
            BufferArg::from_raw_parts(init_bindings.slots[3].0.clone(), init_bindings.slots[3].1),
            BufferArg::from_raw_parts(init_bindings.slots[4].0.clone(), init_bindings.slots[4].1),
            BufferArg::from_raw_parts(init_bindings.slots[5].0.clone(), init_bindings.slots[5].1),
            BufferArg::from_raw_parts(init_bindings.slots[6].0.clone(), init_bindings.slots[6].1),
            BufferArg::from_raw_parts(init_bindings.slots[7].0.clone(), init_bindings.slots[7].1),
            BufferArg::from_raw_parts(init_bindings.slots[8].0.clone(), init_bindings.slots[8].1),
            BufferArg::from_raw_parts(init_bindings.slots[9].0.clone(), init_bindings.slots[9].1),
            BufferArg::from_raw_parts(init_bindings.slots[10].0.clone(), init_bindings.slots[10].1),
            BufferArg::from_raw_parts(init_bindings.slots[11].0.clone(), init_bindings.slots[11].1),
            BufferArg::from_raw_parts(checked_len_handle(exec, len)?, 1),
            BufferArg::from_raw_parts(inclusive_offsets, inclusive_bindings.offsets.len()),
            BufferArg::from_raw_parts(init_offsets, init_bindings.offsets.len()),
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

pub(crate) fn segmented_fixed<R, Item, Op>(
    exec: &Executor<R>,
    input: &FixedStorage<R, Item>,
    heads: &DeviceVec<R, u32>,
    init: Option<Item>,
    _op: Op,
    mode: u8,
) -> Result<FixedStorage<R, Item>, Error>
where
    R: Runtime,
    Item: crate::api::iter::MItem<R>,
    Op: ReductionOp<Item>,
{
    let len = input.len()?;
    if heads.len() != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: heads.len(),
        });
    }
    let output = exec.alloc_canonical::<Item>(len);
    match mode {
        0 => segmented_inclusive_fixed::<R, Item, Op>(exec, input, heads, &output)?,
        1 => {
            if len == 0 {
                return Ok(output);
            }
            let inclusive = exec.alloc_canonical::<Item>(len);
            segmented_inclusive_fixed::<R, Item, Op>(exec, input, heads, &inclusive)?;
            let initial = crate::allocation::singleton(
                exec,
                init.expect("exclusive segmented scan requires init"),
            )?;
            launch_exclusive::<R, Item, Op>(exec, &inclusive, heads, &initial, &output)?;
        }
        2 => {
            if len == 0 {
                return Ok(output);
            }
            let inclusive = exec.alloc_canonical::<Item>(len);
            segmented_inclusive_fixed::<R, Item, Op>(exec, input, heads, &inclusive)?;
            let initial = crate::allocation::singleton(
                exec,
                init.expect("segmented reduction requires init"),
            )?;
            launch_apply_init::<R, Item, Op>(exec, &inclusive, &initial, &output)?;
        }
        _ => unreachable!("invalid segmented operation"),
    }
    Ok(output)
}
