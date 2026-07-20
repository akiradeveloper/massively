//! Fixed-output-ABI device dispatch for reductions.

use super::*;

macro_rules! impl_padded_reduce_pass_dispatch {
    ($arity:ty,$eval:ident,$kernel:ident,$read_env:ty,$write_env:ty; [$( $leaf:ident:$index:literal ),+]) => {
        impl<R, Input, Output, Item, Op, $( $leaf, )+ O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11>
            ReducePassDispatch<R, Input, Output, Item, Op, $read_env, $write_env>
            for Dispatch<$arity, S12>
        where
            R: Runtime,
            Item: StorageLayout + Send + Sync + 'static,
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
            Op: ReductionOp<Item>,
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: crate::output::OutputExpression<Item = Item>
                + crate::output::LowerOutputExpression
                + crate::output::StageOutput<R, Env0>,
            Output::Slots: crate::output::PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Item::StorageLeaves: StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                > + SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout: Decompose<Item, Leaves = Item::StorageLeaves>
                + Recompose<Item, Leaves = Item::StorageLeaves>,
        {
            fn execute_pass(
                exec: &Executor<R>,
                input: &Input,
                output: &Output,
            ) -> Result<(), Error> {
                let len = input.logical_len()?;
                debug_assert!(len != 0);
                let blocks = pass_block_count(len);
                let mut bindings = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut bindings)?;
                bindings.pad_to_thirteen(exec.client());
                let mut output_bindings = OutputBindings::new();
                output.stage_output(exec.id(), &mut output_bindings)?;
                output_bindings.pad_to_twelve(exec.client());
                let offsets = exec.client().create_from_slice(u32::as_bytes(&bindings.offsets));
                let zero_values = [0u32; 12];
                let zero_offsets = exec.client().create_from_slice(u32::as_bytes(&zero_values));
                let len_handle = input.logical_extent()?.materialize(exec)?;
                unsafe {
                    $kernel::launch_unchecked::<
                        Item, $( $leaf, )+
                        O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
                        Item::StorageLeaves, Item::DeviceLayout, Input::DeviceExpr, Op, R,
                    >(
                        exec.client(),
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(bindings.slots[$index].0.clone(), bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets, bindings.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(zero_offsets, 12),
                        BufferArg::from_raw_parts(output_bindings.slots[0].0.clone(), output_bindings.slots[0].1),
                        BufferArg::from_raw_parts(output_bindings.slots[1].0.clone(), output_bindings.slots[1].1),
                        BufferArg::from_raw_parts(output_bindings.slots[2].0.clone(), output_bindings.slots[2].1),
                        BufferArg::from_raw_parts(output_bindings.slots[3].0.clone(), output_bindings.slots[3].1),
                        BufferArg::from_raw_parts(output_bindings.slots[4].0.clone(), output_bindings.slots[4].1),
                        BufferArg::from_raw_parts(output_bindings.slots[5].0.clone(), output_bindings.slots[5].1),
                        BufferArg::from_raw_parts(output_bindings.slots[6].0.clone(), output_bindings.slots[6].1),
                        BufferArg::from_raw_parts(output_bindings.slots[7].0.clone(), output_bindings.slots[7].1),
                        BufferArg::from_raw_parts(output_bindings.slots[8].0.clone(), output_bindings.slots[8].1),
                        BufferArg::from_raw_parts(output_bindings.slots[9].0.clone(), output_bindings.slots[9].1),
                        BufferArg::from_raw_parts(output_bindings.slots[10].0.clone(), output_bindings.slots[10].1),
                        BufferArg::from_raw_parts(output_bindings.slots[11].0.clone(), output_bindings.slots[11].1),
                    );
                }
                Ok(())
            }
        }
    };
}

impl_padded_reduce_pass_dispatch!(
    A13,
    Eval13,
    padded_reduce_a13,
    Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>,
    Env12<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11>;
    [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12]
);
