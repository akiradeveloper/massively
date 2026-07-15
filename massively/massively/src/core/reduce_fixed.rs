//! Fixed-output-ABI host dispatch for reductions.

use super::*;

#[cfg(any())]
mod legacy_output_host {
    use super::*;
    use crate::DeviceVec;

    pub trait ReduceOutputHost<R: Runtime, Item: StorageLayout>:
        StorePadded12 + Send + Sync + 'static
    {
        type Partials;

        fn allocate(exec: &Executor<R>, blocks: usize) -> (Self::Partials, OutputBindings);
        fn finish<Op: ReductionOp<Item>>(
            exec: &Executor<R>,
            partials: &Self::Partials,
            blocks: usize,
            init: Item,
        ) -> Result<Item, Error>;
    }

    impl<R, Item> ReduceOutputHost<R, Item> for Last<Item>
    where
        R: Runtime,
        Item: ReduceStorage1<StorageLeaves = Last<Item>>,
        Last<Item>: StorePadded12<
                O0 = Item,
                O1 = u32,
                O2 = u32,
                O3 = u32,
                O4 = u32,
                O5 = u32,
                O6 = u32,
                O7 = u32,
                O8 = u32,
                O9 = u32,
                O10 = u32,
                O11 = u32,
            >,
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
            blocks: usize,
            init: Item,
        ) -> Result<Item, Error> {
            finish_storage1_reduce::<R, Item, Op>(exec, partials.handle.clone(), blocks, init)
        }
    }

    macro_rules! impl_reduce_output_host {
    (
        $leaves:ty, $storage_kernel:ident, $finalize_kernel:ident,
        $load:ident, $store:ident;
        [$a0:ty,$a1:ty,$a2:ty,$a3:ty,$a4:ty,$a5:ty,$a6:ty,$a7:ty,$a8:ty,$a9:ty,$a10:ty,$a11:ty];
        [$( $out:ident:$index:tt:$partial:ident:$current:ident:$next:ident:$init_handle:ident:$output_handle:ident:$value:ident:$first_field:ident $(.$rest_field:ident)* ),+]
    ) => {
        impl<R, Item, $( $out ),+> ReduceOutputHost<R, Item> for $leaves
        where
            R: Runtime,
            Item: StorageLayout<StorageLeaves = $leaves> + Send + Sync + 'static,
            Item::DeviceLayout: Decompose<Item, Leaves = $leaves>
                + Recompose<Item, Leaves = $leaves>,
            $( $out: MStorageElement, )+
            $leaves: $load<$( $out ),+>
                + $store<$( $out ),+>
                + StorePadded12<
                    O0 = $a0, O1 = $a1, O2 = $a2, O3 = $a3,
                    O4 = $a4, O5 = $a5, O6 = $a6, O7 = $a7,
                    O8 = $a8, O9 = $a9, O10 = $a10, O11 = $a11,
                >
                + Send + Sync + 'static,
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
                blocks: usize,
                init: Item,
            ) -> Result<Item, Error> {
                let client = exec.client();
                $( let mut $current = partials.$index.handle.clone(); )+
                let zero_values = [0u32; 12];
                let zero_handle = client.create_from_slice(u32::as_bytes(&zero_values));
                let mut current_len = blocks;
                while current_len > 1 {
                    let next_len = pass_block_count(current_len);
                    let current_len_handle = client.create_from_slice(u32::as_bytes(&[
                        checked_u32(current_len)?
                    ]));
                    $( let $next = client.empty(next_len * size_of::<$out>()); )+
                    unsafe {
                        $storage_kernel::launch_unchecked::<
                            Item, $( $out, )+ $leaves, Item::DeviceLayout, Op, R,
                        >(
                            client,
                            cube_count_1d(next_len)?,
                            CubeDim::new_1d(BLOCK_SIZE),
                            $( BufferArg::from_raw_parts($current.clone(), current_len), )+
                            BufferArg::from_raw_parts(current_len_handle, 1),
                            BufferArg::from_raw_parts(zero_handle.clone(), 12),
                            $( BufferArg::from_raw_parts($next.clone(), next_len), )+
                        );
                    }
                    $( $current = $next; )+
                    current_len = next_len;
                }

                let init_leaves = init.into_storage_leaves();
                $(
                    let $init_handle = client.create_from_slice($out::as_bytes(&[
                        leaf_value!(init_leaves; $first_field $(.$rest_field)*)
                    ]));
                    let $output_handle = client.empty(size_of::<$out>());
                )+
                unsafe {
                    $finalize_kernel::launch_unchecked::<
                        Item, $( $out, )+ $leaves, Item::DeviceLayout, Op, R,
                    >(
                        client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(1),
                        $( BufferArg::from_raw_parts($current, 1), )+
                        $( BufferArg::from_raw_parts($init_handle, 1), )+
                        BufferArg::from_raw_parts(zero_handle, 12),
                        $( BufferArg::from_raw_parts($output_handle.clone(), 1), )+
                    );
                }
                $(
                    let bytes = client.read_one($output_handle).map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                    let $value = $out::from_bytes(&bytes)[0];
                )+
                Ok(Item::from_storage_leaves(build_leaf_list!($( $value ),+)))
            }
        }
    };
}

    impl_reduce_output_host!(More<O0,Last<O1>>,reduce_storage_s2,reduce_finalize_s2,LoadLeaves2,StoreLeaves2; [O0,O1,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,Last<O2>>>,reduce_storage_s3,reduce_finalize_s3,LoadLeaves3,StoreLeaves3; [O0,O1,O2,u32,u32,u32,u32,u32,u32,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,Last<O3>>>>,reduce_storage_s4,reduce_finalize_s4,LoadLeaves4,StoreLeaves4; [O0,O1,O2,O3,u32,u32,u32,u32,u32,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>,reduce_storage_s5,reduce_finalize_s5,LoadLeaves5,StoreLeaves5; [O0,O1,O2,O3,O4,u32,u32,u32,u32,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>,reduce_storage_s6,reduce_finalize_s6,LoadLeaves6,StoreLeaves6; [O0,O1,O2,O3,O4,O5,u32,u32,u32,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.head,O5:5:p5:c5:n5:i5:o5:v5:tail.tail.tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>,reduce_storage_s7,reduce_finalize_s7,LoadLeaves7,StoreLeaves7; [O0,O1,O2,O3,O4,O5,O6,u32,u32,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.head,O5:5:p5:c5:n5:i5:o5:v5:tail.tail.tail.tail.tail.head,O6:6:p6:c6:n6:i6:o6:v6:tail.tail.tail.tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,Last<O7>>>>>>>>,reduce_storage_s8,reduce_finalize_s8,LoadLeaves8,StoreLeaves8; [O0,O1,O2,O3,O4,O5,O6,O7,u32,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.head,O5:5:p5:c5:n5:i5:o5:v5:tail.tail.tail.tail.tail.head,O6:6:p6:c6:n6:i6:o6:v6:tail.tail.tail.tail.tail.tail.head,O7:7:p7:c7:n7:i7:o7:v7:tail.tail.tail.tail.tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,Last<O8>>>>>>>>>,reduce_storage_s9,reduce_finalize_s9,LoadLeaves9,StoreLeaves9; [O0,O1,O2,O3,O4,O5,O6,O7,O8,u32,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.head,O5:5:p5:c5:n5:i5:o5:v5:tail.tail.tail.tail.tail.head,O6:6:p6:c6:n6:i6:o6:v6:tail.tail.tail.tail.tail.tail.head,O7:7:p7:c7:n7:i7:o7:v7:tail.tail.tail.tail.tail.tail.tail.head,O8:8:p8:c8:n8:i8:o8:v8:tail.tail.tail.tail.tail.tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,Last<O9>>>>>>>>>>,reduce_storage_s10,reduce_finalize_s10,LoadLeaves10,StoreLeaves10; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,u32,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.head,O5:5:p5:c5:n5:i5:o5:v5:tail.tail.tail.tail.tail.head,O6:6:p6:c6:n6:i6:o6:v6:tail.tail.tail.tail.tail.tail.head,O7:7:p7:c7:n7:i7:o7:v7:tail.tail.tail.tail.tail.tail.tail.head,O8:8:p8:c8:n8:i8:o8:v8:tail.tail.tail.tail.tail.tail.tail.tail.head,O9:9:p9:c9:n9:i9:o9:v9:tail.tail.tail.tail.tail.tail.tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,Last<O10>>>>>>>>>>>,reduce_storage_s11,reduce_finalize_s11,LoadLeaves11,StoreLeaves11; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,u32]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.head,O5:5:p5:c5:n5:i5:o5:v5:tail.tail.tail.tail.tail.head,O6:6:p6:c6:n6:i6:o6:v6:tail.tail.tail.tail.tail.tail.head,O7:7:p7:c7:n7:i7:o7:v7:tail.tail.tail.tail.tail.tail.tail.head,O8:8:p8:c8:n8:i8:o8:v8:tail.tail.tail.tail.tail.tail.tail.tail.head,O9:9:p9:c9:n9:i9:o9:v9:tail.tail.tail.tail.tail.tail.tail.tail.tail.head,O10:10:p10:c10:n10:i10:o10:v10:tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.value]);
    impl_reduce_output_host!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,More<O10,Last<O11>>>>>>>>>>>>,reduce_storage_s12,reduce_finalize_s12,LoadLeaves12,StoreLeaves12; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11]; [O0:0:p0:c0:n0:i0:o0:v0:head,O1:1:p1:c1:n1:i1:o1:v1:tail.head,O2:2:p2:c2:n2:i2:o2:v2:tail.tail.head,O3:3:p3:c3:n3:i3:o3:v3:tail.tail.tail.head,O4:4:p4:c4:n4:i4:o4:v4:tail.tail.tail.tail.head,O5:5:p5:c5:n5:i5:o5:v5:tail.tail.tail.tail.tail.head,O6:6:p6:c6:n6:i6:o6:v6:tail.tail.tail.tail.tail.tail.head,O7:7:p7:c7:n7:i7:o7:v7:tail.tail.tail.tail.tail.tail.tail.head,O8:8:p8:c8:n8:i8:o8:v8:tail.tail.tail.tail.tail.tail.tail.tail.head,O9:9:p9:c9:n9:i9:o9:v9:tail.tail.tail.tail.tail.tail.tail.tail.tail.head,O10:10:p10:c10:n10:i10:o10:v10:tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.head,O11:11:p11:c11:n11:i11:o11:v11:tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.value]);
}

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
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[
                    checked_u32(len)?,
                ]));
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
                        BufferArg::from_raw_parts(len_handle, 1),
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
