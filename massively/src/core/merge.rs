//! Stable two-range merge control and arity-independent payload application.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    A13, Column, DeviceVec, Error, Executor, MStorageElement, ReadExpression, RowAlloc, RowStorage,
    StorageLayout,
    allocation::{CopyStorage, NormalizeInput},
    eval::Eval13,
    indexed::GatherInput,
    ordering::BinaryPredicateOp,
    read::{Env0, Env13, FixedRead, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
};

const BLOCK_SIZE: u32 = 256;

macro_rules! define_merge_control_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $left_leaf:ident:$left_slot:ident:$right_leaf:ident:$right_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $left_leaf: CubePrimitive, )+
            $( $right_leaf: CubePrimitive, )+
            Left: $eval<Item, $( $left_leaf ),+>,
            Right: $eval<Item, $( $right_leaf ),+>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $left_slot: &[$left_leaf], )+
            left_offsets: &[u32],
            $( $right_slot: &[$right_leaf], )+
            right_offsets: &[u32],
            left_length: &[u32],
            right_length: &[u32],
            parameters: &[u32],
            permutation: &mut [u32],
        ) {
            let out = RuntimeCell::<usize>::new(ABSOLUTE_POS as usize);
            let stride = (CUBE_COUNT as usize) * (CUBE_DIM as usize);
            let left_len = left_length[0] as usize;
            let right_len = right_length[0] as usize;
            let right_base = parameters[0];
            let total = left_len + right_len;
            while out.read() < total {
                let rank = out.read();
                let low_init = if rank > right_len { rank - right_len } else { 0usize };
                let high_init = if rank < left_len { rank } else { left_len };
                let low = RuntimeCell::<usize>::new(low_init);
                let high = RuntimeCell::<usize>::new(high_init);
                while low.read() < high.read() {
                    let left_rank = (low.read() + high.read()) / 2usize;
                    let right_rank = rank - left_rank;
                    if left_rank < left_len
                        && right_rank > 0usize
                        && !crate::ordering::binary_predicate::<Item, Less>(
                            Right::$method($( $right_slot, )+ right_offsets, right_rank - 1usize),
                            Left::$method($( $left_slot, )+ left_offsets, left_rank),
                        )
                    {
                        low.store(left_rank + 1usize);
                    } else {
                        high.store(left_rank);
                    }
                }
                let left_rank = low.read();
                let right_rank = rank - left_rank;
                if left_rank < left_len {
                    if right_rank >= right_len
                        || !crate::ordering::binary_predicate::<Item, Less>(
                            Right::$method($( $right_slot, )+ right_offsets, right_rank),
                            Left::$method($( $left_slot, )+ left_offsets, left_rank),
                        )
                    {
                        permutation[rank] = left_rank as u32;
                    } else {
                        permutation[rank] = right_base + right_rank as u32;
                    }
                } else {
                    permutation[rank] = right_base + right_rank as u32;
                }
                out.store(rank + stride);
            }
        }
    };
}

define_merge_control_kernel!(merge_control_a13,Eval13,eval13; [LL0:left0:RL0:right0,LL1:left1:RL1:right1,LL2:left2:RL2:right2,LL3:left3:RL3:right3,LL4:left4:RL4:right4,LL5:left5:RL5:right5,LL6:left6:RL6:right6,LL7:left7:RL7:right7,LL8:left8:RL8:right8,LL9:left9:RL9:right9,LL10:left10:RL10:right10,LL11:left11:RL11:right11,LL12:left12:RL12:right12]);

pub(crate) struct MergeDispatch<Storage>(PhantomData<fn() -> Storage>);

pub(crate) trait MergeControlDispatch<R, Left, Right, Item, LeftSlots, RightSlots, Less>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error>;
}

macro_rules! impl_merge_control_dispatch {
    ($storage:ty,$arity:ty,$eval:ident,$kernel:ident; [$( $left_leaf:ident:$left_index:literal:$right_leaf:ident:$right_index:literal ),+]) => {
        impl<R, Left, Right, Item, Less, $( $left_leaf, )+ $( $right_leaf ),+>
            MergeControlDispatch<
                R,
                Left,
                Right,
                Item,
                Env13<$( $left_leaf ),+>,
                Env13<$( $right_leaf ),+>,
                Less,
            >
            for MergeDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Less: BinaryPredicateOp<Item>,
            $( $left_leaf: MStorageElement, )+
            $( $right_leaf: MStorageElement, )+
            Left: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = Env13<$( $left_leaf ),+>>
                + StageRead<R, Env0>,
            Right: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = Env13<$( $right_leaf ),+>>
                + StageRead<R, Env0>,
            Left::DeviceExpr: $eval<Item, $( $left_leaf ),+>,
            Right::DeviceExpr: $eval<Item, $( $right_leaf ),+>,
        {
            fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error> {
                let left_capacity = left.logical_len()?;
                let right_capacity = right.logical_len()?;
                let total_capacity = left_capacity.checked_add(right_capacity).ok_or(Error::LengthTooLarge { len: usize::MAX })?;
                let left_extent = left.logical_extent()?;
                let right_extent = right.logical_extent()?;
                let total_extent = crate::extent::LogicalExtent::add(
                    exec,
                    &left_extent,
                    &right_extent,
                    total_capacity,
                )?;
                let mut permutation = exec.alloc_row::<u32>(total_capacity);
                permutation.set_logical_extent(total_extent);
                if total_capacity == 0 {
                    return Ok(MergeControl {
                        permutation,
                        left_capacity,
                        right_capacity,
                        left_extent,
                        right_extent,
                    });
                }
                let mut left_bindings = StagedBindings::new();
                let mut right_bindings = StagedBindings::new();
                left.stage_at(exec.client(), exec.id(), &mut left_bindings)?;
                right.stage_at(exec.client(), exec.id(), &mut right_bindings)?;
                let left_offsets = exec.client().create_from_slice(u32::as_bytes(&left_bindings.offsets));
                let right_offsets = exec.client().create_from_slice(u32::as_bytes(&right_bindings.offsets));
                let left_length = left_extent.materialize(exec)?;
                let right_length = right_extent.materialize(exec)?;
                let right_base = u32::try_from(left_capacity)
                    .map_err(|_| Error::LengthTooLarge { len: left_capacity })?;
                let parameters = exec.client().create_from_slice(u32::as_bytes(&[right_base]));
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $left_leaf, )+ $( $right_leaf, )+ Left::DeviceExpr, Right::DeviceExpr, Less, R>(
                        exec.client(),
                        crate::launch::cube_count_1d(total_capacity.div_ceil(BLOCK_SIZE as usize).min(256))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(left_bindings.slots[$left_index].0.clone(), left_bindings.slots[$left_index].1), )+
                        BufferArg::from_raw_parts(left_offsets, left_bindings.offsets.len()),
                        $( BufferArg::from_raw_parts(right_bindings.slots[$right_index].0.clone(), right_bindings.slots[$right_index].1), )+
                        BufferArg::from_raw_parts(right_offsets, right_bindings.offsets.len()),
                        BufferArg::from_raw_parts(left_length.handle.clone(), 1),
                        BufferArg::from_raw_parts(right_length.handle.clone(), 1),
                        BufferArg::from_raw_parts(parameters, 1),
                        BufferArg::from_raw_parts(
                            permutation.handle.clone(),
                            permutation.capacity(),
                        ),
                    );
                }
                Ok(MergeControl {
                    permutation,
                    left_capacity,
                    right_capacity,
                    left_extent,
                    right_extent,
                })
            }
        }
    };
}

impl_merge_control_dispatch!(crate::S12,A13,Eval13,merge_control_a13; [LL0:0:RL0:0,LL1:1:RL1:1,LL2:2:RL2:2,LL3:3:RL3:3,LL4:4:RL4:4,LL5:5:RL5:5,LL6:6:RL6:6,LL7:7:RL7:7,LL8:8:RL8:8,LL9:9:RL9:9,LL10:10:RL10:10,LL11:11:RL11:11,LL12:12:RL12:12]);

/// Stable merge permutation over a conceptual `left || right` payload.
#[doc(hidden)]
pub struct MergeControl<R: Runtime> {
    pub(crate) permutation: DeviceVec<R, u32>,
    pub(crate) left_capacity: usize,
    pub(crate) right_capacity: usize,
    pub(crate) left_extent: crate::extent::LogicalExtent,
    pub(crate) right_extent: crate::extent::LogicalExtent,
}

/// Internal public-API capability for merge control construction.
#[doc(hidden)]
pub(crate) trait MergeControlInput<R: Runtime, Right, Less>: NormalizeInput<R> {
    fn merge_control(self, exec: &Executor<R>, right: Right) -> Result<MergeControl<R>, Error>;
}

impl<R, Left, Right, Less> MergeControlInput<R, Right, Less> for Left
where
    R: Runtime,
    Left: NormalizeInput<R>,
    Left::Item: StorageLayout + RowAlloc<R>,
    Right: NormalizeInput<R> + ReadExpression<Item = Left::Item>,
    Left::SemanticRead: LowerReadExpression,
    Right::SemanticRead: LowerReadExpression,
    MergeDispatch<crate::S12>: MergeControlDispatch<
            R,
            FixedRead<Left::SemanticRead>,
            FixedRead<Right::SemanticRead>,
            Left::Item,
            <FixedRead<Left::SemanticRead> as LowerReadExpression>::Slots,
            <FixedRead<Right::SemanticRead> as LowerReadExpression>::Slots,
            Less,
        >,
{
    fn merge_control(self, exec: &Executor<R>, right: Right) -> Result<MergeControl<R>, Error> {
        let left = self.normalize(exec)?;
        let right = right.normalize(exec)?;
        let left_read = FixedRead::new(Left::semantic_read(&left));
        let right_read = FixedRead::new(Right::semantic_read(&right));
        <MergeDispatch<crate::S12> as MergeControlDispatch<
            R,
            FixedRead<Left::SemanticRead>,
            FixedRead<Right::SemanticRead>,
            Left::Item,
            <FixedRead<Left::SemanticRead> as LowerReadExpression>::Slots,
            <FixedRead<Right::SemanticRead> as LowerReadExpression>::Slots,
            Less,
        >>::run(exec, &left_read, &right_read)
    }
}

pub(crate) fn merge_control_with<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<MergeControl<R>, Error>
where
    R: Runtime,
    Left: MergeControlInput<R, Right, Less>,
{
    left.merge_control(exec, right)
}

pub(crate) fn merge_control_fixed<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<MergeControl<R>, Error>
where
    R: Runtime,
    Left: ReadExpression<ReadArity = A13> + LowerReadExpression + StageRead<R, Env0>,
    Right: ReadExpression<Item = Left::Item, ReadArity = A13>
        + LowerReadExpression
        + StageRead<R, Env0>,
    MergeDispatch<crate::S12>:
        MergeControlDispatch<R, Left, Right, Left::Item, Left::Slots, Right::Slots, Less>,
{
    <MergeDispatch<crate::S12> as MergeControlDispatch<
        R,
        Left,
        Right,
        Left::Item,
        Left::Slots,
        Right::Slots,
        Less,
    >>::run(exec, &left, &right)
}

/// Applies a merge permutation to two already-normalized payloads.
///
/// The payload layout is handled once through its semantic leaf list.  Key
/// and value arities therefore never occur in the same dispatch obligation.
pub(crate) fn apply_storage<R, Item, Output>(
    exec: &Executor<R>,
    left: &<Item as crate::allocation::ScratchStorage<R>>::Storage,
    right: &<Item as crate::allocation::ScratchStorage<R>>::Storage,
    control: &MergeControl<R>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::allocation::ScratchStorage<R>,
    Item::StorageLeaves: crate::core::facade::KernelValue,
    <Item as crate::allocation::ScratchStorage<R>>::Storage: RowStorage<R>,
    Output: crate::core::facade::KernelOutput<R> + crate::output::OutputExpression<Item = Item>,
{
    let left_capacity = left.len()?;
    let right_capacity = right.len()?;
    if left_capacity != control.left_capacity || right_capacity != control.right_capacity {
        return Err(Error::LengthMismatch {
            left: left_capacity + right_capacity,
            right: control.left_capacity + control.right_capacity,
        });
    }
    left.logical_extent().zipped(&control.left_extent)?;
    right.logical_extent().zipped(&control.right_extent)?;

    let total = left_capacity
        .checked_add(right_capacity)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    let combined = Item::alloc_scratch(exec, total);

    let left_read = crate::read::FixedRead::new(left.read());
    crate::transform::materialize_fixed(exec, &left_read, &combined.slice_mut(..left_capacity))?;

    let right_read = crate::read::FixedRead::new(right.read());
    crate::transform::materialize_fixed(exec, &right_read, &combined.slice_mut(left_capacity..))?;

    crate::indexed::gather_direct(
        exec,
        crate::read::FixedRead::new(combined.read()),
        control.permutation.column(),
        output,
    )
}

/// Applies a conceptual concatenation control to one payload pair.
#[doc(hidden)]
pub(crate) trait ConcatApply<R: Runtime, Right, Output>: NormalizeInput<R> {
    fn concat_apply(
        self,
        exec: &Executor<R>,
        right: Right,
        control: &MergeControl<R>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Left, Right, Output> ConcatApply<R, Right, Output> for Left
where
    R: Runtime,
    Left: NormalizeInput<R>,
    Left::Item: RowAlloc<R, RowStorage = Left::Storage>,
    Left::Storage: CopyStorage<R>,
    Right: NormalizeInput<R, Storage = Left::Storage> + ReadExpression<Item = Left::Item>,
    <Left::Storage as RowStorage<R>>::Read: GatherInput<R, Column<crate::MIndex>, Output>,
{
    fn concat_apply(
        self,
        exec: &Executor<R>,
        right: Right,
        control: &MergeControl<R>,
        output: Output,
    ) -> Result<(), Error> {
        let left = self.normalize(exec)?;
        let right = right.normalize(exec)?;
        let left_capacity = left.len()?;
        let right_capacity = right.len()?;
        if left_capacity != control.left_capacity || right_capacity != control.right_capacity {
            return Err(Error::LengthMismatch {
                left: left_capacity + right_capacity,
                right: control.left_capacity + control.right_capacity,
            });
        }
        left.logical_extent().zipped(&control.left_extent)?;
        right.logical_extent().zipped(&control.right_extent)?;
        let combined = exec.alloc_row::<Left::Item>(left_capacity + right_capacity);
        left.copy_storage(exec, combined.slice_mut(..left_capacity))?;
        right.copy_storage(exec, combined.slice_mut(left_capacity..))?;
        crate::indexed::gather_direct(exec, combined.read(), control.permutation.column(), output)
    }
}

/// Stably merges two sorted semantic ranges.
pub(crate) fn merge<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: Clone + MergeControlInput<R, Right, Less> + ConcatApply<R, Right, Output>,
    Right: Clone,
{
    let control = left.clone().merge_control(exec, right.clone())?;
    left.concat_apply(exec, right, &control, output)
}

/// Stably merges key/value ranges using one key control and an independent
/// value apply phase.
pub(crate) fn merge_by_key<
    R,
    LeftKeys,
    LeftValues,
    RightKeys,
    RightValues,
    Less,
    KeyOutput,
    ValueOutput,
>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    _less: Less,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    LeftKeys: Clone + MergeControlInput<R, RightKeys, Less> + ConcatApply<R, RightKeys, KeyOutput>,
    RightKeys: Clone,
    LeftValues: ConcatApply<R, RightValues, ValueOutput>,
{
    let control = left_keys.clone().merge_control(exec, right_keys.clone())?;
    left_keys.concat_apply(exec, right_keys, &control, key_output)?;
    left_values.concat_apply(exec, right_values, &control, value_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Zip;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MBool {
            crate::op::mbool(lhs < rhs)
        }
    }

    #[test]
    fn merge_is_stable_and_merge_by_key_reuses_one_control() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 2, 5]);
        let right = exec.to_device(&[2_u32, 3, 4]);
        let output = exec.to_device(&[0_u32; 7]);
        merge(
            &exec,
            left.column(),
            right.column(),
            LessU32,
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 2, 2, 3, 4, 5]);

        let left_values = exec.to_device(&[10_u32, 20, 21, 50]);
        let right_values = exec.to_device(&[200_u32, 300, 400]);
        let out_keys = exec.to_device(&[0_u32; 7]);
        let out_values = exec.to_device(&[0_u32; 7]);
        merge_by_key(
            &exec,
            left.column(),
            left_values.column(),
            right.column(),
            right_values.column(),
            LessU32,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 2, 2, 3, 4, 5]);
        assert_eq!(
            exec.to_host(&out_values).unwrap(),
            vec![10, 20, 21, 200, 300, 400, 50]
        );

        // Keep a binary output in the monomorphization surface as well.
        let pair_out = Zip::new(
            exec.to_device(&[0_u32; 7]).slice_mut(..),
            exec.to_device(&[0_u32; 7]).slice_mut(..),
        );
        let pair_left = Zip::new(left.column(), left_values.column());
        let pair_right = Zip::new(right.column(), right_values.column());
        struct LessPair;
        #[cubecl::cube]
        impl BinaryPredicateOp<(u32, u32)> for LessPair {
            fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> crate::MBool {
                crate::op::mbool(lhs.0 < rhs.0)
            }
        }
        merge(&exec, pair_left, pair_right, LessPair, pair_out).unwrap();
    }
}
