//! Stable two-range merge control and arity-independent payload application.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, CanonicalAlloc, CanonicalStorage, Column, DeviceVec, Error,
    Executor, MStorageElement, ReadExpression, StorageLayout,
    allocation::{CopyStorage, NormalizeInput},
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7},
    indexed::GatherInput,
    ordering::BinaryPredicateOp,
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
};

const BLOCK_SIZE: u32 = 256;

macro_rules! define_merge_control_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$left_slot:ident:$right_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Left: $eval<Item, $( $leaf ),+>,
            Right: $eval<Item, $( $leaf ),+>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $left_slot: &[$leaf], )+
            left_offsets: &[u32],
            $( $right_slot: &[$leaf], )+
            right_offsets: &[u32],
            lengths: &[u32],
            permutation: &mut [u32],
        ) {
            let out = RuntimeCell::<usize>::new(ABSOLUTE_POS as usize);
            let stride = (CUBE_COUNT as usize) * (CUBE_DIM as usize);
            let left_len = lengths[0] as usize;
            let right_len = lengths[1] as usize;
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
                        && !Less::apply(
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
                        || !Less::apply(
                            Right::$method($( $right_slot, )+ right_offsets, right_rank),
                            Left::$method($( $left_slot, )+ left_offsets, left_rank),
                        )
                    {
                        permutation[rank] = left_rank as u32;
                    } else {
                        permutation[rank] = (left_len + right_rank) as u32;
                    }
                } else {
                    permutation[rank] = (left_len + right_rank) as u32;
                }
                out.store(rank + stride);
            }
        }
    };
}

define_merge_control_kernel!(merge_control_s1,Eval1,eval1; [L0:left0:right0]);
define_merge_control_kernel!(merge_control_s2,Eval2,eval2; [L0:left0:right0,L1:left1:right1]);
define_merge_control_kernel!(merge_control_s3,Eval3,eval3; [L0:left0:right0,L1:left1:right1,L2:left2:right2]);
define_merge_control_kernel!(merge_control_s4,Eval4,eval4; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3]);
define_merge_control_kernel!(merge_control_s5,Eval5,eval5; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3,L4:left4:right4]);
define_merge_control_kernel!(merge_control_s6,Eval6,eval6; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3,L4:left4:right4,L5:left5:right5]);
define_merge_control_kernel!(merge_control_s7,Eval7,eval7; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3,L4:left4:right4,L5:left5:right5,L6:left6:right6]);

struct MergeDispatch<Storage>(PhantomData<fn() -> Storage>);

trait MergeControlDispatch<R, Left, Right, Item, LeftSlots, RightSlots, Less>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error>;
}

macro_rules! impl_merge_control_dispatch {
    ($storage:ty,$arity:ty,$eval:ident,$kernel:ident,$env:ty; [$( $leaf:ident:$index:literal ),+]) => {
        impl<R, Left, Right, Item, Less, $( $leaf ),+>
            MergeControlDispatch<R, Left, Right, Item, $env, $env, Less>
            for MergeDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Less: BinaryPredicateOp<Item>,
            $( $leaf: MStorageElement, )+
            Left: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Right: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Left::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Right::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error> {
                let left_len = left.logical_len()?;
                let right_len = right.logical_len()?;
                let total = left_len.checked_add(right_len).ok_or(Error::LengthTooLarge { len: usize::MAX })?;
                let permutation = exec.alloc_canonical::<u32>(total);
                if total == 0 {
                    return Ok(MergeControl { permutation, left_len, right_len });
                }
                let mut left_bindings = StagedBindings::new();
                let mut right_bindings = StagedBindings::new();
                left.stage_at(exec.client(), exec.id(), &mut left_bindings)?;
                right.stage_at(exec.client(), exec.id(), &mut right_bindings)?;
                let left_offsets = exec.client().create_from_slice(u32::as_bytes(&left_bindings.offsets));
                let right_offsets = exec.client().create_from_slice(u32::as_bytes(&right_bindings.offsets));
                let left_u32 = u32::try_from(left_len).map_err(|_| Error::LengthTooLarge { len: left_len })?;
                let right_u32 = u32::try_from(right_len).map_err(|_| Error::LengthTooLarge { len: right_len })?;
                let lengths = exec.client().create_from_slice(u32::as_bytes(&[left_u32, right_u32]));
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $leaf, )+ Left::DeviceExpr, Right::DeviceExpr, Less, R>(
                        exec.client(),
                        crate::launch::cube_count_1d(total.div_ceil(BLOCK_SIZE as usize).min(256))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(left_bindings.slots[$index].0.clone(), left_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(left_offsets, left_bindings.offsets.len()),
                        $( BufferArg::from_raw_parts(right_bindings.slots[$index].0.clone(), right_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(right_offsets, right_bindings.offsets.len()),
                        BufferArg::from_raw_parts(lengths, 2),
                        BufferArg::from_raw_parts(permutation.handle.clone(), permutation.len()),
                    );
                }
                Ok(MergeControl { permutation, left_len, right_len })
            }
        }
    };
}

impl_merge_control_dispatch!(crate::S1,A1,Eval1,merge_control_s1,Env1<L0>; [L0:0]);
impl_merge_control_dispatch!(crate::S2,A2,Eval2,merge_control_s2,Env2<L0,L1>; [L0:0,L1:1]);
impl_merge_control_dispatch!(crate::S3,A3,Eval3,merge_control_s3,Env3<L0,L1,L2>; [L0:0,L1:1,L2:2]);
impl_merge_control_dispatch!(crate::S4,A4,Eval4,merge_control_s4,Env4<L0,L1,L2,L3>; [L0:0,L1:1,L2:2,L3:3]);
impl_merge_control_dispatch!(crate::S5,A5,Eval5,merge_control_s5,Env5<L0,L1,L2,L3,L4>; [L0:0,L1:1,L2:2,L3:3,L4:4]);
impl_merge_control_dispatch!(crate::S6,A6,Eval6,merge_control_s6,Env6<L0,L1,L2,L3,L4,L5>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5]);
impl_merge_control_dispatch!(crate::S7,A7,Eval7,merge_control_s7,Env7<L0,L1,L2,L3,L4,L5,L6>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6]);

/// Stable merge permutation over a conceptual `left || right` payload.
#[doc(hidden)]
pub struct MergeControl<R: Runtime> {
    pub(crate) permutation: DeviceVec<R, u32>,
    pub(crate) left_len: usize,
    pub(crate) right_len: usize,
}

/// Internal public-API capability for merge control construction.
#[doc(hidden)]
pub trait MergeControlInput<R: Runtime, Right, Less>: NormalizeInput<R> {
    fn merge_control(self, exec: &Executor<R>, right: Right) -> Result<MergeControl<R>, Error>;
}

impl<R, Left, Right, Less> MergeControlInput<R, Right, Less> for Left
where
    R: Runtime,
    Left: NormalizeInput<R>,
    Left::Item: StorageLayout,
    Right: NormalizeInput<R> + ReadExpression<Item = Left::Item>,
    Left::SemanticRead: LowerReadExpression,
    Right::SemanticRead: LowerReadExpression,
    MergeDispatch<<Left::Item as StorageLayout>::StorageArity>: MergeControlDispatch<
            R,
            Left::SemanticRead,
            Right::SemanticRead,
            Left::Item,
            <Left::SemanticRead as LowerReadExpression>::Slots,
            <Right::SemanticRead as LowerReadExpression>::Slots,
            Less,
        >,
{
    fn merge_control(self, exec: &Executor<R>, right: Right) -> Result<MergeControl<R>, Error> {
        let left = self.normalize(exec)?;
        let right = right.normalize(exec)?;
        let left_read = Left::semantic_read(&left);
        let right_read = Right::semantic_read(&right);
        <MergeDispatch<<Left::Item as StorageLayout>::StorageArity> as MergeControlDispatch<
            R,
            Left::SemanticRead,
            Right::SemanticRead,
            Left::Item,
            <Left::SemanticRead as LowerReadExpression>::Slots,
            <Right::SemanticRead as LowerReadExpression>::Slots,
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

/// Applies a conceptual concatenation control to one payload pair.
#[doc(hidden)]
pub trait ConcatApply<R: Runtime, Right, Output>: NormalizeInput<R> {
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
    Left::Item: CanonicalAlloc<R, CanonicalStorage = Left::Storage>,
    Left::Storage: CopyStorage<R>,
    Right: NormalizeInput<R, Storage = Left::Storage> + ReadExpression<Item = Left::Item>,
    <Left::Storage as CanonicalStorage<R>>::Read: GatherInput<R, Column<u32>, Output>,
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
        let left_len = left.len()?;
        let right_len = right.len()?;
        if left_len != control.left_len || right_len != control.right_len {
            return Err(Error::LengthMismatch {
                left: left_len + right_len,
                right: control.left_len + control.right_len,
            });
        }
        let combined = exec.alloc_canonical::<Left::Item>(left_len + right_len);
        left.copy_storage(exec, combined.slice_mut(..left_len))?;
        right.copy_storage(exec, combined.slice_mut(left_len..))?;
        combined
            .read()
            .gather(exec, control.permutation.column(), output)
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
        fn apply(lhs: u32, rhs: u32) -> bool {
            lhs < rhs
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
            fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
                lhs.0 < rhs.0
            }
        }
        merge(&exec, pair_left, pair_right, LessPair, pair_out).unwrap();
    }
}
