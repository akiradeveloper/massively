//! Ordering controls derived from adjacent semantic items.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, A8, Column, DeviceVec, Dispatch, Error, Executor, MAlloc, MStorage,
    MStorageElement, ReadExpression,
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7, Eval8},
    indexed::GatherInput,
    launch::cube_count_1d,
    output::{LowerOutputExpression, OutputExpression, StageOutput},
    read::{
        Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, Env8, LowerReadExpression, Reassociate,
    },
    reduce::{StageRead, StagedBindings},
    transform::{MaterializeDispatch, materialize},
};

const BLOCK_SIZE: u32 = 256;

/// Compile-time binary predicate over two semantic items.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, sort};
/// use massively::op::BinaryPredicateOp;
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs < rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[3_u32, 1, 2]);
/// let output = exec.alloc::<u32>(input.len());
/// sort(&exec, input.slice(..), Less, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
#[cubecl::cube]
pub trait BinaryPredicateOp<Item: CubeType>: 'static + Send + Sync {
    fn apply(lhs: Item, rhs: Item) -> bool;
}

struct ReverseLess<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> BinaryPredicateOp<Item> for ReverseLess<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn apply(lhs: Item, rhs: Item) -> bool {
        Less::apply(rhs, lhs)
    }
}

#[cubecl::cube]
trait AdjacentFlagOp<Item: CubeType>: 'static + Send + Sync {
    fn first() -> u32;
    fn apply(previous: Item, current: Item) -> u32;
}

struct SortedBreak<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> AdjacentFlagOp<Item> for SortedBreak<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn first() -> u32 {
        0u32
    }

    fn apply(previous: Item, current: Item) -> u32 {
        if Less::apply(current, previous) {
            1u32
        } else {
            0u32
        }
    }
}

struct AdjacentMatch<Equal>(PhantomData<fn() -> Equal>);

#[cubecl::cube]
impl<Item, Equal> AdjacentFlagOp<Item> for AdjacentMatch<Equal>
where
    Item: CubeType + 'static,
    Equal: BinaryPredicateOp<Item>,
{
    fn first() -> u32 {
        0u32
    }

    fn apply(previous: Item, current: Item) -> u32 {
        if Equal::apply(previous, current) {
            1u32
        } else {
            0u32
        }
    }
}

pub(crate) struct UniqueHead<Equal>(PhantomData<fn() -> Equal>);

#[cubecl::cube]
impl<Item, Equal> AdjacentFlagOp<Item> for UniqueHead<Equal>
where
    Item: CubeType + 'static,
    Equal: BinaryPredicateOp<Item>,
{
    fn first() -> u32 {
        1u32
    }

    fn apply(previous: Item, current: Item) -> u32 {
        if Equal::apply(previous, current) {
            0u32
        } else {
            1u32
        }
    }
}

macro_rules! define_adjacent_flags_kernel {
    ($name:ident,$eval:ident,$method:ident; $( $leaf:ident:$slot:ident ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Op: AdjacentFlagOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            len: &[u32],
            flags: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                flags[index] = if index == 0usize {
                    Op::first()
                } else {
                    Op::apply(
                        Expr::$method($( $slot, )+ offsets, index - 1usize),
                        Expr::$method($( $slot, )+ offsets, index),
                    )
                };
            }
        }
    };
}

define_adjacent_flags_kernel!(adjacent_flags_a1,Eval1,eval1; L0:slot0);
define_adjacent_flags_kernel!(adjacent_flags_a2,Eval2,eval2; L0:slot0,L1:slot1);
define_adjacent_flags_kernel!(adjacent_flags_a3,Eval3,eval3; L0:slot0,L1:slot1,L2:slot2);
define_adjacent_flags_kernel!(adjacent_flags_a4,Eval4,eval4; L0:slot0,L1:slot1,L2:slot2,L3:slot3);
define_adjacent_flags_kernel!(adjacent_flags_a5,Eval5,eval5; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4);
define_adjacent_flags_kernel!(adjacent_flags_a6,Eval6,eval6; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5);
define_adjacent_flags_kernel!(adjacent_flags_a7,Eval7,eval7; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6);
define_adjacent_flags_kernel!(adjacent_flags_a8,Eval8,eval8; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7);

trait AdjacentFlagDispatch<R, Input, Item, Slots, Op>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error>;
}

macro_rules! impl_adjacent_flags_dispatch {
    ($arity:ty,$eval:ident,$kernel:ident; [$( $leaf:ident:$index:literal ),+],$env:ty) => {
        impl<R, Input, Item, Op, $( $leaf ),+>
            AdjacentFlagDispatch<R, Input, Item, $env, Op> for Dispatch<$arity, crate::S1>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Op: AdjacentFlagOp<Item>,
            $( $leaf: MStorageElement, )+
            Input: ReadExpression<Item = Item>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error> {
                let len = input.logical_len()?;
                let flags = exec.alloc::<u32>(len);
                if len == 0 {
                    return Ok(flags);
                }
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Op, R>(
                        exec.client(),
                        cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(flags.handle.clone(), flags.len()),
                    );
                }
                Ok(flags)
            }
        }
    };
}

impl_adjacent_flags_dispatch!(A1,Eval1,adjacent_flags_a1; [L0:0],Env1<L0>);
impl_adjacent_flags_dispatch!(A2,Eval2,adjacent_flags_a2; [L0:0,L1:1],Env2<L0,L1>);
impl_adjacent_flags_dispatch!(A3,Eval3,adjacent_flags_a3; [L0:0,L1:1,L2:2],Env3<L0,L1,L2>);
impl_adjacent_flags_dispatch!(A4,Eval4,adjacent_flags_a4; [L0:0,L1:1,L2:2,L3:3],Env4<L0,L1,L2,L3>);
impl_adjacent_flags_dispatch!(A5,Eval5,adjacent_flags_a5; [L0:0,L1:1,L2:2,L3:3,L4:4],Env5<L0,L1,L2,L3,L4>);
impl_adjacent_flags_dispatch!(A6,Eval6,adjacent_flags_a6; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5],Env6<L0,L1,L2,L3,L4,L5>);
impl_adjacent_flags_dispatch!(A7,Eval7,adjacent_flags_a7; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6],Env7<L0,L1,L2,L3,L4,L5,L6>);
impl_adjacent_flags_dispatch!(A8,Eval8,adjacent_flags_a8; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7],Env8<L0,L1,L2,L3,L4,L5,L6,L7>);

#[cubecl::cube(launch_unchecked)]
fn iota_indices_kernel(len: &[u32], indices: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        indices[index] = index as u32;
    }
}

macro_rules! define_merge_permutation_kernel {
    ($name:ident,$eval:ident,$method:ident; $( $leaf:ident:$slot:ident ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            input: &[u32],
            output: &mut [u32],
            len: &[u32],
            width: &[u32],
        ) {
            let out = RuntimeCell::<usize>::new(
                (CUBE_POS as usize) * (CUBE_DIM as usize) + UNIT_POS as usize,
            );
            let stride = (CUBE_COUNT as usize) * (CUBE_DIM as usize);
            let logical_len = len[0] as usize;
            let run_width = width[0] as usize;
            while out.read() < logical_len {
                let out_index = out.read();
                let run = out_index / run_width;
                let base = (run - run % 2usize) * run_width;
                let left_remaining = logical_len - base;
                let left_len = if left_remaining < run_width {
                    left_remaining
                } else {
                    run_width
                };
                let right_start = base + left_len;
                let right_remaining = logical_len - right_start;
                let right_len = if right_remaining < run_width {
                    right_remaining
                } else {
                    run_width
                };
                let rank = out_index - base;
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
                            Expr::$method(
                                $( $slot, )+
                                offsets,
                                input[right_start + right_rank - 1usize] as usize,
                            ),
                            Expr::$method(
                                $( $slot, )+
                                offsets,
                                input[base + left_rank] as usize,
                            ),
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
                    let left_index = input[base + left_rank];
                    if right_rank >= right_len {
                        output[out_index] = left_index;
                    } else {
                        let right_index = input[right_start + right_rank];
                        if !Less::apply(
                            Expr::$method($( $slot, )+ offsets, right_index as usize),
                            Expr::$method($( $slot, )+ offsets, left_index as usize),
                        ) {
                            output[out_index] = left_index;
                        } else {
                            output[out_index] = right_index;
                        }
                    }
                } else {
                    output[out_index] = input[right_start + right_rank];
                }
                out.store(out_index + stride);
            }
        }
    };
}

define_merge_permutation_kernel!(merge_permutation_a1,Eval1,eval1; L0:slot0);
define_merge_permutation_kernel!(merge_permutation_a2,Eval2,eval2; L0:slot0,L1:slot1);
define_merge_permutation_kernel!(merge_permutation_a3,Eval3,eval3; L0:slot0,L1:slot1,L2:slot2);
define_merge_permutation_kernel!(merge_permutation_a4,Eval4,eval4; L0:slot0,L1:slot1,L2:slot2,L3:slot3);
define_merge_permutation_kernel!(merge_permutation_a5,Eval5,eval5; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4);
define_merge_permutation_kernel!(merge_permutation_a6,Eval6,eval6; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5);
define_merge_permutation_kernel!(merge_permutation_a7,Eval7,eval7; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6);
define_merge_permutation_kernel!(merge_permutation_a8,Eval8,eval8; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7);

trait SortControlDispatch<R, Input, Item, Slots, Less>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error>;
}

macro_rules! impl_sort_control_dispatch {
    ($arity:ty,$eval:ident,$kernel:ident; [$( $leaf:ident:$index:literal ),+],$env:ty) => {
        impl<R, Input, Item, Less, $( $leaf ),+>
            SortControlDispatch<R, Input, Item, $env, Less> for Dispatch<$arity, crate::S1>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Less: BinaryPredicateOp<Item>,
            $( $leaf: MStorageElement, )+
            Input: ReadExpression<Item = Item>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error> {
                let len = input.logical_len()?;
                let mut current = exec.alloc::<u32>(len);
                if len == 0 {
                    return Ok(current);
                }
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                let count = cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?;
                unsafe {
                    iota_indices_kernel::launch_unchecked::<R>(
                        exec.client(),
                        count.clone(),
                        CubeDim::new_1d(BLOCK_SIZE),
                        BufferArg::from_raw_parts(len_handle.clone(), 1),
                        BufferArg::from_raw_parts(current.handle.clone(), current.len()),
                    );
                }
                if len == 1 {
                    return Ok(current);
                }
                let mut next = exec.alloc::<u32>(len);
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let mut width = 1usize;
                while width < len {
                    let width_u32 = u32::try_from(width)
                        .map_err(|_| Error::LengthTooLarge { len: width })?;
                    let width_handle = exec.client().create_from_slice(u32::as_bytes(&[width_u32]));
                    unsafe {
                        $kernel::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Less, R>(
                            exec.client(),
                            count.clone(),
                            CubeDim::new_1d(BLOCK_SIZE),
                            $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                            BufferArg::from_raw_parts(offsets.clone(), reads.offsets.len()),
                            BufferArg::from_raw_parts(current.handle.clone(), current.len()),
                            BufferArg::from_raw_parts(next.handle.clone(), next.len()),
                            BufferArg::from_raw_parts(len_handle.clone(), 1),
                            BufferArg::from_raw_parts(width_handle, 1),
                        );
                    }
                    core::mem::swap(&mut current, &mut next);
                    width = width.saturating_mul(2);
                }
                Ok(current)
            }
        }
    };
}

impl_sort_control_dispatch!(A1,Eval1,merge_permutation_a1; [L0:0],Env1<L0>);
impl_sort_control_dispatch!(A2,Eval2,merge_permutation_a2; [L0:0,L1:1],Env2<L0,L1>);
impl_sort_control_dispatch!(A3,Eval3,merge_permutation_a3; [L0:0,L1:1,L2:2],Env3<L0,L1,L2>);
impl_sort_control_dispatch!(A4,Eval4,merge_permutation_a4; [L0:0,L1:1,L2:2,L3:3],Env4<L0,L1,L2,L3>);
impl_sort_control_dispatch!(A5,Eval5,merge_permutation_a5; [L0:0,L1:1,L2:2,L3:3,L4:4],Env5<L0,L1,L2,L3,L4>);
impl_sort_control_dispatch!(A6,Eval6,merge_permutation_a6; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5],Env6<L0,L1,L2,L3,L4,L5>);
impl_sort_control_dispatch!(A7,Eval7,merge_permutation_a7; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6],Env7<L0,L1,L2,L3,L4,L5,L6>);
impl_sort_control_dispatch!(A8,Eval8,merge_permutation_a8; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7],Env8<L0,L1,L2,L3,L4,L5,L6,L7>);

/// Generates a stable ordering permutation from one read expression.
#[doc(hidden)]
pub trait SortControlInput<R: Runtime, Less>: ReadExpression + Sized {
    fn sort_control(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Less> SortControlInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Less: BinaryPredicateOp<Input::Item>,
    Dispatch<Input::ReadArity, crate::S1>:
        SortControlDispatch<R, Input, Input::Item, Input::Slots, Less>,
{
    fn sort_control(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<Input::ReadArity, crate::S1> as SortControlDispatch<
            R,
            Input,
            Input::Item,
            Input::Slots,
            Less,
        >>::run(exec, &self)
    }
}

pub(crate) fn sort_control_with<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: SortControlInput<R, Less>,
{
    input.sort_control(exec)
}

/// Public sort capability.  The blanket implementation first normalizes an
/// arbitrary semantic expression into canonical storage, builds a stable
/// permutation from that storage, and applies it to the output.  The
/// permutation is the only connection between ordering and payload movement.
#[doc(hidden)]
pub trait SortInput<R: Runtime, Less, Output>: ReadExpression + Sized {
    fn sort_into(self, exec: &Executor<R>, output: Output) -> Result<(), Error>;
}

impl<R, Input, Less, Output> SortInput<R, Less, Output> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: MAlloc<R>,
    <Input::Item as MAlloc<R>>::Storage: MStorage<R>,
    <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write:
        LowerOutputExpression + StageOutput<R, Env0>,
    <<<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as OutputExpression>::Item:
        crate::WriteFrom<Input::Item>,
    Dispatch<
        Input::ReadArity,
        <<<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as OutputExpression>::StorageArity,
    >: MaterializeDispatch<
            R,
            Input,
            <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write,
            Input::Slots,
            <<<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as LowerOutputExpression>::Slots,
        >,
    Input::Item: crate::WriteFrom<
            <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Item,
        >,
    Reassociate<
        <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Read,
        Input::Item,
    >: SortControlInput<R, Less>,
    <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Read:
        GatherInput<R, Column<u32>, Output>,
{
    fn sort_into(self, exec: &Executor<R>, output: Output) -> Result<(), Error> {
        let len = self.logical_len()?;
        let temporary = exec.alloc::<Input::Item>(len);
        materialize(exec, self, temporary.write())?;
        let semantic = Reassociate::<_, Input::Item>::new(temporary.read());
        let permutation = semantic.sort_control(exec)?;
        crate::indexed::gather_direct(exec, temporary.read(), permutation.column(), output)
    }
}

/// Stably sorts an input into preallocated output storage.
pub(crate) fn sort<R, Input, Less, Output>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: SortInput<R, Less, Output>,
{
    input.sort_into(exec, output)
}

pub(crate) trait AdjacentFlagInput<R: Runtime, Op>: ReadExpression + Sized {
    fn adjacent_len(&self) -> Result<usize, Error>;
    fn adjacent_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Op> AdjacentFlagInput<R, Op> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Op: AdjacentFlagOp<Input::Item>,
    Dispatch<Input::ReadArity, crate::S1>:
        AdjacentFlagDispatch<R, Input, Input::Item, Input::Slots, Op>,
{
    fn adjacent_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn adjacent_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<Input::ReadArity, crate::S1> as AdjacentFlagDispatch<
            R,
            Input,
            Input::Item,
            Input::Slots,
            Op,
        >>::run(exec, &self)
    }
}

pub(crate) fn unique_head_flags<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: AdjacentFlagInput<R, UniqueHead<Equal>>,
{
    input.adjacent_flags(exec)
}

/// Internal public-API capability for adjacent pair search.
#[doc(hidden)]
pub trait AdjacentFindInput<R: Runtime, Equal>: ReadExpression + Sized {
    fn adjacent_find_len(&self) -> Result<usize, Error>;
    fn adjacent_match_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Equal> AdjacentFindInput<R, Equal> for Input
where
    R: Runtime,
    Input: ReadExpression + AdjacentFlagInput<R, AdjacentMatch<Equal>>,
{
    fn adjacent_find_len(&self) -> Result<usize, Error> {
        self.adjacent_len()
    }

    fn adjacent_match_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        self.adjacent_flags(exec)
    }
}

/// Finds the first element of the first adjacent pair accepted by `equal`.
pub(crate) fn adjacent_find<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
    _equal: Equal,
) -> Result<Option<u32>, Error>
where
    R: Runtime,
    Input: AdjacentFindInput<R, Equal>,
{
    if input.adjacent_find_len()? < 2 {
        return Ok(None);
    }
    let flags = input.adjacent_match_flags(exec)?;
    let control = crate::selection::SelectionControl::from_flags(exec, flags)?;
    Ok(control.first_index(exec)?.map(|index| index - 1))
}

/// Internal public-API capability for stable adjacent deduplication.
#[doc(hidden)]
pub trait UniqueInput<R: Runtime, Equal, Output>: ReadExpression + Sized {
    fn unique_into(self, exec: &Executor<R>, output: Output) -> Result<u32, Error>;
}

impl<R, Input, Equal, Output> UniqueInput<R, Equal, Output> for Input
where
    R: Runtime,
    Input: ReadExpression
        + Clone
        + AdjacentFlagInput<R, UniqueHead<Equal>>
        + crate::selection::CopySelected<R, Output>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    fn unique_into(self, exec: &Executor<R>, output: Output) -> Result<u32, Error> {
        let flags = unique_head_flags::<R, _, Equal>(exec, self.clone())?;
        let control = crate::selection::SelectionControl::from_flags(exec, flags)?;
        self.copy_selected(exec, &control, output)
    }
}

/// Removes consecutive duplicates, keeping the first item in each run.
pub(crate) fn unique<R, Input, Equal, Output>(
    exec: &Executor<R>,
    input: Input,
    _equal: Equal,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: UniqueInput<R, Equal, Output>,
{
    input.unique_into(exec, output)
}

/// Internal public-API capability hiding the concrete adjacent-control
/// dispatch from the function signature.
#[doc(hidden)]
pub trait SortedInput<R: Runtime, Less>: ReadExpression + Sized {
    fn sorted_len(&self) -> Result<usize, Error>;
    fn sorted_break_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Less> SortedInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression + AdjacentFlagInput<R, SortedBreak<Less>>,
{
    fn sorted_len(&self) -> Result<usize, Error> {
        self.adjacent_len()
    }

    fn sorted_break_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        self.adjacent_flags(exec)
    }
}

struct NonZero;

#[cubecl::cube]
impl crate::PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> bool {
        input != 0u32
    }
}

/// Returns the first index at which the input ceases to be sorted, or its
/// length when the whole input is sorted.
pub(crate) fn is_sorted_until<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: SortedInput<R, Less>,
{
    let len = input.sorted_len()?;
    let end = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let flags = input.sorted_break_flags(exec)?;
    Ok(crate::predicate::find_if(exec, flags.column(), NonZero)?.unwrap_or(end))
}

/// Returns whether the input is sorted according to `less`.
pub(crate) fn is_sorted<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: SortedInput<R, Less>,
{
    let len = input.sorted_len()?;
    let end = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    Ok(is_sorted_until(exec, input, less)? == end)
}

/// Internal public-API capability for extremum queries.
#[doc(hidden)]
pub trait ExtremumInput<R: Runtime, Less>: ReadExpression + Sized {
    fn extremum_len(&self) -> Result<usize, Error>;
    fn ascending(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn descending(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Less> ExtremumInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression
        + StageRead<R, Env0>
        + SortControlInput<R, Less>
        + SortControlInput<R, ReverseLess<Less>>,
{
    fn extremum_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn ascending(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Input as SortControlInput<R, Less>>::sort_control(self, exec)
    }

    fn descending(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Input as SortControlInput<R, ReverseLess<Less>>>::sort_control(self, exec)
    }
}

/// Returns the first minimum element index.
pub(crate) fn min_element<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<Option<u32>, Error>
where
    R: Runtime,
    Input: ExtremumInput<R, Less>,
{
    if input.extremum_len()? == 0 {
        return Ok(None);
    }
    let order = input.ascending(exec)?;
    crate::scan::first_u32(exec, &order)
}

/// Returns the first maximum element index.
pub(crate) fn max_element<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<Option<u32>, Error>
where
    R: Runtime,
    Input: ExtremumInput<R, Less>,
{
    if input.extremum_len()? == 0 {
        return Ok(None);
    }
    let order = input.descending(exec)?;
    crate::scan::first_u32(exec, &order)
}

/// Returns the last minimum and first maximum indices.
pub(crate) fn minmax_element<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<Option<(u32, u32)>, Error>
where
    R: Runtime,
    Input: ExtremumInput<R, Less>,
{
    if input.extremum_len()? == 0 {
        return Ok(None);
    }
    let descending = input.descending(exec)?;
    let max = crate::scan::first_u32(exec, &descending)?.expect("non-empty permutation");
    let min = crate::scan::last_u32(exec, &descending)?;
    Ok(Some((min, max)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));
    struct LexicographicLess;

    #[cubecl::cube]
    impl BinaryPredicateOp<Seven> for LexicographicLess {
        fn apply(lhs: Seven, rhs: Seven) -> bool {
            lhs.0 < rhs.0
        }
    }

    struct EqualSeven;

    #[cubecl::cube]
    impl BinaryPredicateOp<Seven> for EqualSeven {
        fn apply(lhs: Seven, rhs: Seven) -> bool {
            lhs.0 == rhs.0
                && lhs.1.0 == rhs.1.0
                && lhs.1.1.0 == rhs.1.1.0
                && lhs.1.1.1.0 == rhs.1.1.1.0
                && lhs.1.1.1.1.0 == rhs.1.1.1.1.0
                && lhs.1.1.1.1.1.0 == rhs.1.1.1.1.1.0
                && lhs.1.1.1.1.1.1 == rhs.1.1.1.1.1.1
        }
    }

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> bool {
            lhs < rhs
        }
    }

    #[test]
    fn sorted_queries_dispatch_eval8_without_flattening_items() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let first = exec.to_device(&[1_u32, 1, 2, 2]);
        let second = exec.to_device(&[0_u32, 1, 0, 1]);
        let zeros: Vec<_> = (0..5).map(|_| exec.to_device(&[0_u32; 4])).collect();
        let make_input = || {
            let seven = Zip::new(
                first.column(),
                Zip::new(
                    second.column(),
                    Zip::new(
                        zeros[0].column(),
                        Zip::new(
                            zeros[1].column(),
                            Zip::new(
                                zeros[2].column(),
                                Zip::new(zeros[3].column(), zeros[4].column()),
                            ),
                        ),
                    ),
                ),
            );
            Permute::new(seven, Counting::new(0, 4))
        };
        assert!(is_sorted(&exec, make_input(), LexicographicLess).unwrap());

        let bad_first = exec.to_device(&[1_u32, 2, 1, 3]);
        let bad_input = Permute::new(
            Zip::new(
                bad_first.column(),
                Zip::new(
                    second.column(),
                    Zip::new(
                        zeros[0].column(),
                        Zip::new(
                            zeros[1].column(),
                            Zip::new(
                                zeros[2].column(),
                                Zip::new(zeros[3].column(), zeros[4].column()),
                            ),
                        ),
                    ),
                ),
            ),
            Counting::new(0, 4),
        );
        assert_eq!(
            is_sorted_until(&exec, bad_input, LexicographicLess).unwrap(),
            2
        );
    }

    #[test]
    fn sort_normalizes_eval8_then_applies_one_stable_permutation_to_storage7() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 513usize;
        let keys: Vec<u32> = (0..len).map(|index| (index as u32 * 37) % 23).collect();
        let rows: Vec<u32> = (0..len as u32).collect();
        let key_column = exec.to_device(&keys);
        let row_column = exec.to_device(&rows);
        let payloads: Vec<_> = (2_u32..7)
            .map(|column| {
                let values: Vec<u32> = rows.iter().map(|row| row + column * 1_000).collect();
                exec.to_device(&values)
            })
            .collect();
        let seven = Zip::new(
            key_column.column(),
            Zip::new(
                row_column.column(),
                Zip::new(
                    payloads[0].column(),
                    Zip::new(
                        payloads[1].column(),
                        Zip::new(
                            payloads[2].column(),
                            Zip::new(payloads[3].column(), payloads[4].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, len));
        let output = exec.alloc::<Seven>(len);

        sort(&exec, input, LexicographicLess, output.write()).unwrap();

        let sorted_keys = exec.to_host(&output.0.0.0.0.0.0).unwrap();
        let sorted_rows = exec.to_host(&output.0.0.0.0.0.1).unwrap();
        for index in 1..len {
            assert!(sorted_keys[index - 1] <= sorted_keys[index]);
            if sorted_keys[index - 1] == sorted_keys[index] {
                assert!(sorted_rows[index - 1] < sorted_rows[index]);
            }
        }
        let sorted_payload2 = exec.to_host(&output.0.0.0.0.1).unwrap();
        let sorted_payload6 = exec.to_host(&output.1).unwrap();
        for index in 0..len {
            assert_eq!(sorted_payload2[index], sorted_rows[index] + 2_000);
            assert_eq!(sorted_payload6[index], sorted_rows[index] + 6_000);
        }
    }

    #[test]
    fn adjacent_find_and_unique_cover_eval8_control_and_storage7_apply() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[1_u32, 1, 2, 2, 3]);
        let copies: Vec<_> = (0..6)
            .map(|_| exec.to_device(&[9_u32, 9, 8, 8, 7]))
            .collect();
        let make_input = || {
            Permute::new(
                Zip::new(
                    values.column(),
                    Zip::new(
                        copies[0].column(),
                        Zip::new(
                            copies[1].column(),
                            Zip::new(
                                copies[2].column(),
                                Zip::new(
                                    copies[3].column(),
                                    Zip::new(copies[4].column(), copies[5].column()),
                                ),
                            ),
                        ),
                    ),
                ),
                Counting::new(0, 5),
            )
        };

        assert_eq!(
            adjacent_find(&exec, make_input(), EqualSeven).unwrap(),
            Some(0)
        );
        let output = exec.alloc::<Seven>(5);
        let count = unique(&exec, make_input(), EqualSeven, output.write()).unwrap();
        assert_eq!(count, 3);
        assert_eq!(
            exec.to_host(&output.0.0.0.0.0.0.slice(..count as usize))
                .unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn extremum_queries_preserve_oracle_tie_breaking() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[2_u32, 1, 3, 1, 3]);
        assert_eq!(
            min_element(&exec, input.column(), LessU32).unwrap(),
            Some(1)
        );
        assert_eq!(
            max_element(&exec, input.column(), LessU32).unwrap(),
            Some(2)
        );
        assert_eq!(
            minmax_element(&exec, input.column(), LessU32).unwrap(),
            Some((3, 2))
        );
    }
}
