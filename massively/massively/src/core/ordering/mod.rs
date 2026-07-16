//! Ordering controls derived from adjacent semantic items.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    A13, CanonicalAlloc, DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression,
    arg_reduce::{ArgReduceDispatch, ArgReductionOp, arg_reduce},
    eval::Eval13,
    launch::cube_count_1d,
    op::IndexedBinaryOp,
    output::{LowerOutputExpression, OutputExpression, StageOutput},
    read::{
        AdjacentIndexedTransform, Env0, Env13, KernelReadSlots, LowerReadExpression,
        PaddedReadSlots,
    },
    reduce::{ReduceDispatch, ReductionOp, StageRead, StagedBindings, reduce},
    transform::{MaterializeDispatch, materialize},
};

pub(crate) mod sort;

const BLOCK_SIZE: u32 = 256;

/// Compile-time binary predicate over two semantic items.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::sort};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs < rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[3_u32, 1, 2]);
/// let output = sort(&exec, input.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
#[cubecl::cube]
pub trait BinaryPredicateOp<Item: CubeType>: 'static + Send + Sync {
    fn apply(lhs: Item, rhs: Item) -> bool;
}

#[cubecl::cube]
trait AdjacentFlagOp<Item: CubeType>: 'static + Send + Sync {
    fn first() -> u32;
    fn apply(previous: Item, current: Item) -> u32;
}

struct ArgMinFirst<Less>(PhantomData<fn() -> Less>);
struct ArgMinLast<Less>(PhantomData<fn() -> Less>);
struct ArgMaxFirst<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> ArgReductionOp<Item> for ArgMinFirst<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn rhs_wins(lhs: Item, rhs: Item) -> bool {
        Less::apply(rhs, lhs)
    }
}

#[cubecl::cube]
impl<Item, Less> ArgReductionOp<Item> for ArgMinLast<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn rhs_wins(lhs: Item, rhs: Item) -> bool {
        !Less::apply(lhs, rhs)
    }
}

#[cubecl::cube]
impl<Item, Less> ArgReductionOp<Item> for ArgMaxFirst<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn rhs_wins(lhs: Item, rhs: Item) -> bool {
        Less::apply(lhs, rhs)
    }
}

struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

struct FirstAdjacentMatch<Equal>(PhantomData<fn() -> Equal>);

#[cubecl::cube]
impl<Item, Equal> IndexedBinaryOp<Item> for FirstAdjacentMatch<Equal>
where
    Item: CubeType + 'static,
    Equal: BinaryPredicateOp<Item>,
{
    type Output = u32;

    fn apply(previous: Item, current: Item, index: u32) -> u32 {
        if index != 0u32 && Equal::apply(previous, current) {
            index - 1u32
        } else {
            4_294_967_295u32
        }
    }
}

struct FirstSortedBreak<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> IndexedBinaryOp<Item> for FirstSortedBreak<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    type Output = u32;

    fn apply(previous: Item, current: Item, index: u32) -> u32 {
        if index != 0u32 && Less::apply(current, previous) {
            index
        } else {
            4_294_967_295u32
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

pub(crate) struct SortedBreak<Less>(PhantomData<fn() -> Less>);

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
            order: &[u32],
            use_order: &[u32],
            len: &[u32],
            flags: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                flags[index] = if index == 0usize {
                    Op::first()
                } else {
                    let previous = if use_order[0] != 0u32 {
                        order[index - 1usize] as usize
                    } else {
                        index - 1usize
                    };
                    let current = if use_order[0] != 0u32 {
                        order[index] as usize
                    } else {
                        index
                    };
                    Op::apply(
                        Expr::$method($( $slot, )+ offsets, previous),
                        Expr::$method($( $slot, )+ offsets, current),
                    )
                };
            }
        }
    };
}

define_adjacent_flags_kernel!(adjacent_flags_a13,Eval13,eval13; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12);

trait AdjacentFlagDispatch<R, Input, Item, Slots, Op>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        input: &Input,
        order: Option<&DeviceVec<R, u32>>,
    ) -> Result<DeviceVec<R, u32>, Error>;
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
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(
                exec: &Executor<R>,
                input: &Input,
                order: Option<&DeviceVec<R, u32>>,
            ) -> Result<DeviceVec<R, u32>, Error> {
                let input_len = input.logical_len()?;
                let len = order.map_or(input_len, DeviceVec::len);
                if let Some(order) = order
                    && order.len() != input_len
                {
                    return Err(Error::LengthMismatch {
                        left: input_len,
                        right: order.len(),
                    });
                }
                let flags = exec.alloc_canonical::<u32>(len);
                if len == 0 {
                    return Ok(flags);
                }
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                reads.pad_to_thirteen(exec.client());
                let offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                let use_order = exec
                    .client()
                    .create_from_slice(u32::as_bytes(&[u32::from(order.is_some())]));
                let (order, order_len) = order
                    .map(|order| (order.handle.clone(), order.len()))
                    .unwrap_or_else(|| {
                        (
                            exec.client().create_from_slice(u32::as_bytes(&[0u32])),
                            1,
                        )
                    });
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Op, R>(
                        exec.client(),
                        cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(order, order_len),
                        BufferArg::from_raw_parts(use_order, 1),
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(flags.handle.clone(), flags.len()),
                    );
                }
                Ok(flags)
            }
        }
    };
}

impl_adjacent_flags_dispatch!(A13,Eval13,adjacent_flags_a13; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12],Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>);

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

define_merge_permutation_kernel!(merge_permutation_a13,Eval13,eval13; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12);

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
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error> {
                let len = input.logical_len()?;
                let mut current = exec.alloc_canonical::<u32>(len);
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
                let mut next = exec.alloc_canonical::<u32>(len);
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                reads.pad_to_thirteen(exec.client());
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

impl_sort_control_dispatch!(A13,Eval13,merge_permutation_a13; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12],Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>);

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
    Dispatch<A13, crate::S1>:
        SortControlDispatch<R, Input, Input::Item, KernelReadSlots<Input::Slots>, Less>,
{
    fn sort_control(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<A13, crate::S1> as SortControlDispatch<
            R,
            Input,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            Less,
        >>::run(exec, &self)
    }
}

pub(crate) fn unique_head_flags_ordered<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
    order: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: AdjacentFlagInput<R, UniqueHead<Equal>>,
{
    input.adjacent_flags_ordered(exec, order)
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
/// arbitrary fixed-ABI semantic expression into canonical storage, then runs
/// the storage-width sort and materializes its semantic shape into the output.
#[doc(hidden)]
pub trait SortInput<R: Runtime, Less, Output>: ReadExpression + Sized {
    fn sort_into(self, exec: &Executor<R>, output: Output) -> Result<(), Error>;
}

impl<R, Input, Less, Output> SortInput<R, Less, Output> for Input
where
    R: Runtime,
    Input: crate::allocation::NormalizeInput<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: crate::WritableFrom<Input::Item>,
    Input::Item:
        CanonicalAlloc<R, CanonicalStorage = Input::Storage> + crate::api::iter::SortAbi<R>,
    Dispatch<crate::A13, crate::S12>: MaterializeDispatch<
            R,
            Input::SemanticRead,
            Output,
            crate::read::KernelReadSlots<<Input::SemanticRead as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<<Output as LowerOutputExpression>::Slots>,
        >,
    <Output as LowerOutputExpression>::Slots: crate::output::PaddedOutputSlots,
{
    fn sort_into(self, exec: &Executor<R>, output: Output) -> Result<(), Error> {
        let temporary = self.normalize(exec)?;
        let result = <Input::Item as crate::api::iter::SortAbi<R>>::sort_storage::<Less>(
            exec, temporary, false,
        )?;
        let semantic = Input::semantic_read(&result.sorted_keys);
        materialize(exec, semantic, output)
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

/// Key-sort capability that also retains the stable source permutation.
///
/// Keys are moved by the storage-width sort itself.  Only the permutation is
/// exposed to the independent value phase, so key and value arities never form
/// one combined kernel ABI.
#[doc(hidden)]
pub trait SortKeysInput<R: Runtime, Less, Output>: ReadExpression + Sized {
    fn sort_keys_into(
        self,
        exec: &Executor<R>,
        output: Output,
    ) -> Result<sort::OrderingControl<R>, Error>;
}

impl<R, Input, Less, Output> SortKeysInput<R, Less, Output> for Input
where
    R: Runtime,
    Input: crate::allocation::NormalizeInput<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: crate::WritableFrom<Input::Item>,
    Input::Item:
        CanonicalAlloc<R, CanonicalStorage = Input::Storage> + crate::api::iter::SortAbi<R>,
    Dispatch<crate::A13, crate::S12>: MaterializeDispatch<
            R,
            Input::SemanticRead,
            Output,
            crate::read::KernelReadSlots<<Input::SemanticRead as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<<Output as LowerOutputExpression>::Slots>,
        >,
    <Output as LowerOutputExpression>::Slots: crate::output::PaddedOutputSlots,
{
    fn sort_keys_into(
        self,
        exec: &Executor<R>,
        output: Output,
    ) -> Result<sort::OrderingControl<R>, Error> {
        let temporary = self.normalize(exec)?;
        let result = <Input::Item as crate::api::iter::SortAbi<R>>::sort_storage::<Less>(
            exec, temporary, true,
        )?;
        let semantic = Input::semantic_read(&result.sorted_keys);
        materialize(exec, semantic, output)?;
        Ok(result.control)
    }
}

pub(crate) fn sort_keys_with_control<R, Input, Less, Output>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
    output: Output,
) -> Result<sort::OrderingControl<R>, Error>
where
    R: Runtime,
    Input: SortKeysInput<R, Less, Output>,
{
    input.sort_keys_into(exec, output)
}

pub(crate) trait AdjacentFlagInput<R: Runtime, Op>: ReadExpression + Sized {
    fn adjacent_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn adjacent_flags_ordered(
        self,
        exec: &Executor<R>,
        order: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Op> AdjacentFlagInput<R, Op> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Op: AdjacentFlagOp<Input::Item>,
    Dispatch<A13, crate::S1>:
        AdjacentFlagDispatch<R, Input, Input::Item, KernelReadSlots<Input::Slots>, Op>,
{
    fn adjacent_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<A13, crate::S1> as AdjacentFlagDispatch<
            R,
            Input,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            Op,
        >>::run(exec, &self, None)
    }

    fn adjacent_flags_ordered(
        self,
        exec: &Executor<R>,
        order: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<A13, crate::S1> as AdjacentFlagDispatch<
            R,
            Input,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            Op,
        >>::run(exec, &self, Some(order))
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

pub(crate) fn sorted_break_flags<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: AdjacentFlagInput<R, SortedBreak<Less>>,
{
    input.adjacent_flags(exec)
}

/// Internal public-API capability for adjacent pair search.
#[doc(hidden)]
pub trait AdjacentFindInput<R: Runtime, Equal>: ReadExpression + Sized {
    fn adjacent_find_len(&self) -> Result<usize, Error>;
    fn first_adjacent_match(self, exec: &Executor<R>) -> Result<Option<u32>, Error>;
}

impl<R, Input, Equal> AdjacentFindInput<R, Equal> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Equal: BinaryPredicateOp<Input::Item>,
    AdjacentIndexedTransform<Input, FirstAdjacentMatch<Equal>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>: ReduceDispatch<
            R,
            AdjacentIndexedTransform<Input, FirstAdjacentMatch<Equal>>,
            u32,
            MinU32,
            crate::read::KernelReadSlots<
                <AdjacentIndexedTransform<Input, FirstAdjacentMatch<Equal>> as LowerReadExpression>::Slots,
            >,
        >,
{
    fn adjacent_find_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn first_adjacent_match(self, exec: &Executor<R>) -> Result<Option<u32>, Error> {
        let index = reduce(
            exec,
            AdjacentIndexedTransform::new(self, FirstAdjacentMatch::<Equal>(PhantomData)),
            u32::MAX,
            MinU32,
        )?;
        Ok((index != u32::MAX).then_some(index))
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
    input.first_adjacent_match(exec)
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
    fn first_sorted_break(self, exec: &Executor<R>) -> Result<Option<u32>, Error>;
}

impl<R, Input, Less> SortedInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Less: BinaryPredicateOp<Input::Item>,
    AdjacentIndexedTransform<Input, FirstSortedBreak<Less>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>: ReduceDispatch<
            R,
            AdjacentIndexedTransform<Input, FirstSortedBreak<Less>>,
            u32,
            MinU32,
            crate::read::KernelReadSlots<
                <AdjacentIndexedTransform<Input, FirstSortedBreak<Less>> as LowerReadExpression>::Slots,
            >,
        >,
{
    fn sorted_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn first_sorted_break(self, exec: &Executor<R>) -> Result<Option<u32>, Error> {
        let index = reduce(
            exec,
            AdjacentIndexedTransform::new(self, FirstSortedBreak::<Less>(PhantomData)),
            u32::MAX,
            MinU32,
        )?;
        Ok((index != u32::MAX).then_some(index))
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
    Ok(input.first_sorted_break(exec)?.unwrap_or(end))
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
    fn first_minimum(&self, exec: &Executor<R>) -> Result<Option<u32>, Error>;
    fn last_minimum(&self, exec: &Executor<R>) -> Result<Option<u32>, Error>;
    fn first_maximum(&self, exec: &Executor<R>) -> Result<Option<u32>, Error>;
}

impl<R, Input, Less> ExtremumInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0> + Clone,
    Less: BinaryPredicateOp<Input::Item>,
    Dispatch<crate::A13, crate::S1>: ArgReduceDispatch<R, Input, ArgMinFirst<Less>, crate::read::KernelReadSlots<Input::Slots>>
        + ArgReduceDispatch<R, Input, ArgMinLast<Less>, crate::read::KernelReadSlots<Input::Slots>>
        + ArgReduceDispatch<R, Input, ArgMaxFirst<Less>, crate::read::KernelReadSlots<Input::Slots>>,
{
    fn extremum_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn first_minimum(&self, exec: &Executor<R>) -> Result<Option<u32>, Error> {
        arg_reduce(exec, self.clone(), ArgMinFirst::<Less>(PhantomData))
    }

    fn last_minimum(&self, exec: &Executor<R>) -> Result<Option<u32>, Error> {
        arg_reduce(exec, self.clone(), ArgMinLast::<Less>(PhantomData))
    }

    fn first_maximum(&self, exec: &Executor<R>) -> Result<Option<u32>, Error> {
        arg_reduce(exec, self.clone(), ArgMaxFirst::<Less>(PhantomData))
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
    input.first_minimum(exec)
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
    input.first_maximum(exec)
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
    let min = input
        .last_minimum(exec)?
        .expect("non-empty input has a minimum");
    let max = input
        .first_maximum(exec)?
        .expect("non-empty input has a maximum");
    Ok(Some((min, max)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalStorage, Counting, Permute, Transform, Zip};
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
            Permute::new(
                seven,
                Transform::new(Counting::new(0, 4), crate::op::U32ToUsize),
            )
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
            Transform::new(Counting::new(0, 4), crate::op::U32ToUsize),
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
        let input = Permute::new(
            seven,
            Transform::new(Counting::new(0, len), crate::op::U32ToUsize),
        );
        let output = exec.alloc_canonical::<Seven>(len);

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
                Transform::new(Counting::new(0, 5), crate::op::U32ToUsize),
            )
        };

        assert_eq!(
            adjacent_find(&exec, make_input(), EqualSeven).unwrap(),
            Some(0)
        );
        let output = exec.alloc_canonical::<Seven>(5);
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
