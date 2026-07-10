//! Search controls over independently normalized semantic inputs.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, DeviceVec, Error, Executor, MStorageElement, ReadExpression,
    StorageLayout,
    allocation::NormalizeInput,
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7},
    ordering::BinaryPredicateOp,
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
};

const BLOCK_SIZE: u32 = 256;

struct CodeNonZero;

#[cubecl::cube]
impl crate::UnaryOp<u32> for CodeNonZero {
    type Output = u32;
    fn apply(input: u32) -> u32 {
        if input != 0u32 { 1u32 } else { 0u32 }
    }
}

#[cubecl::cube]
trait PairCodeOp<Item: CubeType>: 'static + Send + Sync {
    fn code(left: Item, right: Item, left_again: Item, right_again: Item) -> u32;
}

struct MismatchCode<Equal>(PhantomData<fn() -> Equal>);

#[cubecl::cube]
impl<Item, Equal> PairCodeOp<Item> for MismatchCode<Equal>
where
    Item: CubeType + 'static,
    Equal: BinaryPredicateOp<Item>,
{
    fn code(left: Item, right: Item, _left_again: Item, _right_again: Item) -> u32 {
        if Equal::apply(left, right) {
            0u32
        } else {
            1u32
        }
    }
}

struct LexicographicalCode<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> PairCodeOp<Item> for LexicographicalCode<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn code(left: Item, right: Item, left_again: Item, right_again: Item) -> u32 {
        if Less::apply(left, right) {
            1u32
        } else if Less::apply(right_again, left_again) {
            2u32
        } else {
            0u32
        }
    }
}

macro_rules! define_pair_code_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$left_slot:ident:$right_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Left: $eval<Item, $( $leaf ),+>,
            Right: $eval<Item, $( $leaf ),+>,
            Op: PairCodeOp<Item>,
        >(
            $( $left_slot: &[$leaf], )+
            left_offsets: &[u32],
            $( $right_slot: &[$leaf], )+
            right_offsets: &[u32],
            len: &[u32],
            codes: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                codes[index] = Op::code(
                    Left::$method($( $left_slot, )+ left_offsets, index),
                    Right::$method($( $right_slot, )+ right_offsets, index),
                    Left::$method($( $left_slot, )+ left_offsets, index),
                    Right::$method($( $right_slot, )+ right_offsets, index),
                );
            }
        }
    };
}

define_pair_code_kernel!(pair_code_s1,Eval1,eval1; [L0:left0:right0]);
define_pair_code_kernel!(pair_code_s2,Eval2,eval2; [L0:left0:right0,L1:left1:right1]);
define_pair_code_kernel!(pair_code_s3,Eval3,eval3; [L0:left0:right0,L1:left1:right1,L2:left2:right2]);
define_pair_code_kernel!(pair_code_s4,Eval4,eval4; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3]);
define_pair_code_kernel!(pair_code_s5,Eval5,eval5; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3,L4:left4:right4]);
define_pair_code_kernel!(pair_code_s6,Eval6,eval6; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3,L4:left4:right4,L5:left5:right5]);
define_pair_code_kernel!(pair_code_s7,Eval7,eval7; [L0:left0:right0,L1:left1:right1,L2:left2:right2,L3:left3:right3,L4:left4:right4,L5:left5:right5,L6:left6:right6]);

macro_rules! define_find_first_of_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$source_slot:ident:$needle_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Source: $eval<Item, $( $leaf ),+>,
            Needles: $eval<Item, $( $leaf ),+>,
            Equal: BinaryPredicateOp<Item>,
        >(
            $( $source_slot: &[$leaf], )+
            source_offsets: &[u32],
            $( $needle_slot: &[$leaf], )+
            needle_offsets: &[u32],
            source_len: &[u32],
            needle_len: &[u32],
            best: &[Atomic<u32>],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < source_len[0] as usize && (index as u32) < best[0].load() {
                let needle = RuntimeCell::<usize>::new(0usize);
                while needle.read() < needle_len[0] as usize
                    && (index as u32) < best[0].load()
                {
                    if Equal::apply(
                        Source::$method($( $source_slot, )+ source_offsets, index),
                        Needles::$method($( $needle_slot, )+ needle_offsets, needle.read()),
                    ) {
                        best[0].fetch_min(index as u32);
                        needle.store(needle_len[0] as usize);
                    } else {
                        needle.store(needle.read() + 1usize);
                    }
                }
            }
        }
    };
}

define_find_first_of_kernel!(find_first_of_s1,Eval1,eval1; [L0:source0:needle0]);
define_find_first_of_kernel!(find_first_of_s2,Eval2,eval2; [L0:source0:needle0,L1:source1:needle1]);
define_find_first_of_kernel!(find_first_of_s3,Eval3,eval3; [L0:source0:needle0,L1:source1:needle1,L2:source2:needle2]);
define_find_first_of_kernel!(find_first_of_s4,Eval4,eval4; [L0:source0:needle0,L1:source1:needle1,L2:source2:needle2,L3:source3:needle3]);
define_find_first_of_kernel!(find_first_of_s5,Eval5,eval5; [L0:source0:needle0,L1:source1:needle1,L2:source2:needle2,L3:source3:needle3,L4:source4:needle4]);
define_find_first_of_kernel!(find_first_of_s6,Eval6,eval6; [L0:source0:needle0,L1:source1:needle1,L2:source2:needle2,L3:source3:needle3,L4:source4:needle4,L5:source5:needle5]);
define_find_first_of_kernel!(find_first_of_s7,Eval7,eval7; [L0:source0:needle0,L1:source1:needle1,L2:source2:needle2,L3:source3:needle3,L4:source4:needle4,L5:source5:needle5,L6:source6:needle6]);

#[cubecl::cube]
trait BoundOp<Item: CubeType>: 'static + Send + Sync {
    fn go_right(candidate: Item, value: Item) -> bool;
}

struct LowerBound<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> BoundOp<Item> for LowerBound<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn go_right(candidate: Item, value: Item) -> bool {
        Less::apply(candidate, value)
    }
}

struct UpperBound<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> BoundOp<Item> for UpperBound<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn go_right(candidate: Item, value: Item) -> bool {
        !Less::apply(value, candidate)
    }
}

macro_rules! define_bound_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$source_slot:ident:$value_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Source: $eval<Item, $( $leaf ),+>,
            Values: $eval<Item, $( $leaf ),+>,
            Op: BoundOp<Item>,
        >(
            $( $source_slot: &[$leaf], )+
            source_offsets: &[u32],
            $( $value_slot: &[$leaf], )+
            value_offsets: &[u32],
            source_len: &[u32],
            value_len: &[u32],
            bounds: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < value_len[0] as usize {
                let low = RuntimeCell::<usize>::new(0usize);
                let high = RuntimeCell::<usize>::new(source_len[0] as usize);
                while low.read() < high.read() {
                    let mid = (low.read() + high.read()) / 2usize;
                    if Op::go_right(
                        Source::$method($( $source_slot, )+ source_offsets, mid),
                        Values::$method($( $value_slot, )+ value_offsets, index),
                    ) {
                        low.store(mid + 1usize);
                    } else {
                        high.store(mid);
                    }
                }
                bounds[index] = low.read() as u32;
            }
        }
    };
}

define_bound_kernel!(bound_s1,Eval1,eval1; [L0:source0:value0]);
define_bound_kernel!(bound_s2,Eval2,eval2; [L0:source0:value0,L1:source1:value1]);
define_bound_kernel!(bound_s3,Eval3,eval3; [L0:source0:value0,L1:source1:value1,L2:source2:value2]);
define_bound_kernel!(bound_s4,Eval4,eval4; [L0:source0:value0,L1:source1:value1,L2:source2:value2,L3:source3:value3]);
define_bound_kernel!(bound_s5,Eval5,eval5; [L0:source0:value0,L1:source1:value1,L2:source2:value2,L3:source3:value3,L4:source4:value4]);
define_bound_kernel!(bound_s6,Eval6,eval6; [L0:source0:value0,L1:source1:value1,L2:source2:value2,L3:source3:value3,L4:source4:value4,L5:source5:value5]);
define_bound_kernel!(bound_s7,Eval7,eval7; [L0:source0:value0,L1:source1:value1,L2:source2:value2,L3:source3:value3,L4:source4:value4,L5:source5:value5,L6:source6:value6]);

struct PairDispatch<Storage>(PhantomData<fn() -> Storage>);

trait PairCodeDispatch<R, Left, Right, Item, LeftSlots, RightSlots, Op>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        left: &Left,
        right: &Right,
    ) -> Result<(DeviceVec<R, u32>, usize, usize), Error>;
}

macro_rules! impl_pair_code_dispatch {
    ($storage:ty,$arity:ty,$eval:ident,$kernel:ident,$env:ty; [$( $leaf:ident:$index:literal ),+]) => {
        impl<R, Left, Right, Item, Op, $( $leaf ),+>
            PairCodeDispatch<R, Left, Right, Item, $env, $env, Op>
            for PairDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Op: PairCodeOp<Item>,
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
            fn run(
                exec: &Executor<R>,
                left: &Left,
                right: &Right,
            ) -> Result<(DeviceVec<R, u32>, usize, usize), Error> {
                let left_len = left.logical_len()?;
                let right_len = right.logical_len()?;
                let len = left_len.min(right_len);
                let codes = exec.alloc::<u32>(len);
                if len == 0 {
                    return Ok((codes, left_len, right_len));
                }
                let mut left_bindings = StagedBindings::new();
                let mut right_bindings = StagedBindings::new();
                left.stage_at(exec.client(), exec.id(), &mut left_bindings)?;
                right.stage_at(exec.client(), exec.id(), &mut right_bindings)?;
                let left_offsets = exec.client().create_from_slice(u32::as_bytes(&left_bindings.offsets));
                let right_offsets = exec.client().create_from_slice(u32::as_bytes(&right_bindings.offsets));
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $leaf, )+ Left::DeviceExpr, Right::DeviceExpr, Op, R>(
                        exec.client(),
                        crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(left_bindings.slots[$index].0.clone(), left_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(left_offsets, left_bindings.offsets.len()),
                        $( BufferArg::from_raw_parts(right_bindings.slots[$index].0.clone(), right_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(right_offsets, right_bindings.offsets.len()),
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(codes.handle.clone(), codes.len()),
                    );
                }
                Ok((codes, left_len, right_len))
            }
        }
    };
}

impl_pair_code_dispatch!(crate::S1,A1,Eval1,pair_code_s1,Env1<L0>; [L0:0]);
impl_pair_code_dispatch!(crate::S2,A2,Eval2,pair_code_s2,Env2<L0,L1>; [L0:0,L1:1]);
impl_pair_code_dispatch!(crate::S3,A3,Eval3,pair_code_s3,Env3<L0,L1,L2>; [L0:0,L1:1,L2:2]);
impl_pair_code_dispatch!(crate::S4,A4,Eval4,pair_code_s4,Env4<L0,L1,L2,L3>; [L0:0,L1:1,L2:2,L3:3]);
impl_pair_code_dispatch!(crate::S5,A5,Eval5,pair_code_s5,Env5<L0,L1,L2,L3,L4>; [L0:0,L1:1,L2:2,L3:3,L4:4]);
impl_pair_code_dispatch!(crate::S6,A6,Eval6,pair_code_s6,Env6<L0,L1,L2,L3,L4,L5>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5]);
impl_pair_code_dispatch!(crate::S7,A7,Eval7,pair_code_s7,Env7<L0,L1,L2,L3,L4,L5,L6>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6]);

trait FindFirstDispatch<R, Source, Needles, Item, SourceSlots, NeedleSlots, Equal>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, source: &Source, needles: &Needles) -> Result<Option<u32>, Error>;
}

trait BoundDispatch<R, Source, Values, Item, SourceSlots, ValueSlots, Op>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        source: &Source,
        values: &Values,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

macro_rules! impl_range_query_dispatch {
    ($storage:ty,$arity:ty,$eval:ident,$find_kernel:ident,$bound_kernel:ident,$env:ty; [$( $leaf:ident:$index:literal ),+]) => {
        impl<R, Source, Needles, Item, Equal, $( $leaf ),+>
            FindFirstDispatch<R, Source, Needles, Item, $env, $env, Equal>
            for PairDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Equal: BinaryPredicateOp<Item>,
            $( $leaf: MStorageElement, )+
            Source: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Needles: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Source::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Needles::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(
                exec: &Executor<R>,
                source: &Source,
                needles: &Needles,
            ) -> Result<Option<u32>, Error> {
                let source_len = source.logical_len()?;
                let needle_len = needles.logical_len()?;
                if source_len == 0 || needle_len == 0 {
                    return Ok(None);
                }
                let mut source_bindings = StagedBindings::new();
                let mut needle_bindings = StagedBindings::new();
                source.stage_at(exec.client(), exec.id(), &mut source_bindings)?;
                needles.stage_at(exec.client(), exec.id(), &mut needle_bindings)?;
                let source_offsets = exec.client().create_from_slice(u32::as_bytes(&source_bindings.offsets));
                let needle_offsets = exec.client().create_from_slice(u32::as_bytes(&needle_bindings.offsets));
                let source_len_u32 = u32::try_from(source_len).map_err(|_| Error::LengthTooLarge { len: source_len })?;
                let needle_len_u32 = u32::try_from(needle_len).map_err(|_| Error::LengthTooLarge { len: needle_len })?;
                let source_len_handle = exec.client().create_from_slice(u32::as_bytes(&[source_len_u32]));
                let needle_len_handle = exec.client().create_from_slice(u32::as_bytes(&[needle_len_u32]));
                let best = exec.to_device(&[source_len_u32]);
                unsafe {
                    $find_kernel::launch_unchecked::<Item, $( $leaf, )+ Source::DeviceExpr, Needles::DeviceExpr, Equal, R>(
                        exec.client(),
                        crate::launch::cube_count_1d(source_len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(source_bindings.slots[$index].0.clone(), source_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(source_offsets, source_bindings.offsets.len()),
                        $( BufferArg::from_raw_parts(needle_bindings.slots[$index].0.clone(), needle_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(needle_offsets, needle_bindings.offsets.len()),
                        BufferArg::from_raw_parts(source_len_handle, 1),
                        BufferArg::from_raw_parts(needle_len_handle, 1),
                        BufferArg::from_raw_parts(best.handle.clone(), best.len()),
                    );
                }
                let index = exec.to_host(&best)?[0];
                Ok((index < source_len_u32).then_some(index))
            }
        }

        impl<R, Source, Values, Item, Op, $( $leaf ),+>
            BoundDispatch<R, Source, Values, Item, $env, $env, Op>
            for PairDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Op: BoundOp<Item>,
            $( $leaf: MStorageElement, )+
            Source: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Values: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Source::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Values::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(
                exec: &Executor<R>,
                source: &Source,
                values: &Values,
            ) -> Result<DeviceVec<R, u32>, Error> {
                let source_len = source.logical_len()?;
                let value_len = values.logical_len()?;
                let bounds = exec.alloc::<u32>(value_len);
                if value_len == 0 {
                    return Ok(bounds);
                }
                let mut source_bindings = StagedBindings::new();
                let mut value_bindings = StagedBindings::new();
                source.stage_at(exec.client(), exec.id(), &mut source_bindings)?;
                values.stage_at(exec.client(), exec.id(), &mut value_bindings)?;
                let source_offsets = exec.client().create_from_slice(u32::as_bytes(&source_bindings.offsets));
                let value_offsets = exec.client().create_from_slice(u32::as_bytes(&value_bindings.offsets));
                let source_len_u32 = u32::try_from(source_len).map_err(|_| Error::LengthTooLarge { len: source_len })?;
                let value_len_u32 = u32::try_from(value_len).map_err(|_| Error::LengthTooLarge { len: value_len })?;
                let source_len_handle = exec.client().create_from_slice(u32::as_bytes(&[source_len_u32]));
                let value_len_handle = exec.client().create_from_slice(u32::as_bytes(&[value_len_u32]));
                unsafe {
                    $bound_kernel::launch_unchecked::<Item, $( $leaf, )+ Source::DeviceExpr, Values::DeviceExpr, Op, R>(
                        exec.client(),
                        crate::launch::cube_count_1d(value_len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(source_bindings.slots[$index].0.clone(), source_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(source_offsets, source_bindings.offsets.len()),
                        $( BufferArg::from_raw_parts(value_bindings.slots[$index].0.clone(), value_bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(value_offsets, value_bindings.offsets.len()),
                        BufferArg::from_raw_parts(source_len_handle, 1),
                        BufferArg::from_raw_parts(value_len_handle, 1),
                        BufferArg::from_raw_parts(bounds.handle.clone(), bounds.len()),
                    );
                }
                Ok(bounds)
            }
        }
    };
}

impl_range_query_dispatch!(crate::S1,A1,Eval1,find_first_of_s1,bound_s1,Env1<L0>; [L0:0]);
impl_range_query_dispatch!(crate::S2,A2,Eval2,find_first_of_s2,bound_s2,Env2<L0,L1>; [L0:0,L1:1]);
impl_range_query_dispatch!(crate::S3,A3,Eval3,find_first_of_s3,bound_s3,Env3<L0,L1,L2>; [L0:0,L1:1,L2:2]);
impl_range_query_dispatch!(crate::S4,A4,Eval4,find_first_of_s4,bound_s4,Env4<L0,L1,L2,L3>; [L0:0,L1:1,L2:2,L3:3]);
impl_range_query_dispatch!(crate::S5,A5,Eval5,find_first_of_s5,bound_s5,Env5<L0,L1,L2,L3,L4>; [L0:0,L1:1,L2:2,L3:3,L4:4]);
impl_range_query_dispatch!(crate::S6,A6,Eval6,find_first_of_s6,bound_s6,Env6<L0,L1,L2,L3,L4,L5>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5]);
impl_range_query_dispatch!(crate::S7,A7,Eval7,find_first_of_s7,bound_s7,Env7<L0,L1,L2,L3,L4,L5,L6>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6]);

trait PairCodeInput<R: Runtime, Right, Op>: NormalizeInput<R> {
    fn pair_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<(DeviceVec<R, u32>, usize, usize), Error>;
}

/// Internal public-API capability for `find_first_of`.
#[doc(hidden)]
pub trait FindFirstOfInput<R: Runtime, Needles, Equal>: NormalizeInput<R> {
    fn find_first(self, exec: &Executor<R>, needles: Needles) -> Result<Option<u32>, Error>;
}

impl<R, Source, Needles, Equal> FindFirstOfInput<R, Needles, Equal> for Source
where
    R: Runtime,
    Source: NormalizeInput<R>,
    Source::Item: StorageLayout,
    Needles: NormalizeInput<R> + ReadExpression<Item = Source::Item>,
    Source::SemanticRead: LowerReadExpression,
    Needles::SemanticRead: LowerReadExpression,
    PairDispatch<<Source::Item as StorageLayout>::StorageArity>: FindFirstDispatch<
            R,
            Source::SemanticRead,
            Needles::SemanticRead,
            Source::Item,
            <Source::SemanticRead as LowerReadExpression>::Slots,
            <Needles::SemanticRead as LowerReadExpression>::Slots,
            Equal,
        >,
{
    fn find_first(self, exec: &Executor<R>, needles: Needles) -> Result<Option<u32>, Error> {
        let source = self.normalize(exec)?;
        let needles = needles.normalize(exec)?;
        let source_read = Source::semantic_read(&source);
        let needle_read = Needles::semantic_read(&needles);
        <PairDispatch<<Source::Item as StorageLayout>::StorageArity> as FindFirstDispatch<
            R,
            Source::SemanticRead,
            Needles::SemanticRead,
            Source::Item,
            <Source::SemanticRead as LowerReadExpression>::Slots,
            <Needles::SemanticRead as LowerReadExpression>::Slots,
            Equal,
        >>::run(exec, &source_read, &needle_read)
    }
}

/// Finds the first source item equal to any needle.
pub(crate) fn find_first_of<R, Source, Needles, Equal>(
    exec: &Executor<R>,
    source: Source,
    needles: Needles,
    _equal: Equal,
) -> Result<Option<u32>, Error>
where
    R: Runtime,
    Source: FindFirstOfInput<R, Needles, Equal>,
{
    source.find_first(exec, needles)
}

/// Internal public-API capability for batched sorted bounds.
#[doc(hidden)]
pub trait SortedBoundsInput<R: Runtime, Values, Less>: NormalizeInput<R> {
    fn lower_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error>;
    fn upper_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Source, Values, Less> SortedBoundsInput<R, Values, Less> for Source
where
    R: Runtime,
    Source: NormalizeInput<R>,
    Source::Item: StorageLayout,
    Values: NormalizeInput<R> + ReadExpression<Item = Source::Item>,
    Source::SemanticRead: LowerReadExpression,
    Values::SemanticRead: LowerReadExpression,
    PairDispatch<<Source::Item as StorageLayout>::StorageArity>: BoundDispatch<
            R,
            Source::SemanticRead,
            Values::SemanticRead,
            Source::Item,
            <Source::SemanticRead as LowerReadExpression>::Slots,
            <Values::SemanticRead as LowerReadExpression>::Slots,
            LowerBound<Less>,
        > + BoundDispatch<
            R,
            Source::SemanticRead,
            Values::SemanticRead,
            Source::Item,
            <Source::SemanticRead as LowerReadExpression>::Slots,
            <Values::SemanticRead as LowerReadExpression>::Slots,
            UpperBound<Less>,
        >,
{
    fn lower_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error> {
        let source = self.normalize(exec)?;
        let values = values.normalize(exec)?;
        let source_read = Source::semantic_read(&source);
        let value_read = Values::semantic_read(&values);
        <PairDispatch<<Source::Item as StorageLayout>::StorageArity> as BoundDispatch<
            R,
            Source::SemanticRead,
            Values::SemanticRead,
            Source::Item,
            <Source::SemanticRead as LowerReadExpression>::Slots,
            <Values::SemanticRead as LowerReadExpression>::Slots,
            LowerBound<Less>,
        >>::run(exec, &source_read, &value_read)
    }

    fn upper_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error> {
        let source = self.normalize(exec)?;
        let values = values.normalize(exec)?;
        let source_read = Source::semantic_read(&source);
        let value_read = Values::semantic_read(&values);
        <PairDispatch<<Source::Item as StorageLayout>::StorageArity> as BoundDispatch<
            R,
            Source::SemanticRead,
            Values::SemanticRead,
            Source::Item,
            <Source::SemanticRead as LowerReadExpression>::Slots,
            <Values::SemanticRead as LowerReadExpression>::Slots,
            UpperBound<Less>,
        >>::run(exec, &source_read, &value_read)
    }
}

pub(crate) fn lower_bounds_storage<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    _less: Less,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Source: SortedBoundsInput<R, Values, Less>,
{
    source.lower_bounds(exec, values)
}

pub(crate) fn upper_bounds_storage<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    _less: Less,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Source: SortedBoundsInput<R, Values, Less>,
{
    source.upper_bounds(exec, values)
}

/// Finds the lower bound of each value in a sorted source.
pub(crate) fn lower_bound<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    _less: Less,
    output: crate::DeviceSliceMut<u32>,
) -> Result<(), Error>
where
    R: Runtime,
    Source: SortedBoundsInput<R, Values, Less>,
{
    let bounds = source.lower_bounds(exec, values)?;
    crate::materialize(exec, bounds.column(), output)
}

/// Finds the upper bound of each value in a sorted source.
pub(crate) fn upper_bound<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    _less: Less,
    output: crate::DeviceSliceMut<u32>,
) -> Result<(), Error>
where
    R: Runtime,
    Source: SortedBoundsInput<R, Values, Less>,
{
    let bounds = source.upper_bounds(exec, values)?;
    crate::materialize(exec, bounds.column(), output)
}

impl<R, Left, Right, Op> PairCodeInput<R, Right, Op> for Left
where
    R: Runtime,
    Left: NormalizeInput<R>,
    Left::Item: StorageLayout,
    Right: NormalizeInput<R> + ReadExpression<Item = Left::Item>,
    PairDispatch<<Left::Item as StorageLayout>::StorageArity>: PairCodeDispatch<
            R,
            Left::SemanticRead,
            Right::SemanticRead,
            Left::Item,
            <Left::SemanticRead as LowerReadExpression>::Slots,
            <Right::SemanticRead as LowerReadExpression>::Slots,
            Op,
        >,
    Left::SemanticRead: LowerReadExpression,
    Right::SemanticRead: LowerReadExpression,
{
    fn pair_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<(DeviceVec<R, u32>, usize, usize), Error> {
        let left = self.normalize(exec)?;
        let right = right.normalize(exec)?;
        let left_read = Left::semantic_read(&left);
        let right_read = Right::semantic_read(&right);
        <PairDispatch<<Left::Item as StorageLayout>::StorageArity> as PairCodeDispatch<
            R,
            Left::SemanticRead,
            Right::SemanticRead,
            Left::Item,
            <Left::SemanticRead as LowerReadExpression>::Slots,
            <Right::SemanticRead as LowerReadExpression>::Slots,
            Op,
        >>::run(exec, &left_read, &right_read)
    }
}

fn first_nonzero<R: Runtime>(
    exec: &Executor<R>,
    codes: DeviceVec<R, u32>,
) -> Result<Option<(u32, u32)>, Error> {
    let flags = exec.alloc::<u32>(codes.len());
    crate::transform::transform(exec, codes.column(), CodeNonZero, flags.slice_mut(..))?;
    let control = crate::selection::SelectionControl::from_flags(exec, flags)?;
    let Some(index) = control.first_index(exec)? else {
        return Ok(None);
    };
    // Read only the selected code; the index itself remains a device control.
    let code = exec.to_host(&codes.slice(index as usize..index as usize + 1))?[0];
    Ok(Some((index, code)))
}

/// Internal capability hiding normalized pair-code dispatch for equality.
#[doc(hidden)]
pub trait EqualityInput<R: Runtime, Right, Equal>: ReadExpression + Sized {
    fn mismatch_control(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<(DeviceVec<R, u32>, usize, usize), Error>;
}

impl<R, Left, Right, Equal> EqualityInput<R, Right, Equal> for Left
where
    R: Runtime,
    Left: ReadExpression + PairCodeInput<R, Right, MismatchCode<Equal>>,
{
    fn mismatch_control(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<(DeviceVec<R, u32>, usize, usize), Error> {
        self.pair_codes(exec, right)
    }
}

/// Returns whether two ranges have equal length and equal items.
pub(crate) fn equal<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _equal: Equal,
) -> Result<bool, Error>
where
    R: Runtime,
    Left: EqualityInput<R, Right, Equal>,
{
    let (codes, left_len, right_len) = left.mismatch_control(exec, right)?;
    Ok(left_len == right_len && first_nonzero(exec, codes)?.is_none())
}

/// Returns the first mismatch, including the shared end when lengths differ.
pub(crate) fn mismatch<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _equal: Equal,
) -> Result<Option<u32>, Error>
where
    R: Runtime,
    Left: EqualityInput<R, Right, Equal>,
{
    let (codes, left_len, right_len) = left.mismatch_control(exec, right)?;
    if let Some((index, _)) = first_nonzero(exec, codes)? {
        Ok(Some(index))
    } else if left_len != right_len {
        Ok(Some(left_len.min(right_len) as u32))
    } else {
        Ok(None)
    }
}

/// Internal capability hiding normalized lexicographical dispatch.
#[doc(hidden)]
pub trait LexicographicalInput<R: Runtime, Right, Less>: ReadExpression + Sized {
    fn lexicographical_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<(DeviceVec<R, u32>, usize, usize), Error>;
}

impl<R, Left, Right, Less> LexicographicalInput<R, Right, Less> for Left
where
    R: Runtime,
    Left: ReadExpression + PairCodeInput<R, Right, LexicographicalCode<Less>>,
{
    fn lexicographical_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<(DeviceVec<R, u32>, usize, usize), Error> {
        self.pair_codes(exec, right)
    }
}

/// Lexicographically compares two semantic item ranges.
pub(crate) fn lexicographical_compare<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<bool, Error>
where
    R: Runtime,
    Left: LexicographicalInput<R, Right, Less>,
{
    let (codes, left_len, right_len) = left.lexicographical_codes(exec, right)?;
    Ok(match first_nonzero(exec, codes)? {
        Some((_, 1)) => true,
        Some(_) => false,
        None => left_len < right_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));

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

    struct LessSeven;

    #[cubecl::cube]
    impl BinaryPredicateOp<Seven> for LessSeven {
        fn apply(lhs: Seven, rhs: Seven) -> bool {
            lhs.0 < rhs.0
        }
    }

    struct EqualU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for EqualU32 {
        fn apply(lhs: u32, rhs: u32) -> bool {
            lhs == rhs
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
    fn pair_queries_normalize_two_eval8_inputs_before_storage7_comparison() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left0 = exec.to_device(&[1_u32, 2, 3]);
        let right0 = exec.to_device(&[1_u32, 4, 3]);
        let left_tail: Vec<_> = (1_u32..7)
            .map(|value| exec.to_device(&[value; 3]))
            .collect();
        let right_tail: Vec<_> = (1_u32..7)
            .map(|value| exec.to_device(&[value; 3]))
            .collect();
        let make_left = || {
            Permute::new(
                Zip::new(
                    left0.column(),
                    Zip::new(
                        left_tail[0].column(),
                        Zip::new(
                            left_tail[1].column(),
                            Zip::new(
                                left_tail[2].column(),
                                Zip::new(
                                    left_tail[3].column(),
                                    Zip::new(left_tail[4].column(), left_tail[5].column()),
                                ),
                            ),
                        ),
                    ),
                ),
                Counting::new(0, 3),
            )
        };
        let make_right = || {
            Permute::new(
                Zip::new(
                    right0.column(),
                    Zip::new(
                        right_tail[0].column(),
                        Zip::new(
                            right_tail[1].column(),
                            Zip::new(
                                right_tail[2].column(),
                                Zip::new(
                                    right_tail[3].column(),
                                    Zip::new(right_tail[4].column(), right_tail[5].column()),
                                ),
                            ),
                        ),
                    ),
                ),
                Counting::new(0, 3),
            )
        };

        assert!(!equal(&exec, make_left(), make_right(), EqualSeven).unwrap());
        assert_eq!(
            mismatch(&exec, make_left(), make_right(), EqualSeven).unwrap(),
            Some(1)
        );
        assert!(lexicographical_compare(&exec, make_left(), make_right(), LessSeven).unwrap());
    }

    #[test]
    fn range_queries_cover_empty_needles_and_batched_bounds() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let source = exec.to_device(&[1_u32, 2, 2, 4]);
        let needles = exec.to_device(&[9_u32, 2]);
        let empty = exec.to_device::<u32>(&[]);
        assert_eq!(
            find_first_of(&exec, source.column(), needles.column(), EqualU32).unwrap(),
            Some(1)
        );
        assert_eq!(
            find_first_of(&exec, source.column(), empty.column(), EqualU32).unwrap(),
            None
        );

        let values = exec.to_device(&[0_u32, 2, 3, 5]);
        let lower = exec.to_device(&[99_u32; 4]);
        let upper = exec.to_device(&[99_u32; 4]);
        lower_bound(
            &exec,
            source.column(),
            values.column(),
            LessU32,
            lower.slice_mut(..),
        )
        .unwrap();
        upper_bound(
            &exec,
            source.column(),
            values.column(),
            LessU32,
            upper.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&lower).unwrap(), vec![0, 1, 3, 4]);
        assert_eq!(exec.to_host(&upper).unwrap(), vec![0, 3, 3, 4]);
    }

    #[test]
    fn lexicographical_direction_code_distinguishes_greater_left_item() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[93_u32]);
        let right = exec.to_device(&[43_u32]);
        let (codes, _, _) =
            <_ as PairCodeInput<WgpuRuntime, _, LexicographicalCode<LessU32>>>::pair_codes(
                left.column(),
                &exec,
                right.column(),
            )
            .unwrap();

        assert_eq!(exec.to_host(&codes).unwrap(), vec![2]);
        assert!(!lexicographical_compare(&exec, left.column(), right.column(), LessU32,).unwrap());
    }
}
