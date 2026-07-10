//! Arity-indexed device evaluators.

use core::marker::PhantomData;
use cubecl::prelude::*;

use crate::{
    op::{IndexedBinaryOp, IndexedUnaryOp, UnaryOp},
    reduce::ReductionOp,
    storage::{Decompose, Recompose, SelectLeaves},
};

/// A type-level device expression producing `Item`.
#[doc(hidden)]
pub trait DeviceExpr<Item: CubeType>: 'static + Send + Sync {}

/// How a staged leaf slot is interpreted.
#[doc(hidden)]
#[cubecl::cube]
pub trait ReadMode<T: CubePrimitive>: 'static + Send + Sync {
    fn read(slot: &[T], offset: u32, index: usize) -> T;
}

/// Reads one element at `offset + index`.
#[doc(hidden)]
pub struct Direct;

/// Reads the single staged value in a constant slot.
#[doc(hidden)]
pub struct Broadcast;

/// Reads a staged start and adds the logical index.
#[doc(hidden)]
pub struct Count;

/// Reads a staged final index and subtracts the logical index.
#[doc(hidden)]
pub struct ReverseCount;

#[cubecl::cube]
impl<T: CubePrimitive> ReadMode<T> for Direct {
    fn read(slot: &[T], offset: u32, index: usize) -> T {
        slot[offset as usize + index]
    }
}

#[cubecl::cube]
impl<T: CubePrimitive> ReadMode<T> for Broadcast {
    fn read(slot: &[T], _offset: u32, _index: usize) -> T {
        slot[0]
    }
}

#[cubecl::cube]
impl ReadMode<u32> for Count {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] + offset + index as u32
    }
}

#[cubecl::cube]
impl ReadMode<u32> for ReverseCount {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] - offset - index as u32
    }
}

macro_rules! define_slots {
    ($($slot:ident),+ $(,)?) => {
        $(
            #[doc(hidden)]
            pub struct $slot<T, Mode> {
                _marker: PhantomData<fn() -> (T, Mode)>,
            }

            impl<T, Mode> DeviceExpr<T> for $slot<T, Mode>
            where
                T: CubePrimitive + 'static,
                Mode: ReadMode<T>,
            {
            }
        )+
    };
}

define_slots!(Slot0, Slot1, Slot2, Slot3, Slot4, Slot5, Slot6, Slot7);

/// Device expression for a binary zip.
#[doc(hidden)]
pub struct ZipExpr<Left, Right> {
    _marker: PhantomData<fn() -> (Left, Right)>,
}

impl<Left, Right, LeftItem, RightItem> DeviceExpr<(LeftItem, RightItem)> for ZipExpr<Left, Right>
where
    LeftItem: CubeType + 'static,
    RightItem: CubeType + 'static,
    Left: DeviceExpr<LeftItem>,
    Right: DeviceExpr<RightItem>,
{
}

/// Device expression for a unary transform.
#[doc(hidden)]
pub struct TransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

/// Device expression for an index-aware unary transform.
#[doc(hidden)]
pub struct IndexedTransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

/// Device expression for an index-aware adjacent transform.
#[doc(hidden)]
pub struct AdjacentIndexedTransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output>
    for AdjacentIndexedTransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: IndexedBinaryOp<InputItem>,
{
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output>
    for IndexedTransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: IndexedUnaryOp<InputItem>,
{
}

/// Device expression for adjacent reduction, preserving the first item.
#[doc(hidden)]
pub struct AdjacentExpr<InputExpr, Item, Op, Layout, Leaves> {
    _marker: PhantomData<fn() -> (InputExpr, Item, Op, Layout, Leaves)>,
}

impl<InputExpr, Item, Op, Layout, Leaves> DeviceExpr<Item>
    for AdjacentExpr<InputExpr, Item, Op, Layout, Leaves>
where
    Item: CubeType + 'static,
    InputExpr: DeviceExpr<Item>,
    Op: ReductionOp<Item>,
    Leaves: CubeType + SelectLeaves + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
{
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output> for TransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: UnaryOp<InputItem>,
{
}

/// Device expression for `values[indices[index]]`.
#[doc(hidden)]
pub struct PermuteExpr<ValuesExpr, IndicesExpr> {
    _marker: PhantomData<fn() -> (ValuesExpr, IndicesExpr)>,
}

impl<ValuesExpr, IndicesExpr, Item> DeviceExpr<Item> for PermuteExpr<ValuesExpr, IndicesExpr>
where
    Item: CubeType + 'static,
    ValuesExpr: DeviceExpr<Item>,
    IndicesExpr: DeviceExpr<u32>,
{
}

#[doc(hidden)]
pub struct ReassociateExpr<InputExpr, InputItem, OutputItem, InputLayout, OutputLayout> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, OutputItem, InputLayout, OutputLayout)>,
}

impl<InputExpr, InputItem, OutputItem, InputLayout, OutputLayout> DeviceExpr<OutputItem>
    for ReassociateExpr<InputExpr, InputItem, OutputItem, InputLayout, OutputLayout>
where
    InputItem: CubeType + 'static,
    OutputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    InputLayout: Decompose<InputItem>,
    OutputLayout: Recompose<OutputItem, Leaves = InputLayout::Leaves>,
{
}

macro_rules! define_eval {
    ($trait_name:ident, $method:ident; $( $leaf:ident : $slot:ident ),+ $(,)?) => {
        #[doc = concat!("Evaluates a device expression using `", stringify!($trait_name), "` staged leaves.")]
        #[cubecl::cube]
        pub trait $trait_name<Item: CubeType, $( $leaf: CubePrimitive ),+>: DeviceExpr<Item> {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item;
        }

        #[cubecl::cube]
        impl<LeftItem, RightItem, LeftExpr, RightExpr, $( $leaf ),+>
            $trait_name<(LeftItem, RightItem), $( $leaf ),+> for ZipExpr<LeftExpr, RightExpr>
        where
            LeftItem: CubeType + 'static,
            RightItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            LeftExpr: $trait_name<LeftItem, $( $leaf ),+>,
            RightExpr: $trait_name<RightItem, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> (LeftItem, RightItem) {
                (
                    LeftExpr::$method($( $slot, )+ slot_offsets, index),
                    RightExpr::$method($( $slot, )+ slot_offsets, index),
                )
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+> for TransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: UnaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let input = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(input)
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+>
            for AdjacentIndexedTransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: IndexedBinaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let previous_index = if index == 0usize { 0usize } else { index - 1usize };
                let previous = InputExpr::$method(
                    $( $slot, )+
                    slot_offsets,
                    previous_index,
                );
                let current = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(previous, current, index as u32)
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+>
            for IndexedTransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: IndexedUnaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let input = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(input, index as u32)
            }
        }

        #[cubecl::cube]
        impl<Item, InputExpr, Op, Layout, Leaves, $( $leaf ),+>
            $trait_name<Item, $( $leaf ),+>
            for AdjacentExpr<InputExpr, Item, Op, Layout, Leaves>
        where
            Item: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
            Leaves: CubeType + SelectLeaves + 'static,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item {
                let previous_index = if index == 0usize { 0usize } else { index - 1usize };
                let first = Layout::decompose(
                    InputExpr::$method($( $slot, )+ slot_offsets, index),
                );
                let adjacent = Layout::decompose(Op::apply(
                    InputExpr::$method($( $slot, )+ slot_offsets, previous_index),
                    InputExpr::$method($( $slot, )+ slot_offsets, index),
                ));
                Layout::recompose(Leaves::select(index == 0usize, first, adjacent))
            }
        }

        #[cubecl::cube]
        impl<Item, ValuesExpr, IndicesExpr, $( $leaf ),+>
            $trait_name<Item, $( $leaf ),+> for PermuteExpr<ValuesExpr, IndicesExpr>
        where
            Item: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            ValuesExpr: $trait_name<Item, $( $leaf ),+>,
            IndicesExpr: $trait_name<u32, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item {
                let gathered = IndicesExpr::$method($( $slot, )+ slot_offsets, index);
                ValuesExpr::$method($( $slot, )+ slot_offsets, gathered as usize)
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, InputLayout, OutputLayout, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+>
            for ReassociateExpr<InputExpr, InputItem, OutputItem, InputLayout, OutputLayout>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            InputLayout: Decompose<InputItem>,
            OutputLayout: Recompose<OutputItem, Leaves = InputLayout::Leaves>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let input = InputExpr::$method($( $slot, )+ slot_offsets, index);
                OutputLayout::recompose(InputLayout::decompose(input))
            }
        }
    };
}

define_eval!(Eval1, eval1; L0: slot0);
define_eval!(Eval2, eval2; L0: slot0, L1: slot1);
define_eval!(Eval3, eval3; L0: slot0, L1: slot1, L2: slot2);
define_eval!(Eval4, eval4; L0: slot0, L1: slot1, L2: slot2, L3: slot3);
define_eval!(Eval5, eval5; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4);
define_eval!(Eval6, eval6; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5);
define_eval!(Eval7, eval7; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6);
define_eval!(Eval8, eval8; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7);

macro_rules! impl_slot_eval {
    (
        $trait_name:ident, $method:ident, $slot_expr:ident, $offset_index:literal;
        <$( $generic:ident ),+>;
        [$( $leaf_ty:ty ),+];
        [$( $slot:ident ),+];
        $selected:ident
    ) => {
        #[cubecl::cube]
        impl<Mode, $( $generic ),+> $trait_name<T, $( $leaf_ty ),+> for $slot_expr<T, Mode>
        where
            $( $generic: CubePrimitive, )+
            Mode: ReadMode<T>,
        {
            fn $method(
                $( $slot: &[$leaf_ty], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> T {
                let _ = ($( $slot, )+);
                Mode::read($selected, slot_offsets[$offset_index], index)
            }
        }
    };
}

impl_slot_eval!(Eval1, eval1, Slot0, 0; <T>; [T]; [slot0]; slot0);

impl_slot_eval!(Eval2, eval2, Slot0, 0; <T, L1>; [T, L1]; [slot0, slot1]; slot0);
impl_slot_eval!(Eval2, eval2, Slot1, 1; <L0, T>; [L0, T]; [slot0, slot1]; slot1);

impl_slot_eval!(Eval3, eval3, Slot0, 0; <T, L1, L2>; [T, L1, L2]; [slot0, slot1, slot2]; slot0);
impl_slot_eval!(Eval3, eval3, Slot1, 1; <L0, T, L2>; [L0, T, L2]; [slot0, slot1, slot2]; slot1);
impl_slot_eval!(Eval3, eval3, Slot2, 2; <L0, L1, T>; [L0, L1, T]; [slot0, slot1, slot2]; slot2);

impl_slot_eval!(Eval4, eval4, Slot0, 0; <T, L1, L2, L3>; [T, L1, L2, L3]; [slot0, slot1, slot2, slot3]; slot0);
impl_slot_eval!(Eval4, eval4, Slot1, 1; <L0, T, L2, L3>; [L0, T, L2, L3]; [slot0, slot1, slot2, slot3]; slot1);
impl_slot_eval!(Eval4, eval4, Slot2, 2; <L0, L1, T, L3>; [L0, L1, T, L3]; [slot0, slot1, slot2, slot3]; slot2);
impl_slot_eval!(Eval4, eval4, Slot3, 3; <L0, L1, L2, T>; [L0, L1, L2, T]; [slot0, slot1, slot2, slot3]; slot3);

impl_slot_eval!(Eval5, eval5, Slot0, 0; <T, L1, L2, L3, L4>; [T, L1, L2, L3, L4]; [slot0, slot1, slot2, slot3, slot4]; slot0);
impl_slot_eval!(Eval5, eval5, Slot1, 1; <L0, T, L2, L3, L4>; [L0, T, L2, L3, L4]; [slot0, slot1, slot2, slot3, slot4]; slot1);
impl_slot_eval!(Eval5, eval5, Slot2, 2; <L0, L1, T, L3, L4>; [L0, L1, T, L3, L4]; [slot0, slot1, slot2, slot3, slot4]; slot2);
impl_slot_eval!(Eval5, eval5, Slot3, 3; <L0, L1, L2, T, L4>; [L0, L1, L2, T, L4]; [slot0, slot1, slot2, slot3, slot4]; slot3);
impl_slot_eval!(Eval5, eval5, Slot4, 4; <L0, L1, L2, L3, T>; [L0, L1, L2, L3, T]; [slot0, slot1, slot2, slot3, slot4]; slot4);

impl_slot_eval!(Eval6, eval6, Slot0, 0; <T, L1, L2, L3, L4, L5>; [T, L1, L2, L3, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot0);
impl_slot_eval!(Eval6, eval6, Slot1, 1; <L0, T, L2, L3, L4, L5>; [L0, T, L2, L3, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot1);
impl_slot_eval!(Eval6, eval6, Slot2, 2; <L0, L1, T, L3, L4, L5>; [L0, L1, T, L3, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot2);
impl_slot_eval!(Eval6, eval6, Slot3, 3; <L0, L1, L2, T, L4, L5>; [L0, L1, L2, T, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot3);
impl_slot_eval!(Eval6, eval6, Slot4, 4; <L0, L1, L2, L3, T, L5>; [L0, L1, L2, L3, T, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot4);
impl_slot_eval!(Eval6, eval6, Slot5, 5; <L0, L1, L2, L3, L4, T>; [L0, L1, L2, L3, L4, T]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot5);

impl_slot_eval!(Eval7, eval7, Slot0, 0; <T, L1, L2, L3, L4, L5, L6>; [T, L1, L2, L3, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot0);
impl_slot_eval!(Eval7, eval7, Slot1, 1; <L0, T, L2, L3, L4, L5, L6>; [L0, T, L2, L3, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot1);
impl_slot_eval!(Eval7, eval7, Slot2, 2; <L0, L1, T, L3, L4, L5, L6>; [L0, L1, T, L3, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot2);
impl_slot_eval!(Eval7, eval7, Slot3, 3; <L0, L1, L2, T, L4, L5, L6>; [L0, L1, L2, T, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot3);
impl_slot_eval!(Eval7, eval7, Slot4, 4; <L0, L1, L2, L3, T, L5, L6>; [L0, L1, L2, L3, T, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot4);
impl_slot_eval!(Eval7, eval7, Slot5, 5; <L0, L1, L2, L3, L4, T, L6>; [L0, L1, L2, L3, L4, T, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot5);
impl_slot_eval!(Eval7, eval7, Slot6, 6; <L0, L1, L2, L3, L4, L5, T>; [L0, L1, L2, L3, L4, L5, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot6);

impl_slot_eval!(Eval8, eval8, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7>; [T, L1, L2, L3, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot0);
impl_slot_eval!(Eval8, eval8, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7>; [L0, T, L2, L3, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot1);
impl_slot_eval!(Eval8, eval8, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7>; [L0, L1, T, L3, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot2);
impl_slot_eval!(Eval8, eval8, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7>; [L0, L1, L2, T, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot3);
impl_slot_eval!(Eval8, eval8, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7>; [L0, L1, L2, L3, T, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot4);
impl_slot_eval!(Eval8, eval8, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7>; [L0, L1, L2, L3, L4, T, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot5);
impl_slot_eval!(Eval8, eval8, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7>; [L0, L1, L2, L3, L4, L5, T, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot6);
impl_slot_eval!(Eval8, eval8, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T>; [L0, L1, L2, L3, L4, L5, L6, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot7);
