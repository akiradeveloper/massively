use cubecl::prelude::*;
use std::marker::PhantomData;

/// Device expression leaf bound to slot 0.
pub struct Slot0<T> {
    _item: PhantomData<fn() -> T>,
}

/// Device expression leaf bound to slot 1.
pub struct Slot1<T> {
    _item: PhantomData<fn() -> T>,
}

/// Device expression leaf bound to slot 2.
pub struct Slot2<T> {
    _item: PhantomData<fn() -> T>,
}

/// Device expression leaf bound to slot 3.
pub struct Slot3<T> {
    _item: PhantomData<fn() -> T>,
}

/// Device expression leaf bound to slot 4.
pub struct Slot4<T> {
    _item: PhantomData<fn() -> T>,
}

/// Device expression leaf bound to slot 5.
pub struct Slot5<T> {
    _item: PhantomData<fn() -> T>,
}

/// Device expression leaf bound to slot 6.
pub struct Slot6<T> {
    _item: PhantomData<fn() -> T>,
}

/// Device expression leaf bound to slot 7.
pub struct Slot7<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 0.
pub struct ConstantSlot0<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 1.
pub struct ConstantSlot1<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 2.
pub struct ConstantSlot2<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 3.
pub struct ConstantSlot3<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 4.
pub struct ConstantSlot4<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 5.
pub struct ConstantSlot5<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 6.
pub struct ConstantSlot6<T> {
    _item: PhantomData<fn() -> T>,
}

/// Constant expression leaf bound to slot 7.
pub struct ConstantSlot7<T> {
    _item: PhantomData<fn() -> T>,
}

/// Counting expression leaf bound to slot 0.
pub struct CountingSlot0;

/// Counting expression leaf bound to slot 1.
pub struct CountingSlot1;

/// Counting expression leaf bound to slot 2.
pub struct CountingSlot2;

/// Counting expression leaf bound to slot 3.
pub struct CountingSlot3;

/// Counting expression leaf bound to slot 4.
pub struct CountingSlot4;

/// Counting expression leaf bound to slot 5.
pub struct CountingSlot5;

/// Counting expression leaf bound to slot 6.
pub struct CountingSlot6;

/// Counting expression leaf bound to slot 7.
pub struct CountingSlot7;

/// Logical expression that evaluates `values[indices[index]]`.
pub struct GatherExpr<ValueExpr, IndexExpr> {
    _expr: PhantomData<fn() -> (ValueExpr, IndexExpr)>,
}

/// Logical expression that applies a unary operation to another expression.
pub struct TransformExpr<InputExpr, InputItem, Op> {
    _expr: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

/// CubeCL expression tree that can evaluate one output element.
#[cube]
pub trait GpuExpr<T: CubePrimitive>: 'static + Send + Sync {
    /// Evaluates this expression for `index`.
    fn eval(input: &[T], indices: &[u32], rhs: &[T], rhs_indices: &[u32], index: usize) -> T;
}

/// CubeCL expression tree over up to four staged device slots.
#[cube]
pub trait DeviceGpuExpr<T: CubePrimitive>: 'static + Send + Sync {
    /// Evaluates this expression for `index`.
    fn eval(
        slot0: &[T],
        slot1: &[T],
        slot2: &[T],
        slot3: &[T],
        slot_offsets: &[u32],
        index: usize,
    ) -> T;
}

/// Type-level expression shape for a logical CubeCL value.
///
/// Unlike [`DeviceGpuExpr`], this trait is not limited to `CubePrimitive`.
/// It records that an expression tree evaluates to the same logical shape as
/// an `MIter::Item`, including nested tuples. Kernel launch support can then
/// be built on top of this shape without flattening the semantic item.
pub trait LogicalDeviceExpr<T: CubeType>: 'static + Send + Sync {}

impl<T> LogicalDeviceExpr<T> for Slot0<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for Slot1<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for Slot2<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for Slot3<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for Slot4<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for Slot5<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for Slot6<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for Slot7<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot0<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot1<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot2<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot3<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot4<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot5<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot6<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot7<T> where T: CubePrimitive + 'static {}
impl LogicalDeviceExpr<u32> for CountingSlot0 {}
impl LogicalDeviceExpr<u32> for CountingSlot1 {}
impl LogicalDeviceExpr<u32> for CountingSlot2 {}
impl LogicalDeviceExpr<u32> for CountingSlot3 {}
impl LogicalDeviceExpr<u32> for CountingSlot4 {}
impl LogicalDeviceExpr<u32> for CountingSlot5 {}
impl LogicalDeviceExpr<u32> for CountingSlot6 {}
impl LogicalDeviceExpr<u32> for CountingSlot7 {}

impl<Item, ValueExpr, IndexExpr> LogicalDeviceExpr<Item> for GatherExpr<ValueExpr, IndexExpr>
where
    Item: CubeType + 'static,
    ValueExpr: LogicalDeviceExpr<Item>,
    IndexExpr: LogicalDeviceExpr<u32>,
{
}

impl<InputItem, OutputItem, InputExpr, Op> LogicalDeviceExpr<OutputItem>
    for TransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    OutputItem: CubeType + 'static,
    InputExpr: LogicalDeviceExpr<InputItem>,
    Op: crate::detail::op::kernel::UnaryOp<InputItem, Output = OutputItem>,
{
}

macro_rules! impl_logical_device_expr_tuple {
    ($( $expr:ident : $item:ident ),+) => {
        impl<$( $expr, $item ),+> LogicalDeviceExpr<($( $item, )+)> for ($( $expr, )+)
        where
            $( $item: CubeType + 'static, )+
            $( $expr: LogicalDeviceExpr<$item>, )+
        {
        }
    };
}

impl_logical_device_expr_tuple!(AExpr: A);
impl_logical_device_expr_tuple!(AExpr: A, BExpr: B);
impl_logical_device_expr_tuple!(AExpr: A, BExpr: B, CExpr: C);
impl_logical_device_expr_tuple!(AExpr: A, BExpr: B, CExpr: C, DExpr: D);
impl_logical_device_expr_tuple!(AExpr: A, BExpr: B, CExpr: C, DExpr: D, EExpr: E);
impl_logical_device_expr_tuple!(AExpr: A, BExpr: B, CExpr: C, DExpr: D, EExpr: E, FExpr: F);
impl_logical_device_expr_tuple!(AExpr: A, BExpr: B, CExpr: C, DExpr: D, EExpr: E, FExpr: F, GExpr: G);
impl_logical_device_expr_tuple!(
    AExpr: A,
    BExpr: B,
    CExpr: C,
    DExpr: D,
    EExpr: E,
    FExpr: F,
    GExpr: G,
    HExpr: H
);

/// Executable logical expression over the first three physical leaf slots.
#[cube]
pub trait LogicalDeviceExpr3<Item: CubeType, A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>:
    LogicalDeviceExpr<Item>
{
    /// Evaluates one logical item from three staged storage leaves.
    fn eval3(slot0: &[A], slot1: &[B], slot2: &[C], slot_offsets: &[u32], index: usize) -> Item;
}

#[cube]
impl<T: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDeviceExpr3<T, T, B, C>
    for Slot0<T>
{
    fn eval3(slot0: &[T], _slot1: &[B], _slot2: &[C], slot_offsets: &[u32], index: usize) -> T {
        slot0[slot_offsets[0] as usize + index]
    }
}

#[cube]
impl<A: CubePrimitive, T: CubePrimitive, C: CubePrimitive> LogicalDeviceExpr3<T, A, T, C>
    for Slot1<T>
{
    fn eval3(_slot0: &[A], slot1: &[T], _slot2: &[C], slot_offsets: &[u32], index: usize) -> T {
        slot1[slot_offsets[1] as usize + index]
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, T: CubePrimitive> LogicalDeviceExpr3<T, A, B, T>
    for Slot2<T>
{
    fn eval3(_slot0: &[A], _slot1: &[B], slot2: &[T], slot_offsets: &[u32], index: usize) -> T {
        slot2[slot_offsets[2] as usize + index]
    }
}

macro_rules! impl_logical_device_expr3_tuple {
    ($( $expr:ident : $item:ident ),+) => {
        #[cube]
        impl<LeafA, LeafB, LeafC, $( $expr, $item ),+>
            LogicalDeviceExpr3<($( $item, )+), LeafA, LeafB, LeafC> for ($( $expr, )+)
        where
            LeafA: CubePrimitive,
            LeafB: CubePrimitive,
            LeafC: CubePrimitive,
            $( $item: CubeType + 'static, )+
            $( $expr: LogicalDeviceExpr3<$item, LeafA, LeafB, LeafC>, )+
        {
            fn eval3(
                slot0: &[LeafA],
                slot1: &[LeafB],
                slot2: &[LeafC],
                slot_offsets: &[u32],
                index: usize,
            ) -> ($( $item, )+) {
                ($( $expr::eval3(slot0, slot1, slot2, slot_offsets, index), )+)
            }
        }
    };
}

impl_logical_device_expr3_tuple!(AExpr: A);
impl_logical_device_expr3_tuple!(AExpr: A, BExpr: B);
impl_logical_device_expr3_tuple!(AExpr: A, BExpr: B, CExpr: C);

/// Executable logical expression over up to eight physical leaf slots.
#[cube]
pub trait LogicalDeviceExpr7<
    Item: CubeType,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    H: CubePrimitive = G,
>: LogicalDeviceExpr<Item>
{
    /// Evaluates one logical item from eight staged storage leaves.
    #[allow(clippy::too_many_arguments)]
    fn eval7(
        slot0: &[A],
        slot1: &[B],
        slot2: &[C],
        slot3: &[D],
        slot4: &[E],
        slot5: &[F],
        slot6: &[G],
        slot7: &[H],
        slot_offsets: &[u32],
        index: usize,
    ) -> Item;
}

macro_rules! impl_logical_device_expr7_slot {
    ($slot_ty:ident, $index:literal; <$( $gen:ident ),+>; $a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty, $h:ty; $s0:ident, $s1:ident, $s2:ident, $s3:ident, $s4:ident, $s5:ident, $s6:ident, $s7:ident; $value:ident) => {
        #[cube]
        impl<$( $gen ),+> LogicalDeviceExpr7<T, $a, $b, $c, $d, $e, $f, $g, $h>
            for $slot_ty<T>
        where
            $( $gen: CubePrimitive, )+
            T: CubePrimitive,
        {
            fn eval7(
                $s0: &[$a],
                $s1: &[$b],
                $s2: &[$c],
                $s3: &[$d],
                $s4: &[$e],
                $s5: &[$f],
                $s6: &[$g],
                $s7: &[$h],
                slot_offsets: &[u32],
                index: usize,
            ) -> T {
                let _ = ($s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7);
                $value[slot_offsets[$index] as usize + index]
            }
        }
    };
}

impl_logical_device_expr7_slot!(Slot0, 0; <T, B, C, D, E, F, G, H>; T, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot0);
impl_logical_device_expr7_slot!(Slot1, 1; <A, T, C, D, E, F, G, H>; A, T, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot1);
impl_logical_device_expr7_slot!(Slot2, 2; <A, B, T, D, E, F, G, H>; A, B, T, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot2);
impl_logical_device_expr7_slot!(Slot3, 3; <A, B, C, T, E, F, G, H>; A, B, C, T, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot3);
impl_logical_device_expr7_slot!(Slot4, 4; <A, B, C, D, T, F, G, H>; A, B, C, D, T, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot4);
impl_logical_device_expr7_slot!(Slot5, 5; <A, B, C, D, E, T, G, H>; A, B, C, D, E, T, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot5);
impl_logical_device_expr7_slot!(Slot6, 6; <A, B, C, D, E, F, T, H>; A, B, C, D, E, F, T, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot6);
impl_logical_device_expr7_slot!(Slot7, 7; <A, B, C, D, E, F, G, T>; A, B, C, D, E, F, G, T; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot7);

macro_rules! impl_logical_device_expr7_tuple {
    ($( $expr:ident : $item:ident ),+) => {
        #[cube]
        impl<Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, $( $expr, $item ),+>
            LogicalDeviceExpr7<($( $item, )+), Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>
            for ($( $expr, )+)
        where
            Leaf0: CubePrimitive,
            Leaf1: CubePrimitive,
            Leaf2: CubePrimitive,
            Leaf3: CubePrimitive,
            Leaf4: CubePrimitive,
            Leaf5: CubePrimitive,
            Leaf6: CubePrimitive,
            Leaf7: CubePrimitive,
            $( $item: CubeType + 'static, )+
            $( $expr: LogicalDeviceExpr7<$item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>, )+
        {
            fn eval7(
                slot0: &[Leaf0],
                slot1: &[Leaf1],
                slot2: &[Leaf2],
                slot3: &[Leaf3],
                slot4: &[Leaf4],
                slot5: &[Leaf5],
                slot6: &[Leaf6],
                slot7: &[Leaf7],
                slot_offsets: &[u32],
                index: usize,
            ) -> ($( $item, )+) {
                ($(
                    $expr::eval7(
                        slot0,
                        slot1,
                        slot2,
                        slot3,
                        slot4,
                        slot5,
                        slot6,
                        slot7,
                        slot_offsets,
                        index,
                    ),
                )+)
            }
        }
    };
}

impl_logical_device_expr7_tuple!(AExpr: A);
impl_logical_device_expr7_tuple!(AExpr: A, BExpr: B);
impl_logical_device_expr7_tuple!(AExpr: A, BExpr: B, CExpr: C);
impl_logical_device_expr7_tuple!(AExpr: A, BExpr: B, CExpr: C, DExpr: D);
impl_logical_device_expr7_tuple!(AExpr: A, BExpr: B, CExpr: C, DExpr: D, EExpr: E);
impl_logical_device_expr7_tuple!(AExpr: A, BExpr: B, CExpr: C, DExpr: D, EExpr: E, FExpr: F);
impl_logical_device_expr7_tuple!(
    AExpr: A,
    BExpr: B,
    CExpr: C,
    DExpr: D,
    EExpr: E,
    FExpr: F,
    GExpr: G
);
impl_logical_device_expr7_tuple!(
    AExpr: A,
    BExpr: B,
    CExpr: C,
    DExpr: D,
    EExpr: E,
    FExpr: F,
    GExpr: G,
    HExpr: H
);

macro_rules! impl_constant_logical_device_expr7 {
    ($expr:ident, $item:ident; <$( $gen:ident ),+>; $a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty, $h:ty; $s0:ident, $s1:ident, $s2:ident, $s3:ident, $s4:ident, $s5:ident, $s6:ident, $s7:ident; $value:expr) => {
        #[cube]
        impl<$( $gen ),+> LogicalDeviceExpr7<$item, $a, $b, $c, $d, $e, $f, $g, $h>
            for $expr<$item>
        where
            $item: CubePrimitive,
            $( $gen: CubePrimitive, )+
        {
            fn eval7(
                $s0: &[$a],
                $s1: &[$b],
                $s2: &[$c],
                $s3: &[$d],
                $s4: &[$e],
                $s5: &[$f],
                $s6: &[$g],
                $s7: &[$h],
                _slot_offsets: &[u32],
                _index: usize,
            ) -> $item {
                let _ = ($s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7);
                $value
            }
        }
    };
}

impl_constant_logical_device_expr7!(ConstantSlot0, A; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot0[0]);
impl_constant_logical_device_expr7!(ConstantSlot1, B; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot1[0]);
impl_constant_logical_device_expr7!(ConstantSlot2, C; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot2[0]);
impl_constant_logical_device_expr7!(ConstantSlot3, D; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot3[0]);
impl_constant_logical_device_expr7!(ConstantSlot4, E; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot4[0]);
impl_constant_logical_device_expr7!(ConstantSlot5, F; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot5[0]);
impl_constant_logical_device_expr7!(ConstantSlot6, G; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot6[0]);
impl_constant_logical_device_expr7!(ConstantSlot7, H; <A, B, C, D, E, F, G, H>; A, B, C, D, E, F, G, H; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot7[0]);

macro_rules! impl_counting_logical_device_expr7 {
    ($expr:ident; <$( $gen:ident ),*>; $a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty, $h:ty; $index:literal; $s0:ident, $s1:ident, $s2:ident, $s3:ident, $s4:ident, $s5:ident, $s6:ident, $s7:ident; $start:expr) => {
        #[cube]
        impl<$( $gen ),*> LogicalDeviceExpr7<u32, $a, $b, $c, $d, $e, $f, $g, $h> for $expr
        where
            $( $gen: CubePrimitive, )*
        {
            fn eval7(
                $s0: &[$a],
                $s1: &[$b],
                $s2: &[$c],
                $s3: &[$d],
                $s4: &[$e],
                $s5: &[$f],
                $s6: &[$g],
                $s7: &[$h],
                slot_offsets: &[u32],
                index: usize,
            ) -> u32 {
                let _ = ($s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7);
                $start + slot_offsets[$index] + index as u32
            }
        }
    };
}

impl_counting_logical_device_expr7!(CountingSlot0; <B, C, D, E, F, G, H>; u32, B, C, D, E, F, G, H; 0; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot0[0]);
impl_counting_logical_device_expr7!(CountingSlot1; <A, C, D, E, F, G, H>; A, u32, C, D, E, F, G, H; 1; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot1[0]);
impl_counting_logical_device_expr7!(CountingSlot2; <A, B, D, E, F, G, H>; A, B, u32, D, E, F, G, H; 2; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot2[0]);
impl_counting_logical_device_expr7!(CountingSlot3; <A, B, C, E, F, G, H>; A, B, C, u32, E, F, G, H; 3; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot3[0]);
impl_counting_logical_device_expr7!(CountingSlot4; <A, B, C, D, F, G, H>; A, B, C, D, u32, F, G, H; 4; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot4[0]);
impl_counting_logical_device_expr7!(CountingSlot5; <A, B, C, D, E, G, H>; A, B, C, D, E, u32, G, H; 5; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot5[0]);
impl_counting_logical_device_expr7!(CountingSlot6; <A, B, C, D, E, F, H>; A, B, C, D, E, F, u32, H; 6; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot6[0]);
impl_counting_logical_device_expr7!(CountingSlot7; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G, u32; 7; slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7; slot7[0]);

#[cube]
impl<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, ValueExpr, IndexExpr>
    LogicalDeviceExpr7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>
    for GatherExpr<ValueExpr, IndexExpr>
where
    Item: CubeType + 'static,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Leaf7: CubePrimitive,
    ValueExpr: LogicalDeviceExpr7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
    IndexExpr: LogicalDeviceExpr7<u32, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
{
    fn eval7(
        slot0: &[Leaf0],
        slot1: &[Leaf1],
        slot2: &[Leaf2],
        slot3: &[Leaf3],
        slot4: &[Leaf4],
        slot5: &[Leaf5],
        slot6: &[Leaf6],
        slot7: &[Leaf7],
        slot_offsets: &[u32],
        index: usize,
    ) -> Item {
        let gathered = IndexExpr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot7,
            slot_offsets,
            index,
        );
        ValueExpr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot7,
            slot_offsets,
            gathered as usize,
        )
    }
}

#[cube]
impl<InputItem, OutputItem, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, InputExpr, Op>
    LogicalDeviceExpr7<OutputItem, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>
    for TransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    OutputItem: CubeType + 'static,
    Leaf0: CubePrimitive,
    Leaf1: CubePrimitive,
    Leaf2: CubePrimitive,
    Leaf3: CubePrimitive,
    Leaf4: CubePrimitive,
    Leaf5: CubePrimitive,
    Leaf6: CubePrimitive,
    Leaf7: CubePrimitive,
    InputExpr:
        LogicalDeviceExpr7<InputItem, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
    Op: crate::detail::op::kernel::UnaryOp<InputItem, Output = OutputItem>,
{
    fn eval7(
        slot0: &[Leaf0],
        slot1: &[Leaf1],
        slot2: &[Leaf2],
        slot3: &[Leaf3],
        slot4: &[Leaf4],
        slot5: &[Leaf5],
        slot6: &[Leaf6],
        slot7: &[Leaf7],
        slot_offsets: &[u32],
        index: usize,
    ) -> OutputItem {
        let input = InputExpr::eval7(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot7,
            slot_offsets,
            index,
        );
        Op::apply(input)
    }
}

/// Type-level leaf set needed to execute a logical expression through the
/// current three-slot logical transform kernel.
pub trait LogicalDeviceExpr3Shape<Item: CubeType>: LogicalDeviceExpr<Item> {
    type LeafA: CubePrimitive;
    type LeafB: CubePrimitive;
    type LeafC: CubePrimitive;
}

#[cube]
pub trait LogicalDevicePack3<Item: CubeType, A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> {
    fn pack(a: A, b: B, c: C) -> Item;
    fn unpack(item: Item) -> (A, B, C);
}

pub trait LogicalHostPack3<Item, A, B, C> {
    fn pack_host(a: A, b: B, c: C) -> Item;
    fn leaves_host(item: Item) -> (A, B, C);
}

#[cube]
pub trait LogicalDevicePack7<
    Item: CubeType,
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    H: CubePrimitive = G,
>
{
    fn pack(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H) -> Item;
    fn unpack(item: Item) -> (A, B, C, D, E, F, G, H);
}

pub trait LogicalHostPack7<Item, A, B, C, D, E, F, G, H = G> {
    fn pack_host(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H) -> Item;
    fn leaves_host(item: Item) -> (A, B, C, D, E, F, G, H);
}

#[cube]
impl<A: CubePrimitive> LogicalDevicePack3<A, A, A, A> for Slot0<A> {
    fn pack(a: A, _b: A, _c: A) -> A {
        a
    }

    fn unpack(item: A) -> (A, A, A) {
        (item, item, item)
    }
}

impl<A: CubePrimitive> LogicalHostPack3<A, A, A, A> for Slot0<A> {
    fn pack_host(a: A, _b: A, _c: A) -> A {
        a
    }

    fn leaves_host(item: A) -> (A, A, A) {
        (item, item, item)
    }
}

impl<A: CubePrimitive> LogicalDeviceExpr3Shape<A> for Slot0<A> {
    type LeafA = A;
    type LeafB = A;
    type LeafC = A;
}

#[cube]
impl<A: CubePrimitive> LogicalDevicePack7<A, A, A, A, A, A, A, A, A> for Slot0<A> {
    fn pack(a: A, _b: A, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> A {
        a
    }

    fn unpack(item: A) -> (A, A, A, A, A, A, A, A) {
        (item, item, item, item, item, item, item, item)
    }
}

impl<A: CubePrimitive> LogicalHostPack7<A, A, A, A, A, A, A, A, A> for Slot0<A> {
    fn pack_host(a: A, _b: A, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> A {
        a
    }

    fn leaves_host(item: A) -> (A, A, A, A, A, A, A, A) {
        (item, item, item, item, item, item, item, item)
    }
}

#[cube]
impl<A: CubePrimitive> LogicalDevicePack3<(A,), A, A, A> for (Slot0<A>,) {
    fn pack(a: A, _b: A, _c: A) -> (A,) {
        (a,)
    }

    fn unpack(item: (A,)) -> (A, A, A) {
        (item.0, item.0, item.0)
    }
}

impl<A: CubePrimitive> LogicalHostPack3<(A,), A, A, A> for (Slot0<A>,) {
    fn pack_host(a: A, _b: A, _c: A) -> (A,) {
        (a,)
    }

    fn leaves_host(item: (A,)) -> (A, A, A) {
        (item.0, item.0, item.0)
    }
}

impl<A: CubePrimitive> LogicalDeviceExpr3Shape<(A,)> for (Slot0<A>,) {
    type LeafA = A;
    type LeafB = A;
    type LeafC = A;
}

#[cube]
impl<A: CubePrimitive> LogicalDevicePack7<(A,), A, A, A, A, A, A, A, A> for (Slot0<A>,) {
    fn pack(a: A, _b: A, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A,) {
        (a,)
    }

    fn unpack(item: (A,)) -> (A, A, A, A, A, A, A, A) {
        (
            item.0, item.0, item.0, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

impl<A: CubePrimitive> LogicalHostPack7<(A,), A, A, A, A, A, A, A, A> for (Slot0<A>,) {
    fn pack_host(a: A, _b: A, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A,) {
        (a,)
    }

    fn leaves_host(item: (A,)) -> (A, A, A, A, A, A, A, A) {
        (
            item.0, item.0, item.0, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive> LogicalDevicePack3<(A, B), A, B, A>
    for (Slot0<A>, Slot1<B>)
{
    fn pack(a: A, b: B, _c: A) -> (A, B) {
        (a, b)
    }

    fn unpack(item: (A, B)) -> (A, B, A) {
        (item.0, item.1, item.0)
    }
}

impl<A: CubePrimitive, B: CubePrimitive> LogicalHostPack3<(A, B), A, B, A>
    for (Slot0<A>, Slot1<B>)
{
    fn pack_host(a: A, b: B, _c: A) -> (A, B) {
        (a, b)
    }

    fn leaves_host(item: (A, B)) -> (A, B, A) {
        (item.0, item.1, item.0)
    }
}

impl<A: CubePrimitive, B: CubePrimitive> LogicalDeviceExpr3Shape<(A, B)> for (Slot0<A>, Slot1<B>) {
    type LeafA = A;
    type LeafB = B;
    type LeafC = A;
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive> LogicalDevicePack7<(A, B), A, B, A, A, A, A, A, A>
    for (Slot0<A>, Slot1<B>)
{
    fn pack(a: A, b: B, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A, B) {
        (a, b)
    }

    fn unpack(item: (A, B)) -> (A, B, A, A, A, A, A, A) {
        (
            item.0, item.1, item.0, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

impl<A: CubePrimitive, B: CubePrimitive> LogicalHostPack7<(A, B), A, B, A, A, A, A, A, A>
    for (Slot0<A>, Slot1<B>)
{
    fn pack_host(a: A, b: B, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A, B) {
        (a, b)
    }

    fn leaves_host(item: (A, B)) -> (A, B, A, A, A, A, A, A) {
        (
            item.0, item.1, item.0, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDevicePack3<(A, B, C), A, B, C>
    for (Slot0<A>, Slot1<B>, Slot2<C>)
{
    fn pack(a: A, b: B, c: C) -> (A, B, C) {
        (a, b, c)
    }

    fn unpack(item: (A, B, C)) -> (A, B, C) {
        item
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalHostPack3<(A, B, C), A, B, C>
    for (Slot0<A>, Slot1<B>, Slot2<C>)
{
    fn pack_host(a: A, b: B, c: C) -> (A, B, C) {
        (a, b, c)
    }

    fn leaves_host(item: (A, B, C)) -> (A, B, C) {
        item
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDeviceExpr3Shape<(A, B, C)>
    for (Slot0<A>, Slot1<B>, Slot2<C>)
{
    type LeafA = A;
    type LeafB = B;
    type LeafC = C;
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>
    LogicalDevicePack7<(A, B, C), A, B, C, A, A, A, A, A> for (Slot0<A>, Slot1<B>, Slot2<C>)
{
    fn pack(a: A, b: B, c: C, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A, B, C) {
        (a, b, c)
    }

    fn unpack(item: (A, B, C)) -> (A, B, C, A, A, A, A, A) {
        (
            item.0, item.1, item.2, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>
    LogicalHostPack7<(A, B, C), A, B, C, A, A, A, A, A> for (Slot0<A>, Slot1<B>, Slot2<C>)
{
    fn pack_host(a: A, b: B, c: C, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A, B, C) {
        (a, b, c)
    }

    fn leaves_host(item: (A, B, C)) -> (A, B, C, A, A, A, A, A) {
        (
            item.0, item.1, item.2, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive> LogicalDevicePack3<((A,), B), A, B, A>
    for ((Slot0<A>,), Slot1<B>)
{
    fn pack(a: A, b: B, _c: A) -> ((A,), B) {
        ((a,), b)
    }

    fn unpack(item: ((A,), B)) -> (A, B, A) {
        (item.0.0, item.1, item.0.0)
    }
}

impl<A: CubePrimitive, B: CubePrimitive> LogicalHostPack3<((A,), B), A, B, A>
    for ((Slot0<A>,), Slot1<B>)
{
    fn pack_host(a: A, b: B, _c: A) -> ((A,), B) {
        ((a,), b)
    }

    fn leaves_host(item: ((A,), B)) -> (A, B, A) {
        (item.0.0, item.1, item.0.0)
    }
}

impl<A: CubePrimitive, B: CubePrimitive> LogicalDeviceExpr3Shape<((A,), B)>
    for ((Slot0<A>,), Slot1<B>)
{
    type LeafA = A;
    type LeafB = B;
    type LeafC = A;
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive> LogicalDevicePack7<((A,), B), A, B, A, A, A, A, A, A>
    for ((Slot0<A>,), Slot1<B>)
{
    fn pack(a: A, b: B, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> ((A,), B) {
        ((a,), b)
    }

    fn unpack(item: ((A,), B)) -> (A, B, A, A, A, A, A, A) {
        (
            item.0.0, item.1, item.0.0, item.0.0, item.0.0, item.0.0, item.0.0, item.0.0,
        )
    }
}

impl<A: CubePrimitive, B: CubePrimitive> LogicalHostPack7<((A,), B), A, B, A, A, A, A, A, A>
    for ((Slot0<A>,), Slot1<B>)
{
    fn pack_host(a: A, b: B, _c: A, _d: A, _e: A, _f: A, _g: A, _h: A) -> ((A,), B) {
        ((a,), b)
    }

    fn leaves_host(item: ((A,), B)) -> (A, B, A, A, A, A, A, A) {
        (
            item.0.0, item.1, item.0.0, item.0.0, item.0.0, item.0.0, item.0.0, item.0.0,
        )
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDevicePack3<((A, B), C), A, B, C>
    for ((Slot0<A>, Slot1<B>), Slot2<C>)
{
    fn pack(a: A, b: B, c: C) -> ((A, B), C) {
        ((a, b), c)
    }

    fn unpack(item: ((A, B), C)) -> (A, B, C) {
        (item.0.0, item.0.1, item.1)
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalHostPack3<((A, B), C), A, B, C>
    for ((Slot0<A>, Slot1<B>), Slot2<C>)
{
    fn pack_host(a: A, b: B, c: C) -> ((A, B), C) {
        ((a, b), c)
    }

    fn leaves_host(item: ((A, B), C)) -> (A, B, C) {
        (item.0.0, item.0.1, item.1)
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDeviceExpr3Shape<((A, B), C)>
    for ((Slot0<A>, Slot1<B>), Slot2<C>)
{
    type LeafA = A;
    type LeafB = B;
    type LeafC = C;
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>
    LogicalDevicePack7<((A, B), C), A, B, C, A, A, A, A, A> for ((Slot0<A>, Slot1<B>), Slot2<C>)
{
    fn pack(a: A, b: B, c: C, _d: A, _e: A, _f: A, _g: A, _h: A) -> ((A, B), C) {
        ((a, b), c)
    }

    fn unpack(item: ((A, B), C)) -> (A, B, C, A, A, A, A, A) {
        (
            item.0.0, item.0.1, item.1, item.0.0, item.0.0, item.0.0, item.0.0, item.0.0,
        )
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>
    LogicalHostPack7<((A, B), C), A, B, C, A, A, A, A, A> for ((Slot0<A>, Slot1<B>), Slot2<C>)
{
    fn pack_host(a: A, b: B, c: C, _d: A, _e: A, _f: A, _g: A, _h: A) -> ((A, B), C) {
        ((a, b), c)
    }

    fn leaves_host(item: ((A, B), C)) -> (A, B, C, A, A, A, A, A) {
        (
            item.0.0, item.0.1, item.1, item.0.0, item.0.0, item.0.0, item.0.0, item.0.0,
        )
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDevicePack3<(A, (B, C)), A, B, C>
    for (Slot0<A>, (Slot1<B>, Slot2<C>))
{
    fn pack(a: A, b: B, c: C) -> (A, (B, C)) {
        (a, (b, c))
    }

    fn unpack(item: (A, (B, C))) -> (A, B, C) {
        (item.0, item.1.0, item.1.1)
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalHostPack3<(A, (B, C)), A, B, C>
    for (Slot0<A>, (Slot1<B>, Slot2<C>))
{
    fn pack_host(a: A, b: B, c: C) -> (A, (B, C)) {
        (a, (b, c))
    }

    fn leaves_host(item: (A, (B, C))) -> (A, B, C) {
        (item.0, item.1.0, item.1.1)
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDeviceExpr3Shape<(A, (B, C))>
    for (Slot0<A>, (Slot1<B>, Slot2<C>))
{
    type LeafA = A;
    type LeafB = B;
    type LeafC = C;
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>
    LogicalDevicePack7<(A, (B, C)), A, B, C, A, A, A, A, A> for (Slot0<A>, (Slot1<B>, Slot2<C>))
{
    fn pack(a: A, b: B, c: C, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A, (B, C)) {
        (a, (b, c))
    }

    fn unpack(item: (A, (B, C))) -> (A, B, C, A, A, A, A, A) {
        (
            item.0, item.1.0, item.1.1, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>
    LogicalHostPack7<(A, (B, C)), A, B, C, A, A, A, A, A> for (Slot0<A>, (Slot1<B>, Slot2<C>))
{
    fn pack_host(a: A, b: B, c: C, _d: A, _e: A, _f: A, _g: A, _h: A) -> (A, (B, C)) {
        (a, (b, c))
    }

    fn leaves_host(item: (A, (B, C))) -> (A, B, C, A, A, A, A, A) {
        (
            item.0, item.1.0, item.1.1, item.0, item.0, item.0, item.0, item.0,
        )
    }
}

/// Type-level leaf set for executing a logical expression through seven slots.
pub trait LogicalDeviceExpr7Shape<Item: CubeType>: LogicalDeviceExpr<Item> {
    type Leaf0: CubePrimitive;
    type Leaf1: CubePrimitive;
    type Leaf2: CubePrimitive;
    type Leaf3: CubePrimitive;
    type Leaf4: CubePrimitive;
    type Leaf5: CubePrimitive;
    type Leaf6: CubePrimitive;
    type Leaf7: CubePrimitive;
}

impl<A: CubePrimitive> LogicalDeviceExpr7Shape<A> for Slot0<A> {
    type Leaf0 = A;
    type Leaf1 = A;
    type Leaf2 = A;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
    type Leaf7 = A;
}

impl<A: CubePrimitive> LogicalDeviceExpr7Shape<(A,)> for (Slot0<A>,) {
    type Leaf0 = A;
    type Leaf1 = A;
    type Leaf2 = A;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
    type Leaf7 = A;
}

macro_rules! impl_logical_device_expr7_shape_flat {
    (
        impl < $( $gen:ident ),+ >;
        $item:ty,
        $expr:ty;
        $leaf0:ty,
        $leaf1:ty,
        $leaf2:ty,
        $leaf3:ty,
        $leaf4:ty,
        $leaf5:ty,
        $leaf6:ty,
        $leaf7:ty
    ) => {
        impl<$( $gen: CubePrimitive ),+> LogicalDeviceExpr7Shape<$item> for $expr {
            type Leaf0 = $leaf0;
            type Leaf1 = $leaf1;
            type Leaf2 = $leaf2;
            type Leaf3 = $leaf3;
            type Leaf4 = $leaf4;
            type Leaf5 = $leaf5;
            type Leaf6 = $leaf6;
            type Leaf7 = $leaf7;
        }
    };
}

impl_logical_device_expr7_shape_flat!(
    impl<A, B>;
    (A, B),
    (Slot0<A>, Slot1<B>);
    A,
    B,
    A,
    A,
    A,
    A,
    A,
    A
);
impl_logical_device_expr7_shape_flat!(
    impl<A, B, C>;
    (A, B, C),
    (Slot0<A>, Slot1<B>, Slot2<C>);
    A,
    B,
    C,
    A,
    A,
    A,
    A,
    A
);
impl_logical_device_expr7_shape_flat!(
    impl<A, B, C, D>;
    (A, B, C, D),
    (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>);
    A,
    B,
    C,
    D,
    A,
    A,
    A,
    A
);
impl_logical_device_expr7_shape_flat!(
    impl<A, B, C, D, E>;
    (A, B, C, D, E),
    (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>);
    A,
    B,
    C,
    D,
    E,
    A,
    A,
    A
);
impl_logical_device_expr7_shape_flat!(
    impl<A, B, C, D, E, F>;
    (A, B, C, D, E, F),
    (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>, Slot5<F>);
    A,
    B,
    C,
    D,
    E,
    F,
    A,
    A
);
impl_logical_device_expr7_shape_flat!(
    impl<A, B, C, D, E, F, G>;
    (A, B, C, D, E, F, G),
    (
        Slot0<A>,
        Slot1<B>,
        Slot2<C>,
        Slot3<D>,
        Slot4<E>,
        Slot5<F>,
        Slot6<G>
    );
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    G
);

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>
    LogicalDevicePack7<(A, B, C, D), A, B, C, D, A, A, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>)
{
    fn pack(a: A, b: B, c: C, d: D, _e: A, _f: A, _g: A, _h: A) -> (A, B, C, D) {
        (a, b, c, d)
    }

    fn unpack(item: (A, B, C, D)) -> (A, B, C, D, A, A, A, A) {
        (
            item.0, item.1, item.2, item.3, item.0, item.0, item.0, item.0,
        )
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>
    LogicalHostPack7<(A, B, C, D), A, B, C, D, A, A, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>)
{
    fn pack_host(a: A, b: B, c: C, d: D, _e: A, _f: A, _g: A, _h: A) -> (A, B, C, D) {
        (a, b, c, d)
    }

    fn leaves_host(item: (A, B, C, D)) -> (A, B, C, D, A, A, A, A) {
        (
            item.0, item.1, item.2, item.3, item.0, item.0, item.0, item.0,
        )
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive, E: CubePrimitive>
    LogicalDevicePack7<(A, B, C, D, E), A, B, C, D, E, A, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>)
{
    fn pack(a: A, b: B, c: C, d: D, e: E, _f: A, _g: A, _h: A) -> (A, B, C, D, E) {
        (a, b, c, d, e)
    }

    fn unpack(item: (A, B, C, D, E)) -> (A, B, C, D, E, A, A, A) {
        (
            item.0, item.1, item.2, item.3, item.4, item.0, item.0, item.0,
        )
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive, E: CubePrimitive>
    LogicalHostPack7<(A, B, C, D, E), A, B, C, D, E, A, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>)
{
    fn pack_host(a: A, b: B, c: C, d: D, e: E, _f: A, _g: A, _h: A) -> (A, B, C, D, E) {
        (a, b, c, d, e)
    }

    fn leaves_host(item: (A, B, C, D, E)) -> (A, B, C, D, E, A, A, A) {
        (
            item.0, item.1, item.2, item.3, item.4, item.0, item.0, item.0,
        )
    }
}

#[cube]
impl<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
> LogicalDevicePack7<(A, B, C, D, E, F), A, B, C, D, E, F, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>, Slot5<F>)
{
    fn pack(a: A, b: B, c: C, d: D, e: E, f: F, _g: A, _h: A) -> (A, B, C, D, E, F) {
        (a, b, c, d, e, f)
    }

    fn unpack(item: (A, B, C, D, E, F)) -> (A, B, C, D, E, F, A, A) {
        (
            item.0, item.1, item.2, item.3, item.4, item.5, item.0, item.0,
        )
    }
}

impl<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
> LogicalHostPack7<(A, B, C, D, E, F), A, B, C, D, E, F, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>, Slot5<F>)
{
    fn pack_host(a: A, b: B, c: C, d: D, e: E, f: F, _g: A, _h: A) -> (A, B, C, D, E, F) {
        (a, b, c, d, e, f)
    }

    fn leaves_host(item: (A, B, C, D, E, F)) -> (A, B, C, D, E, F, A, A) {
        (
            item.0, item.1, item.2, item.3, item.4, item.5, item.0, item.0,
        )
    }
}

#[cube]
impl<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
> LogicalDevicePack7<(A, B, C, D, E, F, G), A, B, C, D, E, F, G, G>
    for (
        Slot0<A>,
        Slot1<B>,
        Slot2<C>,
        Slot3<D>,
        Slot4<E>,
        Slot5<F>,
        Slot6<G>,
    )
{
    fn pack(a: A, b: B, c: C, d: D, e: E, f: F, g: G, _h: G) -> (A, B, C, D, E, F, G) {
        (a, b, c, d, e, f, g)
    }

    fn unpack(item: (A, B, C, D, E, F, G)) -> (A, B, C, D, E, F, G, G) {
        (
            item.0, item.1, item.2, item.3, item.4, item.5, item.6, item.6,
        )
    }
}

impl<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
> LogicalHostPack7<(A, B, C, D, E, F, G), A, B, C, D, E, F, G, G>
    for (
        Slot0<A>,
        Slot1<B>,
        Slot2<C>,
        Slot3<D>,
        Slot4<E>,
        Slot5<F>,
        Slot6<G>,
    )
{
    fn pack_host(a: A, b: B, c: C, d: D, e: E, f: F, g: G, _h: G) -> (A, B, C, D, E, F, G) {
        (a, b, c, d, e, f, g)
    }

    fn leaves_host(item: (A, B, C, D, E, F, G)) -> (A, B, C, D, E, F, G, G) {
        (
            item.0, item.1, item.2, item.3, item.4, item.5, item.6, item.6,
        )
    }
}

#[cube]
impl<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    H: CubePrimitive,
> LogicalDevicePack7<(A, B, C, D, E, F, G, H), A, B, C, D, E, F, G, H>
    for (
        Slot0<A>,
        Slot1<B>,
        Slot2<C>,
        Slot3<D>,
        Slot4<E>,
        Slot5<F>,
        Slot6<G>,
        Slot7<H>,
    )
{
    fn pack(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H) -> (A, B, C, D, E, F, G, H) {
        (a, b, c, d, e, f, g, h)
    }

    fn unpack(item: (A, B, C, D, E, F, G, H)) -> (A, B, C, D, E, F, G, H) {
        (
            item.0, item.1, item.2, item.3, item.4, item.5, item.6, item.7,
        )
    }
}

impl<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
    H: CubePrimitive,
> LogicalHostPack7<(A, B, C, D, E, F, G, H), A, B, C, D, E, F, G, H>
    for (
        Slot0<A>,
        Slot1<B>,
        Slot2<C>,
        Slot3<D>,
        Slot4<E>,
        Slot5<F>,
        Slot6<G>,
        Slot7<H>,
    )
{
    fn pack_host(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H) -> (A, B, C, D, E, F, G, H) {
        (a, b, c, d, e, f, g, h)
    }

    fn leaves_host(item: (A, B, C, D, E, F, G, H)) -> (A, B, C, D, E, F, G, H) {
        (
            item.0, item.1, item.2, item.3, item.4, item.5, item.6, item.7,
        )
    }
}

pub trait LogicalItemPack7: CubeType + Copy + Send + Sync + 'static {
    type Leaf0: CubePrimitive + CubeElement;
    type Leaf1: CubePrimitive + CubeElement;
    type Leaf2: CubePrimitive + CubeElement;
    type Leaf3: CubePrimitive + CubeElement;
    type Leaf4: CubePrimitive + CubeElement;
    type Leaf5: CubePrimitive + CubeElement;
    type Leaf6: CubePrimitive + CubeElement;
    type Leaf7: CubePrimitive + CubeElement;
    type Pack: LogicalDevicePack7<
            Self,
            Self::Leaf0,
            Self::Leaf1,
            Self::Leaf2,
            Self::Leaf3,
            Self::Leaf4,
            Self::Leaf5,
            Self::Leaf6,
            Self::Leaf7,
        >
        + LogicalHostPack7<
            Self,
            Self::Leaf0,
            Self::Leaf1,
            Self::Leaf2,
            Self::Leaf3,
            Self::Leaf4,
            Self::Leaf5,
            Self::Leaf6,
            Self::Leaf7,
        >
        + 'static
        + Send
        + Sync;
}

macro_rules! impl_scalar_logical_item_pack7 {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl LogicalItemPack7 for $ty {
                type Leaf0 = $ty;
                type Leaf1 = $ty;
                type Leaf2 = $ty;
                type Leaf3 = $ty;
                type Leaf4 = $ty;
                type Leaf5 = $ty;
                type Leaf6 = $ty;
                type Leaf7 = $ty;
                type Pack = Slot0<$ty>;
            }
        )+
    };
}

impl_scalar_logical_item_pack7!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[doc(hidden)]
pub trait LogicalPackLeaf:
    CubePrimitive + CubeElement + LogicalItemPack7 + Send + Sync + 'static
{
}

macro_rules! impl_logical_pack_leaf {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl LogicalPackLeaf for $ty {}
        )+
    };
}

impl_logical_pack_leaf!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[cube]
impl LogicalDevicePack7<bool, u32, u32, u32, u32, u32, u32, u32, u32> for Slot0<u32> {
    fn pack(a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32, _g: u32, _h: u32) -> bool {
        a != 0
    }

    fn unpack(item: bool) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
        let value = if item { 1u32 } else { 0u32 };
        (value, value, value, value, value, value, value, value)
    }
}

impl LogicalHostPack7<bool, u32, u32, u32, u32, u32, u32, u32, u32> for Slot0<u32> {
    fn pack_host(a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32, _g: u32, _h: u32) -> bool {
        a != 0
    }

    fn leaves_host(item: bool) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
        let value = u32::from(item);
        (value, value, value, value, value, value, value, value)
    }
}

impl LogicalItemPack7 for bool {
    type Leaf0 = u32;
    type Leaf1 = u32;
    type Leaf2 = u32;
    type Leaf3 = u32;
    type Leaf4 = u32;
    type Leaf5 = u32;
    type Leaf6 = u32;
    type Leaf7 = u32;
    type Pack = Slot0<u32>;
}

macro_rules! impl_logical_item_pack7 {
    (
        impl < $( $gen:ident ),+ >;
        $item:ty;
        $pack:ty;
        $leaf0:ty,
        $leaf1:ty,
        $leaf2:ty,
        $leaf3:ty,
        $leaf4:ty,
        $leaf5:ty,
        $leaf6:ty,
        $leaf7:ty
    ) => {
        impl<$( $gen ),+> LogicalItemPack7 for $item
        where
            $( $gen: LogicalPackLeaf, )+
            $item: CubeType + Copy + Send + Sync + 'static,
        {
            type Leaf0 = $leaf0;
            type Leaf1 = $leaf1;
            type Leaf2 = $leaf2;
            type Leaf3 = $leaf3;
            type Leaf4 = $leaf4;
            type Leaf5 = $leaf5;
            type Leaf6 = $leaf6;
            type Leaf7 = $leaf7;
            type Pack = $pack;
        }
    };
}

impl_logical_item_pack7!(
    impl<A>;
    (A,);
    (Slot0<A>,);
    A,
    A,
    A,
    A,
    A,
    A,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B>;
    (A, B);
    (Slot0<A>, Slot1<B>);
    A,
    B,
    A,
    A,
    A,
    A,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B, C>;
    (A, B, C);
    (Slot0<A>, Slot1<B>, Slot2<C>);
    A,
    B,
    C,
    A,
    A,
    A,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B, C, D>;
    (A, B, C, D);
    (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>);
    A,
    B,
    C,
    D,
    A,
    A,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B, C, D, E>;
    (A, B, C, D, E);
    (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>);
    A,
    B,
    C,
    D,
    E,
    A,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B, C, D, E, F>;
    (A, B, C, D, E, F);
    (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>, Slot5<F>);
    A,
    B,
    C,
    D,
    E,
    F,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B, C, D, E, F, G>;
    (A, B, C, D, E, F, G);
    (
        Slot0<A>,
        Slot1<B>,
        Slot2<C>,
        Slot3<D>,
        Slot4<E>,
        Slot5<F>,
        Slot6<G>
    );
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    G
);
impl_logical_item_pack7!(
    impl<A, B, C, D, E, F, G, H>;
    (A, B, C, D, E, F, G, H);
    (
        Slot0<A>,
        Slot1<B>,
        Slot2<C>,
        Slot3<D>,
        Slot4<E>,
        Slot5<F>,
        Slot6<G>,
        Slot7<H>
    );
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H
);
impl_logical_item_pack7!(
    impl<A, B>;
    ((A,), B);
    ((Slot0<A>,), Slot1<B>);
    A,
    B,
    A,
    A,
    A,
    A,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B, C>;
    ((A, B), C);
    ((Slot0<A>, Slot1<B>), Slot2<C>);
    A,
    B,
    C,
    A,
    A,
    A,
    A,
    A
);
impl_logical_item_pack7!(
    impl<A, B, C>;
    (A, (B, C));
    (Slot0<A>, (Slot1<B>, Slot2<C>));
    A,
    B,
    C,
    A,
    A,
    A,
    A,
    A
);
impl<A: CubePrimitive, B: CubePrimitive> LogicalDeviceExpr7Shape<((A,), B)>
    for ((Slot0<A>,), Slot1<B>)
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = A;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
    type Leaf7 = A;
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDeviceExpr7Shape<((A, B), C)>
    for ((Slot0<A>, Slot1<B>), Slot2<C>)
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = C;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
    type Leaf7 = A;
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive> LogicalDeviceExpr7Shape<(A, (B, C))>
    for (Slot0<A>, (Slot1<B>, Slot2<C>))
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = C;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
    type Leaf7 = A;
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> GpuExpr<C> for Slot0<T> {
    fn eval(input: &[C], indices: &[u32], _rhs: &[C], _rhs_indices: &[u32], index: usize) -> C {
        input[indices[0] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> GpuExpr<C> for Slot1<T> {
    fn eval(_input: &[C], _indices: &[u32], rhs: &[C], rhs_indices: &[u32], index: usize) -> C {
        rhs[rhs_indices[0] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> GpuExpr<C> for Slot2<T> {
    fn eval(input: &[C], indices: &[u32], _rhs: &[C], _rhs_indices: &[u32], index: usize) -> C {
        input[indices[0] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> GpuExpr<C> for Slot3<T> {
    fn eval(input: &[C], indices: &[u32], _rhs: &[C], _rhs_indices: &[u32], index: usize) -> C {
        input[indices[0] as usize + index]
    }
}

#[cube]
impl GpuExpr<u32> for ConstantSlot0<u32> {
    fn eval(
        input: &[u32],
        _indices: &[u32],
        _rhs: &[u32],
        _rhs_indices: &[u32],
        _index: usize,
    ) -> u32 {
        input[0]
    }
}

#[cube]
impl GpuExpr<u32> for CountingSlot0 {
    fn eval(
        input: &[u32],
        indices: &[u32],
        _rhs: &[u32],
        _rhs_indices: &[u32],
        index: usize,
    ) -> u32 {
        input[0] + indices[0] + index as u32
    }
}

#[cube]
impl<T, InputExpr, Op> GpuExpr<T> for TransformExpr<InputExpr, T, Op>
where
    T: CubePrimitive,
    InputExpr: GpuExpr<T>,
    Op: crate::detail::op::kernel::UnaryOp<T, Output = T>,
{
    fn eval(input: &[T], indices: &[u32], rhs: &[T], rhs_indices: &[u32], index: usize) -> T {
        Op::apply(InputExpr::eval(input, indices, rhs, rhs_indices, index))
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> DeviceGpuExpr<C> for Slot0<T> {
    fn eval(
        slot0: &[C],
        _slot1: &[C],
        _slot2: &[C],
        _slot3: &[C],
        slot_offsets: &[u32],
        index: usize,
    ) -> C {
        slot0[slot_offsets[0] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> DeviceGpuExpr<C> for Slot1<T> {
    fn eval(
        _slot0: &[C],
        slot1: &[C],
        _slot2: &[C],
        _slot3: &[C],
        slot_offsets: &[u32],
        index: usize,
    ) -> C {
        slot1[slot_offsets[1] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> DeviceGpuExpr<C> for Slot2<T> {
    fn eval(
        _slot0: &[C],
        _slot1: &[C],
        slot2: &[C],
        _slot3: &[C],
        slot_offsets: &[u32],
        index: usize,
    ) -> C {
        slot2[slot_offsets[2] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> DeviceGpuExpr<C> for Slot3<T> {
    fn eval(
        _slot0: &[C],
        _slot1: &[C],
        _slot2: &[C],
        slot3: &[C],
        slot_offsets: &[u32],
        index: usize,
    ) -> C {
        slot3[slot_offsets[3] as usize + index]
    }
}

#[cube]
impl DeviceGpuExpr<u32> for ConstantSlot0<u32> {
    fn eval(
        slot0: &[u32],
        _slot1: &[u32],
        _slot2: &[u32],
        _slot3: &[u32],
        _slot_offsets: &[u32],
        _index: usize,
    ) -> u32 {
        slot0[0]
    }
}

#[cube]
impl DeviceGpuExpr<u32> for CountingSlot0 {
    fn eval(
        slot0: &[u32],
        _slot1: &[u32],
        _slot2: &[u32],
        _slot3: &[u32],
        slot_offsets: &[u32],
        index: usize,
    ) -> u32 {
        slot0[0] + slot_offsets[0] + index as u32
    }
}

#[cube]
impl<T, InputExpr, Op> DeviceGpuExpr<T> for TransformExpr<InputExpr, T, Op>
where
    T: CubePrimitive,
    InputExpr: DeviceGpuExpr<T>,
    Op: crate::detail::op::kernel::UnaryOp<T, Output = T>,
{
    fn eval(
        slot0: &[T],
        slot1: &[T],
        slot2: &[T],
        slot3: &[T],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        Op::apply(InputExpr::eval(
            slot0,
            slot1,
            slot2,
            slot3,
            slot_offsets,
            index,
        ))
    }
}
