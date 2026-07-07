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
impl<T> LogicalDeviceExpr<T> for ConstantSlot0<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot1<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot2<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot3<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot4<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot5<T> where T: CubePrimitive + 'static {}
impl<T> LogicalDeviceExpr<T> for ConstantSlot6<T> where T: CubePrimitive + 'static {}
impl LogicalDeviceExpr<u32> for CountingSlot0 {}
impl LogicalDeviceExpr<u32> for CountingSlot1 {}
impl LogicalDeviceExpr<u32> for CountingSlot2 {}
impl LogicalDeviceExpr<u32> for CountingSlot3 {}
impl LogicalDeviceExpr<u32> for CountingSlot4 {}
impl LogicalDeviceExpr<u32> for CountingSlot5 {}
impl LogicalDeviceExpr<u32> for CountingSlot6 {}

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

/// Executable logical expression over up to seven physical leaf slots.
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
>: LogicalDeviceExpr<Item>
{
    /// Evaluates one logical item from seven staged storage leaves.
    #[allow(clippy::too_many_arguments)]
    fn eval7(
        slot0: &[A],
        slot1: &[B],
        slot2: &[C],
        slot3: &[D],
        slot4: &[E],
        slot5: &[F],
        slot6: &[G],
        slot_offsets: &[u32],
        index: usize,
    ) -> Item;
}

#[cube]
impl<T, B, C, D, E, F, G> LogicalDeviceExpr7<T, T, B, C, D, E, F, G> for Slot0<T>
where
    T: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
{
    fn eval7(
        slot0: &[T],
        _slot1: &[B],
        _slot2: &[C],
        _slot3: &[D],
        _slot4: &[E],
        _slot5: &[F],
        _slot6: &[G],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        slot0[slot_offsets[0] as usize + index]
    }
}

#[cube]
impl<A, T, C, D, E, F, G> LogicalDeviceExpr7<T, A, T, C, D, E, F, G> for Slot1<T>
where
    A: CubePrimitive,
    T: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
{
    fn eval7(
        _slot0: &[A],
        slot1: &[T],
        _slot2: &[C],
        _slot3: &[D],
        _slot4: &[E],
        _slot5: &[F],
        _slot6: &[G],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        slot1[slot_offsets[1] as usize + index]
    }
}

#[cube]
impl<A, B, T, D, E, F, G> LogicalDeviceExpr7<T, A, B, T, D, E, F, G> for Slot2<T>
where
    A: CubePrimitive,
    B: CubePrimitive,
    T: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
{
    fn eval7(
        _slot0: &[A],
        _slot1: &[B],
        slot2: &[T],
        _slot3: &[D],
        _slot4: &[E],
        _slot5: &[F],
        _slot6: &[G],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        slot2[slot_offsets[2] as usize + index]
    }
}

#[cube]
impl<A, B, C, T, E, F, G> LogicalDeviceExpr7<T, A, B, C, T, E, F, G> for Slot3<T>
where
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    T: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
{
    fn eval7(
        _slot0: &[A],
        _slot1: &[B],
        _slot2: &[C],
        slot3: &[T],
        _slot4: &[E],
        _slot5: &[F],
        _slot6: &[G],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        slot3[slot_offsets[3] as usize + index]
    }
}

#[cube]
impl<A, B, C, D, T, F, G> LogicalDeviceExpr7<T, A, B, C, D, T, F, G> for Slot4<T>
where
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    T: CubePrimitive,
    F: CubePrimitive,
    G: CubePrimitive,
{
    fn eval7(
        _slot0: &[A],
        _slot1: &[B],
        _slot2: &[C],
        _slot3: &[D],
        slot4: &[T],
        _slot5: &[F],
        _slot6: &[G],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        slot4[slot_offsets[4] as usize + index]
    }
}

#[cube]
impl<A, B, C, D, E, T, G> LogicalDeviceExpr7<T, A, B, C, D, E, T, G> for Slot5<T>
where
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    T: CubePrimitive,
    G: CubePrimitive,
{
    fn eval7(
        _slot0: &[A],
        _slot1: &[B],
        _slot2: &[C],
        _slot3: &[D],
        _slot4: &[E],
        slot5: &[T],
        _slot6: &[G],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        slot5[slot_offsets[5] as usize + index]
    }
}

#[cube]
impl<A, B, C, D, E, F, T> LogicalDeviceExpr7<T, A, B, C, D, E, F, T> for Slot6<T>
where
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
    T: CubePrimitive,
{
    fn eval7(
        _slot0: &[A],
        _slot1: &[B],
        _slot2: &[C],
        _slot3: &[D],
        _slot4: &[E],
        _slot5: &[F],
        slot6: &[T],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        slot6[slot_offsets[6] as usize + index]
    }
}

macro_rules! impl_logical_device_expr7_tuple {
    ($( $expr:ident : $item:ident ),+) => {
        #[cube]
        impl<Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, $( $expr, $item ),+>
            LogicalDeviceExpr7<($( $item, )+), Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
            for ($( $expr, )+)
        where
            Leaf0: CubePrimitive,
            Leaf1: CubePrimitive,
            Leaf2: CubePrimitive,
            Leaf3: CubePrimitive,
            Leaf4: CubePrimitive,
            Leaf5: CubePrimitive,
            Leaf6: CubePrimitive,
            $( $item: CubeType + 'static, )+
            $( $expr: LogicalDeviceExpr7<$item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>, )+
        {
            fn eval7(
                slot0: &[Leaf0],
                slot1: &[Leaf1],
                slot2: &[Leaf2],
                slot3: &[Leaf3],
                slot4: &[Leaf4],
                slot5: &[Leaf5],
                slot6: &[Leaf6],
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

macro_rules! impl_constant_logical_device_expr7 {
    ($expr:ident, $item:ident; <$( $gen:ident ),+>; $a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty; $s0:ident, $s1:ident, $s2:ident, $s3:ident, $s4:ident, $s5:ident, $s6:ident; $value:expr) => {
        #[cube]
        impl<$( $gen ),+> LogicalDeviceExpr7<$item, $a, $b, $c, $d, $e, $f, $g>
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
                _slot_offsets: &[u32],
                _index: usize,
            ) -> $item {
                let _ = ($s0, $s1, $s2, $s3, $s4, $s5, $s6);
                $value
            }
        }
    };
}

impl_constant_logical_device_expr7!(ConstantSlot0, A; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot0[0]);
impl_constant_logical_device_expr7!(ConstantSlot1, B; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot1[0]);
impl_constant_logical_device_expr7!(ConstantSlot2, C; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot2[0]);
impl_constant_logical_device_expr7!(ConstantSlot3, D; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot3[0]);
impl_constant_logical_device_expr7!(ConstantSlot4, E; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot4[0]);
impl_constant_logical_device_expr7!(ConstantSlot5, F; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot5[0]);
impl_constant_logical_device_expr7!(ConstantSlot6, G; <A, B, C, D, E, F, G>; A, B, C, D, E, F, G; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot6[0]);

macro_rules! impl_counting_logical_device_expr7 {
    ($expr:ident; <$( $gen:ident ),*>; $a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty; $index:literal; $s0:ident, $s1:ident, $s2:ident, $s3:ident, $s4:ident, $s5:ident, $s6:ident; $start:expr) => {
        #[cube]
        impl<$( $gen ),*> LogicalDeviceExpr7<u32, $a, $b, $c, $d, $e, $f, $g> for $expr
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
                slot_offsets: &[u32],
                index: usize,
            ) -> u32 {
                let _ = ($s0, $s1, $s2, $s3, $s4, $s5, $s6);
                $start + slot_offsets[$index] + index as u32
            }
        }
    };
}

impl_counting_logical_device_expr7!(CountingSlot0; <B, C, D, E, F, G>; u32, B, C, D, E, F, G; 0; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot0[0]);
impl_counting_logical_device_expr7!(CountingSlot1; <A, C, D, E, F, G>; A, u32, C, D, E, F, G; 1; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot1[0]);
impl_counting_logical_device_expr7!(CountingSlot2; <A, B, D, E, F, G>; A, B, u32, D, E, F, G; 2; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot2[0]);
impl_counting_logical_device_expr7!(CountingSlot3; <A, B, C, E, F, G>; A, B, C, u32, E, F, G; 3; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot3[0]);
impl_counting_logical_device_expr7!(CountingSlot4; <A, B, C, D, F, G>; A, B, C, D, u32, F, G; 4; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot4[0]);
impl_counting_logical_device_expr7!(CountingSlot5; <A, B, C, D, E, G>; A, B, C, D, E, u32, G; 5; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot5[0]);
impl_counting_logical_device_expr7!(CountingSlot6; <A, B, C, D, E, F>; A, B, C, D, E, F, u32; 6; slot0, slot1, slot2, slot3, slot4, slot5, slot6; slot6[0]);

#[cube]
impl<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, ValueExpr, IndexExpr>
    LogicalDeviceExpr7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
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
    ValueExpr: LogicalDeviceExpr7<Item, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
    IndexExpr: LogicalDeviceExpr7<u32, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
{
    fn eval7(
        slot0: &[Leaf0],
        slot1: &[Leaf1],
        slot2: &[Leaf2],
        slot3: &[Leaf3],
        slot4: &[Leaf4],
        slot5: &[Leaf5],
        slot6: &[Leaf6],
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
            slot_offsets,
            gathered as usize,
        )
    }
}

#[cube]
impl<InputItem, OutputItem, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, InputExpr, Op>
    LogicalDeviceExpr7<OutputItem, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>
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
    InputExpr: LogicalDeviceExpr7<InputItem, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6>,
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
>
{
    fn pack(a: A, b: B, c: C, d: D, e: E, f: F, g: G) -> Item;
    fn unpack(item: Item) -> (A, B, C, D, E, F, G);
}

pub trait LogicalHostPack7<Item, A, B, C, D, E, F, G> {
    fn pack_host(a: A, b: B, c: C, d: D, e: E, f: F, g: G) -> Item;
    fn leaves_host(item: Item) -> (A, B, C, D, E, F, G);
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

/// Type-level leaf set for executing a logical expression through seven slots.
pub trait LogicalDeviceExpr7Shape<Item: CubeType>: LogicalDeviceExpr<Item> {
    type Leaf0: CubePrimitive;
    type Leaf1: CubePrimitive;
    type Leaf2: CubePrimitive;
    type Leaf3: CubePrimitive;
    type Leaf4: CubePrimitive;
    type Leaf5: CubePrimitive;
    type Leaf6: CubePrimitive;
}

impl<A: CubePrimitive> LogicalDeviceExpr7Shape<A> for Slot0<A> {
    type Leaf0 = A;
    type Leaf1 = A;
    type Leaf2 = A;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
}

impl<A: CubePrimitive> LogicalDeviceExpr7Shape<(A,)> for (Slot0<A>,) {
    type Leaf0 = A;
    type Leaf1 = A;
    type Leaf2 = A;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
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
        $leaf6:ty
    ) => {
        impl<$( $gen: CubePrimitive ),+> LogicalDeviceExpr7Shape<$item> for $expr {
            type Leaf0 = $leaf0;
            type Leaf1 = $leaf1;
            type Leaf2 = $leaf2;
            type Leaf3 = $leaf3;
            type Leaf4 = $leaf4;
            type Leaf5 = $leaf5;
            type Leaf6 = $leaf6;
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
    G
);

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>
    LogicalDevicePack7<(A, B, C, D), A, B, C, D, A, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>)
{
    fn pack(a: A, b: B, c: C, d: D, _e: A, _f: A, _g: A) -> (A, B, C, D) {
        (a, b, c, d)
    }

    fn unpack(item: (A, B, C, D)) -> (A, B, C, D, A, A, A) {
        (item.0, item.1, item.2, item.3, item.0, item.0, item.0)
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>
    LogicalHostPack7<(A, B, C, D), A, B, C, D, A, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>)
{
    fn pack_host(a: A, b: B, c: C, d: D, _e: A, _f: A, _g: A) -> (A, B, C, D) {
        (a, b, c, d)
    }

    fn leaves_host(item: (A, B, C, D)) -> (A, B, C, D, A, A, A) {
        (item.0, item.1, item.2, item.3, item.0, item.0, item.0)
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive, E: CubePrimitive>
    LogicalDevicePack7<(A, B, C, D, E), A, B, C, D, E, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>)
{
    fn pack(a: A, b: B, c: C, d: D, e: E, _f: A, _g: A) -> (A, B, C, D, E) {
        (a, b, c, d, e)
    }

    fn unpack(item: (A, B, C, D, E)) -> (A, B, C, D, E, A, A) {
        (item.0, item.1, item.2, item.3, item.4, item.0, item.0)
    }
}

impl<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive, E: CubePrimitive>
    LogicalHostPack7<(A, B, C, D, E), A, B, C, D, E, A, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>)
{
    fn pack_host(a: A, b: B, c: C, d: D, e: E, _f: A, _g: A) -> (A, B, C, D, E) {
        (a, b, c, d, e)
    }

    fn leaves_host(item: (A, B, C, D, E)) -> (A, B, C, D, E, A, A) {
        (item.0, item.1, item.2, item.3, item.4, item.0, item.0)
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
> LogicalDevicePack7<(A, B, C, D, E, F), A, B, C, D, E, F, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>, Slot5<F>)
{
    fn pack(a: A, b: B, c: C, d: D, e: E, f: F, _g: A) -> (A, B, C, D, E, F) {
        (a, b, c, d, e, f)
    }

    fn unpack(item: (A, B, C, D, E, F)) -> (A, B, C, D, E, F, A) {
        (item.0, item.1, item.2, item.3, item.4, item.5, item.0)
    }
}

impl<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    E: CubePrimitive,
    F: CubePrimitive,
> LogicalHostPack7<(A, B, C, D, E, F), A, B, C, D, E, F, A>
    for (Slot0<A>, Slot1<B>, Slot2<C>, Slot3<D>, Slot4<E>, Slot5<F>)
{
    fn pack_host(a: A, b: B, c: C, d: D, e: E, f: F, _g: A) -> (A, B, C, D, E, F) {
        (a, b, c, d, e, f)
    }

    fn leaves_host(item: (A, B, C, D, E, F)) -> (A, B, C, D, E, F, A) {
        (item.0, item.1, item.2, item.3, item.4, item.5, item.0)
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
> LogicalDevicePack7<(A, B, C, D, E, F, G), A, B, C, D, E, F, G>
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
    fn pack(a: A, b: B, c: C, d: D, e: E, f: F, g: G) -> (A, B, C, D, E, F, G) {
        (a, b, c, d, e, f, g)
    }

    fn unpack(item: (A, B, C, D, E, F, G)) -> (A, B, C, D, E, F, G) {
        item
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
> LogicalHostPack7<(A, B, C, D, E, F, G), A, B, C, D, E, F, G>
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
    fn pack_host(a: A, b: B, c: C, d: D, e: E, f: F, g: G) -> (A, B, C, D, E, F, G) {
        (a, b, c, d, e, f, g)
    }

    fn leaves_host(item: (A, B, C, D, E, F, G)) -> (A, B, C, D, E, F, G) {
        item
    }
}

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
