use crate::op::BinaryOp;
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Type-level input node for a device expression.
pub struct Input<T> {
    _item: PhantomData<fn() -> T>,
}

/// Type-level binary map node over two device expressions.
pub struct BinaryMap<Left, Right, Op> {
    _left: PhantomData<fn() -> Left>,
    _right: PhantomData<fn() -> Right>,
    _op: PhantomData<fn() -> Op>,
}

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

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> GpuExpr<C> for Input<T> {
    fn eval(input: &[C], indices: &[u32], _rhs: &[C], _rhs_indices: &[u32], index: usize) -> C {
        input[indices[0] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> GpuExpr<C> for Slot0<T> {
    fn eval(input: &[C], indices: &[u32], _rhs: &[C], _rhs_indices: &[u32], index: usize) -> C {
        input[indices[0] as usize + index]
    }
}

#[cube]
impl<T: CubePrimitive, C: CubePrimitive> GpuExpr<C> for Slot1<T> {
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

#[cube]
impl<T, Left, Right, Op> GpuExpr<T> for BinaryMap<Left, Right, Op>
where
    T: CubePrimitive,
    Left: GpuExpr<T>,
    Right: GpuExpr<T>,
    Op: BinaryOp<T>,
{
    fn eval(input: &[T], indices: &[u32], rhs: &[T], rhs_indices: &[u32], index: usize) -> T {
        Op::apply(
            Left::eval(input, indices, rhs, rhs_indices, index),
            Right::eval(rhs, rhs_indices, input, indices, index),
        )
    }
}

#[cube]
impl<T, Left, Right, Op> DeviceGpuExpr<T> for BinaryMap<Left, Right, Op>
where
    T: CubePrimitive,
    Left: DeviceGpuExpr<T>,
    Right: DeviceGpuExpr<T>,
    Op: BinaryOp<T>,
{
    fn eval(
        slot0: &[T],
        slot1: &[T],
        slot2: &[T],
        slot3: &[T],
        slot_offsets: &[u32],
        index: usize,
    ) -> T {
        Op::apply(
            Left::eval(slot0, slot1, slot2, slot3, slot_offsets, index),
            Right::eval(slot0, slot1, slot2, slot3, slot_offsets, index),
        )
    }
}
