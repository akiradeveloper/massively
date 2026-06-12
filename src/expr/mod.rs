use crate::op::{BinaryOp, UnaryOp};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Type-level input node for a device expression.
pub struct Input<T> {
    _item: PhantomData<fn() -> T>,
}

/// Type-level map node for a device expression.
pub struct Map<Source, Op> {
    _source: PhantomData<fn() -> Source>,
    _op: PhantomData<fn() -> Op>,
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
    fn eval(
        input: &Array<T>,
        indices: &Array<u32>,
        rhs: &Array<T>,
        rhs_indices: &Array<u32>,
        index: usize,
    ) -> T;
}

/// CubeCL expression tree over up to four staged device slots.
#[cube]
pub trait DeviceGpuExpr<T: CubePrimitive>: 'static + Send + Sync {
    /// Evaluates this expression for `index`.
    fn eval(
        slot0: &Array<T>,
        slot1: &Array<T>,
        slot2: &Array<T>,
        slot3: &Array<T>,
        index: usize,
    ) -> T;
}

#[cube]
impl<T: CubePrimitive> GpuExpr<T> for Input<T> {
    fn eval(
        input: &Array<T>,
        _indices: &Array<u32>,
        _rhs: &Array<T>,
        _rhs_indices: &Array<u32>,
        index: usize,
    ) -> T {
        input[index]
    }
}

#[cube]
impl<T: CubePrimitive> GpuExpr<T> for Slot0<T> {
    fn eval(
        input: &Array<T>,
        _indices: &Array<u32>,
        _rhs: &Array<T>,
        _rhs_indices: &Array<u32>,
        index: usize,
    ) -> T {
        input[index]
    }
}

#[cube]
impl<T: CubePrimitive> GpuExpr<T> for Slot1<T> {
    fn eval(
        input: &Array<T>,
        _indices: &Array<u32>,
        _rhs: &Array<T>,
        _rhs_indices: &Array<u32>,
        index: usize,
    ) -> T {
        input[index]
    }
}

#[cube]
impl<T: CubePrimitive> DeviceGpuExpr<T> for Slot0<T> {
    fn eval(
        slot0: &Array<T>,
        _slot1: &Array<T>,
        _slot2: &Array<T>,
        _slot3: &Array<T>,
        index: usize,
    ) -> T {
        slot0[index]
    }
}

#[cube]
impl<T: CubePrimitive> DeviceGpuExpr<T> for Slot1<T> {
    fn eval(
        _slot0: &Array<T>,
        slot1: &Array<T>,
        _slot2: &Array<T>,
        _slot3: &Array<T>,
        index: usize,
    ) -> T {
        slot1[index]
    }
}

#[cube]
impl<T: CubePrimitive> DeviceGpuExpr<T> for Slot2<T> {
    fn eval(
        _slot0: &Array<T>,
        _slot1: &Array<T>,
        slot2: &Array<T>,
        _slot3: &Array<T>,
        index: usize,
    ) -> T {
        slot2[index]
    }
}

#[cube]
impl<T: CubePrimitive> DeviceGpuExpr<T> for Slot3<T> {
    fn eval(
        _slot0: &Array<T>,
        _slot1: &Array<T>,
        _slot2: &Array<T>,
        slot3: &Array<T>,
        index: usize,
    ) -> T {
        slot3[index]
    }
}

#[cube]
impl<T, Source, Op> GpuExpr<T> for Map<Source, Op>
where
    T: CubePrimitive,
    Source: GpuExpr<T>,
    Op: UnaryOp<T, Output = T>,
{
    fn eval(
        input: &Array<T>,
        indices: &Array<u32>,
        rhs: &Array<T>,
        rhs_indices: &Array<u32>,
        index: usize,
    ) -> T {
        Op::apply(Source::eval(input, indices, rhs, rhs_indices, index))
    }
}

#[cube]
impl<T, Source, Op> DeviceGpuExpr<T> for Map<Source, Op>
where
    T: CubePrimitive,
    Source: DeviceGpuExpr<T>,
    Op: UnaryOp<T, Output = T>,
{
    fn eval(
        slot0: &Array<T>,
        slot1: &Array<T>,
        slot2: &Array<T>,
        slot3: &Array<T>,
        index: usize,
    ) -> T {
        Op::apply(Source::eval(slot0, slot1, slot2, slot3, index))
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
    fn eval(
        input: &Array<T>,
        indices: &Array<u32>,
        rhs: &Array<T>,
        rhs_indices: &Array<u32>,
        index: usize,
    ) -> T {
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
        slot0: &Array<T>,
        slot1: &Array<T>,
        slot2: &Array<T>,
        slot3: &Array<T>,
        index: usize,
    ) -> T {
        Op::apply(
            Left::eval(slot0, slot1, slot2, slot3, index),
            Right::eval(slot0, slot1, slot2, slot3, index),
        )
    }
}
