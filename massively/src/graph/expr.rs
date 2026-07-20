//! Edge-context expressions consumed by a traversal terminal.

#![allow(private_interfaces)]

use cubecl::prelude::Runtime;

use crate::api::iter::Zipped;
use crate::{DeviceVec, Error, Executor, MAlloc, MIndex, MIter, MStorage, Zip};

use super::control::TraversalControl;

#[doc(hidden)]
pub mod private {
    use super::*;

    pub trait Sealed {}

    pub trait EdgeExprImpl<R: Runtime>: EdgeExpr<R> + Sized {
        type Storage: MStorage<R, Item = <Self as EdgeExpr<R>>::Item>;

        fn materialize(
            self,
            exec: &Executor<R>,
            control: &TraversalControl<R>,
        ) -> Result<Self::Storage, Error>;
    }
}

/// A value expression evaluated once for every traversed edge.
///
/// Expressions are constructed with [`source`], [`destination`], [`edge`], or one of the ID
/// constructors and combined with Massively's `zipN` functions. Implementations are sealed so
/// that the backend can change its fused representation without exposing traversal control.
#[allow(private_bounds)]
pub trait EdgeExpr<R: Runtime>: private::Sealed {
    type Item: MAlloc<R>;
}

#[derive(Clone, Copy, Debug)]
pub struct Source<Values>(Values);

#[derive(Clone, Copy, Debug)]
pub struct Destination<Values>(Values);

#[derive(Clone, Copy, Debug)]
pub struct Edge<Values>(Values);

#[derive(Clone, Copy, Debug, Default)]
pub struct SourceId;

#[derive(Clone, Copy, Debug, Default)]
pub struct DestinationId;

#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeId;

/// Reads a vertex value at the source endpoint of each traversed edge.
pub const fn source<Values>(values: Values) -> Source<Values> {
    Source(values)
}

/// Reads a vertex value at the destination endpoint of each traversed edge.
pub const fn destination<Values>(values: Values) -> Destination<Values> {
    Destination(values)
}

/// Reads an edge value at the CSR edge position of each traversed edge.
pub const fn edge<Values>(values: Values) -> Edge<Values> {
    Edge(values)
}

/// Produces the source vertex ID of each traversed edge.
pub const fn source_id() -> SourceId {
    SourceId
}

/// Produces the destination vertex ID of each traversed edge.
pub const fn destination_id() -> DestinationId {
    DestinationId
}

/// Produces the CSR edge position of each traversed edge.
pub const fn edge_id() -> EdgeId {
    EdgeId
}

impl<Values> private::Sealed for Source<Values> {}
impl<Values> private::Sealed for Destination<Values> {}
impl<Values> private::Sealed for Edge<Values> {}
impl private::Sealed for SourceId {}
impl private::Sealed for DestinationId {}
impl private::Sealed for EdgeId {}

impl<R, Values> EdgeExpr<R> for Source<Values>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    <Values::Item as MAlloc<R>>::Owned: MStorage<R, Item = Values::Item>,
{
    type Item = Values::Item;
}

impl<R, Values> EdgeExpr<R> for Destination<Values>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    <Values::Item as MAlloc<R>>::Owned: MStorage<R, Item = Values::Item>,
{
    type Item = Values::Item;
}

impl<R, Values> EdgeExpr<R> for Edge<Values>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    <Values::Item as MAlloc<R>>::Owned: MStorage<R, Item = Values::Item>,
{
    type Item = Values::Item;
}

impl<R: Runtime> EdgeExpr<R> for SourceId {
    type Item = MIndex;
}

impl<R: Runtime> EdgeExpr<R> for DestinationId {
    type Item = MIndex;
}

impl<R: Runtime> EdgeExpr<R> for EdgeId {
    type Item = MIndex;
}

impl<R, Values> private::EdgeExprImpl<R> for Source<Values>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    <Values::Item as MAlloc<R>>::Owned: MStorage<R, Item = Values::Item>,
{
    type Storage = <Values::Item as MAlloc<R>>::Owned;

    fn materialize(
        self,
        exec: &Executor<R>,
        control: &TraversalControl<R>,
    ) -> Result<Self::Storage, Error> {
        let output = <Self::Storage as MStorage<R>>::allocate(exec, control.capacity as usize);
        crate::vector::gather_into(
            exec,
            self.0,
            control.sources.slice(..),
            output.slice_mut(..),
        )?;
        Ok(output)
    }
}

impl<R, Values> private::EdgeExprImpl<R> for Destination<Values>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    <Values::Item as MAlloc<R>>::Owned: MStorage<R, Item = Values::Item>,
{
    type Storage = <Values::Item as MAlloc<R>>::Owned;

    fn materialize(
        self,
        exec: &Executor<R>,
        control: &TraversalControl<R>,
    ) -> Result<Self::Storage, Error> {
        let output = <Self::Storage as MStorage<R>>::allocate(exec, control.capacity as usize);
        crate::vector::gather_into(
            exec,
            self.0,
            control.destinations.slice(..),
            output.slice_mut(..),
        )?;
        Ok(output)
    }
}

impl<R, Values> private::EdgeExprImpl<R> for Edge<Values>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    <Values::Item as MAlloc<R>>::Owned: MStorage<R, Item = Values::Item>,
{
    type Storage = <Values::Item as MAlloc<R>>::Owned;

    fn materialize(
        self,
        exec: &Executor<R>,
        control: &TraversalControl<R>,
    ) -> Result<Self::Storage, Error> {
        let output = <Self::Storage as MStorage<R>>::allocate(exec, control.capacity as usize);
        crate::vector::gather_into(exec, self.0, control.edges.slice(..), output.slice_mut(..))?;
        Ok(output)
    }
}

impl<R: Runtime> private::EdgeExprImpl<R> for SourceId {
    type Storage = DeviceVec<R, u32>;

    fn materialize(
        self,
        _exec: &Executor<R>,
        control: &TraversalControl<R>,
    ) -> Result<Self::Storage, Error> {
        Ok(control.sources.clone())
    }
}

impl<R: Runtime> private::EdgeExprImpl<R> for DestinationId {
    type Storage = DeviceVec<R, u32>;

    fn materialize(
        self,
        _exec: &Executor<R>,
        control: &TraversalControl<R>,
    ) -> Result<Self::Storage, Error> {
        Ok(control.destinations.clone())
    }
}

impl<R: Runtime> private::EdgeExprImpl<R> for EdgeId {
    type Storage = DeviceVec<R, u32>;

    fn materialize(
        self,
        _exec: &Executor<R>,
        control: &TraversalControl<R>,
    ) -> Result<Self::Storage, Error> {
        Ok(control.edges.clone())
    }
}

impl<Left, Right> private::Sealed for Zipped<Left, Right>
where
    Left: private::Sealed,
    Right: private::Sealed,
{
}

impl<R, Left, Right> EdgeExpr<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: EdgeExpr<R> + private::EdgeExprImpl<R>,
    Right: EdgeExpr<R> + private::EdgeExprImpl<R>,
    Zip<Left::Storage, Right::Storage>: MStorage<R>,
    <Zip<Left::Storage, Right::Storage> as MStorage<R>>::Item: MAlloc<R>,
{
    type Item = <Zip<Left::Storage, Right::Storage> as MStorage<R>>::Item;
}

impl<R, Left, Right> private::EdgeExprImpl<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: EdgeExpr<R> + private::EdgeExprImpl<R>,
    Right: EdgeExpr<R> + private::EdgeExprImpl<R>,
    Zip<Left::Storage, Right::Storage>: MStorage<R>,
    <Zip<Left::Storage, Right::Storage> as MStorage<R>>::Item: MAlloc<R>,
{
    type Storage = Zip<Left::Storage, Right::Storage>;

    fn materialize(
        self,
        exec: &Executor<R>,
        control: &TraversalControl<R>,
    ) -> Result<Self::Storage, Error> {
        let (left, right) = self.into_parts();
        Ok(Zip::new(
            left.materialize(exec, control)?,
            right.materialize(exec, control)?,
        ))
    }
}

pub(super) use private::EdgeExprImpl;
