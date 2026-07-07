//! Lazy read-only massively iterator constructors.

use cubecl::prelude::{CubeElement, Runtime};
use std::marker::PhantomData;

use crate::{Error, MIndex, iter::MIter, op, value::MStorageElement};

/// Lazy constant stream before a finite length is assigned.
#[derive(Debug)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Turns this lazy stream into a finite read-only iterator.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken { expr: self, len }
    }
}

/// Lazy counting stream before a finite length is assigned.
#[derive(Debug)]
pub struct Counting {
    start: MIndex,
}

impl Counting {
    /// Turns this lazy stream into a finite read-only iterator.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken { expr: self, len }
    }
}

/// Finite lazy read-only iterator.
#[derive(Debug)]
pub struct Taken<Expr> {
    expr: Expr,
    len: MIndex,
}

/// Lazy indexed read-only iterator.
#[derive(Debug)]
pub struct Gather<Values, Indices> {
    values: Values,
    indices: Indices,
}

/// Lazy unary transform read-only iterator.
#[derive(Debug)]
pub struct Transform<Input, Op> {
    input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// Creates a lazy constant stream.
pub fn constant<T>(value: T) -> Constant<T> {
    Constant { value }
}

/// Creates a lazy counting stream whose first value is `start`.
pub fn counting(start: MIndex) -> Counting {
    Counting { start }
}

/// Creates a lazy gather expression that reads `values[indices[i]]`.
pub fn gather<Values, Indices>(values: Values, indices: Indices) -> Gather<Values, Indices> {
    Gather { values, indices }
}

/// Creates a lazy transform expression that reads `op(input[i])`.
pub fn transform<Input, Op>(input: Input, _op: Op) -> Transform<Input, Op> {
    Transform {
        input,
        _op: PhantomData,
    }
}

impl<R, T> MIter<R> for Taken<Constant<T>>
where
    R: Runtime,
    T: MStorageElement + 'static,
{
    type Item = T;
    type Inner = ();
    type Read = crate::detail::read::ConstantRead<T>;

    fn len(&self) -> MIndex {
        self.len
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy constant MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        let handle = policy
            .client()
            .create_from_slice(T::as_bytes(&[self.expr.value]));
        Ok(crate::detail::read::ConstantRead::new(
            handle,
            self.len as usize,
        ))
    }

    fn validate_executor(&self, _exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, Values, Indices> MIter<R> for Gather<Values, Indices>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = MIndex>,
    crate::detail::read::GatherRead<Values::Read, Indices::Read>:
        crate::detail::read::KernelReadBoundMany<R, Item = Values::Item>,
{
    type Item = Values::Item;
    type Inner = ();
    type Read = crate::detail::read::GatherRead<Values::Read, Indices::Read>;

    fn len(&self) -> MIndex {
        self.indices.len()
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy gather MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        let values = self.values.lower_read_ref(policy)?;
        let indices = self.indices.lower_read_ref(policy)?;
        Ok(crate::detail::read::GatherRead::new(values, indices))
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.values.validate_executor(exec)?;
        self.indices.validate_executor(exec)
    }
}

impl<R, Input, Op> MIter<R> for Transform<Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
    Op: op::UnaryOp<R, Input::Item, Env = ()>,
    crate::detail::read::TransformRead<Input::Read, Op>:
        crate::detail::read::KernelReadBoundMany<R, Item = Op::Output>,
{
    type Item = Op::Output;
    type Inner = ();
    type Read = crate::detail::read::TransformRead<Input::Read, Op>;

    fn len(&self) -> MIndex {
        self.input.len()
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy transform MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        let input = self.input.lower_read_ref(policy)?;
        Ok(crate::detail::read::TransformRead::new(input))
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.input.validate_executor(exec)
    }
}

impl<R> MIter<R> for Taken<Counting>
where
    R: Runtime,
{
    type Item = MIndex;
    type Inner = ();
    type Read = crate::detail::read::CountingRead;

    fn len(&self) -> MIndex {
        self.len
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy counting MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        let handle = policy
            .client()
            .create_from_slice(MIndex::as_bytes(&[self.expr.start]));
        Ok(crate::detail::read::CountingRead::new(
            handle,
            self.len as usize,
        ))
    }

    fn validate_executor(&self, _exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        Ok(())
    }
}
