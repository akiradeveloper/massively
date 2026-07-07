//! Lazy read-only massively iterator constructors.

use cubecl::prelude::{CubeElement, Runtime};

use crate::{Error, MIndex, iter::MIter, value::MStorageElement};

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

/// Creates a lazy constant stream.
pub fn constant<T>(value: T) -> Constant<T> {
    Constant { value }
}

/// Creates a lazy counting stream whose first value is `start`.
pub fn counting(start: MIndex) -> Counting {
    Counting { start }
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
