use cubecl::prelude::Runtime;

use crate::{Error, Zip};

/// Fixed-ABI form of a logical iterator read.
///
/// This is a structural capability only: it describes how an index is
/// evaluated and how its physical leaves are staged. It deliberately contains
/// no algorithm-specific operations.
pub trait KernelInput<R: Runtime>:
    Clone
    + crate::read::ReadExpression<ReadArity = crate::A13>
    + crate::read::LowerReadExpression
    + crate::reduce::StageRead<R, crate::read::Env0>
{
}

impl<R, Input> KernelInput<R> for Input
where
    R: Runtime,
    Input: Clone
        + crate::read::ReadExpression<ReadArity = crate::A13>
        + crate::read::LowerReadExpression
        + crate::reduce::StageRead<R, crate::read::Env0>,
{
}

/// Device-side operations that follow directly from an item's physical leaf
/// layout. This trait has no algorithm dispatch methods.
pub trait KernelValue:
    Sized
    + Send
    + Sync
    + 'static
    + crate::storage::SelectLeaves
    + crate::storage::SharedLeaves
    + crate::storage::MutableLeaves
    + crate::storage::PlaneShuffleLeaves
    + crate::storage::LoadPadded12
    + crate::storage::LoadMutPadded12
    + crate::output::OutputSlotLayout<
        Slots: crate::output::OutputSlotEnvironment<StorageArity = Self::StorageArity>,
    >
{
    type StorageArity: crate::storage::StorageArity;
}

impl<Leaves> KernelValue for Leaves
where
    Leaves: Sized
        + Send
        + Sync
        + 'static
        + crate::storage::SelectLeaves
        + crate::storage::SharedLeaves
        + crate::storage::MutableLeaves
        + crate::storage::PlaneShuffleLeaves
        + crate::storage::LoadPadded12
        + crate::storage::LoadMutPadded12
        + crate::output::OutputSlotLayout,
    <Leaves as crate::output::OutputSlotLayout>::Slots: crate::output::OutputSlotEnvironment,
{
    type StorageArity = <<Leaves as crate::output::OutputSlotLayout>::Slots as crate::output::OutputSlotEnvironment>::StorageArity;
}

pub trait IterLength {
    fn logical_len(&self) -> Result<usize, Error>;
}

impl<T> IterLength for crate::read::Column<T>
where
    T: crate::MStorageElement,
{
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len())
    }
}

impl<T> IterLength for crate::read::Constant<T> {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl IterLength for crate::read::Counting {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl IterLength for crate::read::ReverseCounting {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl<Source> IterLength for crate::read::Taken<Source> {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len as usize)
    }
}

impl<Left, Right> IterLength for Zip<Left, Right>
where
    Left: IterLength,
    Right: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        let left = self.0.logical_len()?;
        let right = self.1.logical_len()?;
        if left == right {
            Ok(left)
        } else {
            Err(Error::LengthMismatch { left, right })
        }
    }
}

impl<Input, Op> IterLength for crate::read::Transform<Input, Op>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}

impl<Input, Op> IterLength for crate::read::Adjacent<Input, Op>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}

impl<Values, Indices> IterLength for crate::read::Permute<Values, Indices>
where
    Indices: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.indices.logical_len()
    }
}

impl<Values> IterLength for crate::read::Reverse<Values>
where
    Values: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        match self.len {
            Some(len) => Ok(len),
            None => self.values.logical_len(),
        }
    }
}

impl<Input, Output> IterLength for crate::read::Reassociate<Input, Output>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}

impl<Runtime, Input> IterLength for crate::read::Slice<Runtime, Input>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}
