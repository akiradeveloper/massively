use cubecl::prelude::Runtime;

use crate::{Error, Executor, MAlloc, MIndex, MStorage, MVec, op::UnaryOp};

/// One device-resident logical value.
///
/// An `MVal` owns one row of ordinary Massively storage. Tuple values therefore
/// use the same structure-of-arrays layout as tuple vectors; they are not a zip
/// of independently exposed scalar handles. Constructing, mapping, and passing
/// an `MVal` to an algorithm do not copy it to the host; [`MVal::read`] is the
/// explicit synchronization boundary.
#[allow(private_bounds)]
pub struct MVal<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    pub(crate) storage: MVec<R, T>,
}

impl<R, T> Clone for MVal<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
    MVec<R, T>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
    }
}

impl<R, T> MVal<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    pub(crate) fn from_storage(storage: MVec<R, T>) -> Result<Self, Error> {
        let len = storage.capacity()?;
        if len != 1 {
            return Err(Error::LengthMismatch {
                left: len as usize,
                right: 1,
            });
        }
        Ok(Self { storage })
    }

    pub(crate) fn into_storage(self) -> MVec<R, T> {
        self.storage
    }

    pub(crate) fn scratch_storage(&self) -> &<T as crate::allocation::ScratchStorage<R>>::Storage
    where
        T: crate::allocation::ScratchStorage<R>,
    {
        <<T as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::scratch_ref(
            &self.storage,
        )
    }

    pub(crate) fn into_scratch_storage(self) -> <T as crate::allocation::ScratchStorage<R>>::Storage
    where
        T: crate::allocation::ScratchStorage<R>,
    {
        <<T as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::into_scratch(
            self.storage,
        )
    }

    /// Borrows this value as a one-item device iterator.
    pub fn as_iter(&self) -> <MVec<R, T> as MStorage<R>>::Slice<'_> {
        self.storage.slice(..)
    }

    /// Applies a GPU operation to this value without copying it to the host.
    pub fn map<Op>(&self, exec: &Executor<R>, op: Op) -> Result<MVal<R, Op::Output>, Error>
    where
        Op: UnaryOp<T>,
        Op::Output: MAlloc<R>,
    {
        MVal::from_storage(crate::vector::transform(exec, self.as_iter(), op)?)
    }

    /// Repeats this device value as a lazy device iterator.
    pub fn repeat(
        &self,
        len: MIndex,
    ) -> crate::lazy::Permute<
        <MVec<R, T> as MStorage<R>>::Slice<'_>,
        crate::lazy::Taken<crate::lazy::Constant<u32>>,
    > {
        crate::lazy::permute(self.as_iter(), crate::lazy::constant(0u32).take(len))
    }

    /// Explicitly copies this value to the host and waits for its producers.
    pub fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        <<T as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::read_value(
            exec,
            &self.storage,
        )
    }
}

impl<R: Runtime> Executor<R> {
    /// Uploads one host value into a device-resident [`MVal`].
    pub fn value<T>(&self, value: T) -> Result<MVal<R, T>, Error>
    where
        T: MAlloc<R>,
    {
        MVal::from_storage(
            <<T as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::store_value(
                self, value,
            )?,
        )
    }
}

#[allow(private_bounds)]
impl<R> MVal<R, MIndex>
where
    R: Runtime,
    MIndex: crate::allocation::ScratchStorage<R, Storage = crate::DeviceVec<R, MIndex>>,
{
    pub(crate) fn logical_extent(&self, upper_bound: usize) -> crate::extent::LogicalExtent {
        crate::extent::LogicalExtent::from_device(self.scratch_storage(), upper_bound)
    }
}
