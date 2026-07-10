//! Recursive allocation of canonical left-associated SoA storage.

use std::ops::RangeBounds;

use cubecl::prelude::Runtime;

use crate::{
    Column, DeviceSliceMut, DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression,
    S1, StorageLayout, Zip,
    output::{LowerOutputExpression, OutputExpression, StageOutput},
    read::{Env0, LowerReadExpression, Reassociate},
    reduce::StageRead,
    selection::FillOutput,
    storage::{Last, More, WriteFrom},
    transform::{MaterializeDispatch, materialize},
};

/// Owned storage that can produce read and mutable output trees.
pub trait MStorage<R: Runtime> {
    type Item: StorageLayout;
    type Read: ReadExpression<Item = Self::Item>;
    type Write: OutputExpression<Item = Self::Item>;

    fn len(&self) -> Result<usize, Error>;
    fn read(&self) -> Self::Read;
    fn write(&self) -> Self::Write;
    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read;
    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write;
}

/// Copies canonical storage into an equally shaped output tree.
#[doc(hidden)]
pub trait CopyStorage<R: Runtime>: MStorage<R> {
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error>;
}

impl<R, T> MStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>,
{
    type Item = T;
    type Read = Column<T>;
    type Write = DeviceSliceMut<T>;

    fn len(&self) -> Result<usize, Error> {
        Ok(self.len())
    }
    fn read(&self) -> Self::Read {
        self.column()
    }
    fn write(&self) -> Self::Write {
        self.slice_mut(..)
    }
    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        self.slice(range)
    }
    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write {
        self.slice_mut(range)
    }
}

impl<R, T> CopyStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: MStorageElement
        + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>
        + crate::WriteFrom<T>,
    Column<T>: ReadExpression<Item = T, ReadArity = crate::A1>
        + LowerReadExpression<Slots = crate::read::Env1<T>>
        + StageRead<R, Env0>,
    DeviceSliceMut<T>: OutputExpression<Item = T, StorageArity = S1>
        + LowerOutputExpression<Slots = crate::read::Env1<T>>
        + StageOutput<R, Env0>,
    Dispatch<crate::A1, S1>: MaterializeDispatch<
            R,
            Column<T>,
            DeviceSliceMut<T>,
            crate::read::Env1<T>,
            crate::read::Env1<T>,
        >,
{
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error> {
        materialize(exec, self.read(), output)
    }
}

impl<R, Left, Right> MStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: MStorage<R>,
    Right: MStorage<R>,
    (Left::Item, Right::Item): StorageLayout,
    Zip<Left::Read, Right::Read>: ReadExpression<Item = (Left::Item, Right::Item)>,
    Zip<Left::Write, Right::Write>: OutputExpression<Item = (Left::Item, Right::Item)>,
{
    type Item = (Left::Item, Right::Item);
    type Read = Zip<Left::Read, Right::Read>;
    type Write = Zip<Left::Write, Right::Write>;

    fn len(&self) -> Result<usize, Error> {
        let left = self.0.len()?;
        let right = self.1.len()?;
        if left != right {
            Err(Error::LengthMismatch { left, right })
        } else {
            Ok(left)
        }
    }

    fn read(&self) -> Self::Read {
        Zip::new(self.0.read(), self.1.read())
    }
    fn write(&self) -> Self::Write {
        Zip::new(self.0.write(), self.1.write())
    }

    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        let len = self.len().expect("storage columns have equal lengths");
        let (start, count) = crate::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice(start..start + count),
            self.1.slice(start..start + count),
        )
    }

    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write {
        let len = self.len().expect("storage columns have equal lengths");
        let (start, count) = crate::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice_mut(start..start + count),
            self.1.slice_mut(start..start + count),
        )
    }
}

impl<R, Left, Right> CopyStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: CopyStorage<R>,
    Right: CopyStorage<R>,
    (Left::Item, Right::Item): StorageLayout,
    Zip<Left::Read, Right::Read>: ReadExpression<Item = (Left::Item, Right::Item)>,
    Zip<Left::Write, Right::Write>: OutputExpression<Item = (Left::Item, Right::Item)>,
{
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error> {
        self.0.copy_storage(exec, output.0)?;
        self.1.copy_storage(exec, output.1)
    }
}

/// Allocates the canonical storage for a semantic item type.
pub trait MAlloc<R: Runtime>: StorageLayout {
    type Storage: MStorage<R>;
    fn alloc(exec: &Executor<R>, len: usize) -> Self::Storage;
}

#[doc(hidden)]
pub trait AllocColumns<R: Runtime> {
    type Storage: MStorage<R>;
    fn alloc_columns(exec: &Executor<R>, len: usize) -> Self::Storage;
}

impl<R, L0> AllocColumns<R> for Last<L0>
where
    R: Runtime,
    L0: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<L0>>,
{
    type Storage = DeviceVec<R, L0>;
    fn alloc_columns(exec: &Executor<R>, len: usize) -> Self::Storage {
        exec.alloc_column::<L0>(len)
    }
}

macro_rules! alloc_left {
    ($name:ident; $leaves:ty; $storage:ty; $value:expr; $( $leaf:ident ),+) => {
        impl<R, $( $leaf ),+> AllocColumns<R> for $leaves
        where
            R: Runtime,
            $( $leaf: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<$leaf>>, )+
        {
            type Storage = $storage;
            fn alloc_columns(exec: &Executor<R>, len: usize) -> Self::Storage { $value(exec, len) }
        }
    };
}

fn alloc2<R: Runtime, A: MStorageElement, B: MStorageElement>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<DeviceVec<R, A>, DeviceVec<R, B>> {
    Zip::new(exec.alloc_column::<A>(len), exec.alloc_column::<B>(len))
}
fn alloc3<R: Runtime, A: MStorageElement, B: MStorageElement, C: MStorageElement>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>> {
    Zip::new(alloc2::<R, A, B>(exec, len), exec.alloc_column::<C>(len))
}
fn alloc4<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>> {
    Zip::new(alloc3::<R, A, B, C>(exec, len), exec.alloc_column::<D>(len))
}
fn alloc5<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>>,
    DeviceVec<R, E>,
> {
    Zip::new(
        alloc4::<R, A, B, C, D>(exec, len),
        exec.alloc_column::<E>(len),
    )
}
fn alloc6<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
    F: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>>,
        DeviceVec<R, E>,
    >,
    DeviceVec<R, F>,
> {
    Zip::new(
        alloc5::<R, A, B, C, D, E>(exec, len),
        exec.alloc_column::<F>(len),
    )
}
fn alloc7<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
    F: MStorageElement,
    G: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>>,
            DeviceVec<R, E>,
        >,
        DeviceVec<R, F>,
    >,
    DeviceVec<R, G>,
> {
    Zip::new(
        alloc6::<R, A, B, C, D, E, F>(exec, len),
        exec.alloc_column::<G>(len),
    )
}

alloc_left!(A2; More<L0,Last<L1>>; Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>; alloc2::<R,L0,L1>; L0,L1);
alloc_left!(A3; More<L0,More<L1,Last<L2>>>; Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>; alloc3::<R,L0,L1,L2>; L0,L1,L2);
alloc_left!(A4; More<L0,More<L1,More<L2,Last<L3>>>>; Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>; alloc4::<R,L0,L1,L2,L3>; L0,L1,L2,L3);
alloc_left!(A5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>; alloc5::<R,L0,L1,L2,L3,L4>; L0,L1,L2,L3,L4);
alloc_left!(A6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>; alloc6::<R,L0,L1,L2,L3,L4,L5>; L0,L1,L2,L3,L4,L5);
alloc_left!(A7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>,DeviceVec<R,L6>>; alloc7::<R,L0,L1,L2,L3,L4,L5,L6>; L0,L1,L2,L3,L4,L5,L6);

impl<R, Item> MAlloc<R> for Item
where
    R: Runtime,
    Item: StorageLayout,
    Item::StorageLeaves: AllocColumns<R>,
{
    type Storage = <Item::StorageLeaves as AllocColumns<R>>::Storage;
    fn alloc(exec: &Executor<R>, len: usize) -> Self::Storage {
        Item::StorageLeaves::alloc_columns(exec, len)
    }
}

impl<R: Runtime> Executor<R> {
    /// Allocates uninitialized canonical device storage for `len` logical items.
    ///
    /// The storage must be completely written before it is read.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, fill};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let output = exec.alloc::<u32>(4);
    /// fill(&exec, 7, output.slice_mut(..)).unwrap();
    ///
    /// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7, 7]);
    /// ```
    pub fn alloc<Item: MAlloc<R>>(&self, len: usize) -> Item::Storage {
        Item::alloc(self, len)
    }

    /// Allocates canonical storage and fills every logical item with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.full::<u32>(3, 42).unwrap();
    ///
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![42, 42, 42]);
    /// ```
    pub fn full<Item>(&self, len: usize, value: Item) -> Result<Item::Storage, Error>
    where
        Item: MAlloc<R>,
        Item::Storage: MStorage<R>,
        <Item::Storage as MStorage<R>>::Item: WriteFrom<Item>,
        <Item::Storage as MStorage<R>>::Write: FillOutput<R>,
    {
        let storage = self.alloc::<Item>(len);
        let value = <Item::Storage as MStorage<R>>::Item::write_from(value);
        storage.write().fill_output(self, value)?;
        Ok(storage)
    }
}

pub(crate) fn singleton<R, Item>(exec: &Executor<R>, value: Item) -> Result<Item::Storage, Error>
where
    R: Runtime,
    Item: MAlloc<R>,
    Item::Storage: MStorage<R>,
    <Item::Storage as MStorage<R>>::Item: WriteFrom<Item>,
    <Item::Storage as MStorage<R>>::Write: FillOutput<R>,
{
    let storage = exec.alloc::<Item>(1);
    let value = <Item::Storage as MStorage<R>>::Item::write_from(value);
    storage.write().fill_output(exec, value)?;
    Ok(storage)
}

/// Normalizes an arbitrary read expression into canonical left-associated
/// storage without changing its semantic item type.
///
/// Consumers that need to add another read leaf (for example a permutation)
/// use this boundary to turn an `A8` expression back into at most seven
/// physical payload leaves.  The semantic view is recovered with
/// [`Reassociate`], so input tuple association is not leaked into storage.
#[doc(hidden)]
pub trait NormalizeInput<R: Runtime>: ReadExpression + Sized {
    type Storage: MStorage<R>;
    type SemanticRead: ReadExpression<Item = Self::Item>;

    fn normalize(self, exec: &Executor<R>) -> Result<Self::Storage, Error>;
    fn semantic_read(storage: &Self::Storage) -> Self::SemanticRead;
}

/// Normalizes an input after a single semantic prefix item.
#[doc(hidden)]
pub trait PrependInput<R: Runtime>: NormalizeInput<R> {
    fn prepend(self, exec: &Executor<R>, prefix: Self::Item) -> Result<Self::Storage, Error>;
}

impl<R, Input> PrependInput<R> for Input
where
    R: Runtime,
    Input: NormalizeInput<R>,
    Input::Item: MAlloc<R, Storage = Input::Storage>,
    Input::Storage: CopyStorage<R>,
    <Input::Storage as MStorage<R>>::Item: crate::WriteFrom<Input::Item>,
    <Input::Storage as MStorage<R>>::Write: crate::selection::FillOutput<R>,
{
    fn prepend(self, exec: &Executor<R>, prefix: Self::Item) -> Result<Self::Storage, Error> {
        let values = self.normalize(exec)?;
        let len = values.len()?;
        let prefixed = exec.alloc::<Input::Item>(len + 1);
        let prefix = <Input::Storage as MStorage<R>>::Item::write_from(prefix);
        prefixed.slice_mut(..1).fill_output(exec, prefix)?;
        values.copy_storage(exec, prefixed.slice_mut(1..))?;
        Ok(prefixed)
    }
}

impl<R, Input> NormalizeInput<R> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: MAlloc<R>
        + crate::WriteFrom<
            <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Item,
        >,
    <Input::Item as MAlloc<R>>::Storage: MStorage<R>,
    <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write:
        LowerOutputExpression + StageOutput<R, Env0>,
    <<<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as OutputExpression>::Item:
        crate::WriteFrom<Input::Item>,
    Dispatch<
        Input::ReadArity,
        <<<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as OutputExpression>::StorageArity,
    >: MaterializeDispatch<
            R,
            Input,
            <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write,
            Input::Slots,
            <<<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as LowerOutputExpression>::Slots,
        >,
    Reassociate<
        <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Read,
        Input::Item,
    >: ReadExpression<Item = Input::Item>,
{
    type Storage = <Input::Item as MAlloc<R>>::Storage;
    type SemanticRead = Reassociate<
        <<Input::Item as MAlloc<R>>::Storage as MStorage<R>>::Read,
        Input::Item,
    >;

    fn normalize(self, exec: &Executor<R>) -> Result<Self::Storage, Error> {
        let len = self.logical_len()?;
        let storage = exec.alloc::<Input::Item>(len);
        materialize(exec, self, storage.write())?;
        Ok(storage)
    }

    fn semantic_read(storage: &Self::Storage) -> Self::SemanticRead {
        Reassociate::new(storage.read())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn alloc_normalizes_right_associated_item_to_left_storage() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0]);
        let c = exec.to_device(&[-1_i32, -2, -3]);
        let storage = exec.alloc::<(u32, (f32, i32))>(3);

        materialize(
            &exec,
            Zip::new(a.column(), Zip::new(b.column(), c.column())),
            storage.write(),
        )
        .unwrap();

        assert_eq!(exec.to_host(&storage.0.0).unwrap(), vec![1, 2, 3]);
        assert_eq!(exec.to_host(&storage.0.1).unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(exec.to_host(&storage.1).unwrap(), vec![-1, -2, -3]);
        let sliced = storage.slice(1..);
        assert_eq!(sliced.0.0.len(), 2);
    }
}
