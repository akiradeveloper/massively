//! Recursive allocation of canonical left-associated SoA storage.

use std::ops::RangeBounds;

use cubecl::prelude::{CubeType, Runtime};

use crate::{
    Column, DeviceSliceMut, DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression,
    S1, StorageLayout, Zip,
    api::iter::{Allocable, Canonicalizable, MStorage, Materializable},
    output::{
        LowerOutputExpression, OutputExpression, PaddedOutputSlots, SliceOutput, StageOutput,
    },
    read::{Env0, LowerReadExpression, SlotEnvironment},
    reduce::StageRead,
    selection::FillOutput,
    storage::{Last, More, WritableFrom},
    transform::{MaterializeDispatch, materialize},
};

#[doc(hidden)]
pub trait CanonicalLeaves {
    type Item;
}

#[doc(hidden)]
pub trait FoldCanonical<Accumulator> {
    type Item;
}

impl<Item: CubeType> CanonicalLeaves for Last<Item> {
    type Item = Item;
}

impl<Head, Tail> CanonicalLeaves for More<Head, Tail>
where
    Head: CubeType,
    Tail: CubeType + FoldCanonical<Head>,
{
    type Item = Tail::Item;
}

impl<Accumulator, Item: CubeType> FoldCanonical<Accumulator> for Last<Item> {
    type Item = (Accumulator, Item);
}

impl<Accumulator, Head, Tail> FoldCanonical<Accumulator> for More<Head, Tail>
where
    Head: CubeType,
    Tail: CubeType + FoldCanonical<(Accumulator, Head)>,
{
    type Item = Tail::Item;
}

/// Owned storage that can produce read and mutable output trees.
pub trait CanonicalStorage<R: Runtime> {
    type Item: StorageLayout;
    type ReadSlots: SlotEnvironment + crate::read::PaddedReadSlots;
    type WriteSlots: PaddedOutputSlots<Leaves = <Self::Item as StorageLayout>::StorageLeaves>;
    type Read: ReadExpression<Item = Self::Item>
        + LowerReadExpression<Slots = Self::ReadSlots>
        + StageRead<R, Env0>
        + Clone;
    type Write: OutputExpression<Item = Self::Item>
        + LowerOutputExpression<Slots = Self::WriteSlots>
        + StageOutput<R, Env0>
        + SliceOutput
        + FillOutput<R>;

    fn len(&self) -> Result<usize, Error>;
    fn truncate(&mut self, len: usize);
    fn read(&self) -> Self::Read;
    fn write(&self) -> Self::Write;
    fn read_first(&self, exec: &Executor<R>) -> Result<Self::Item, Error>;
    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read;
    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write;
    fn read_item(&self, exec: &Executor<R>, index: usize) -> Result<Self::Item, Error>;
}

/// Copies canonical storage into an equally shaped output tree.
#[doc(hidden)]
pub trait CopyStorage<R: Runtime>: CanonicalStorage<R> {
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error>;
}

impl<R, T> CanonicalStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>,
{
    type Item = T;
    type ReadSlots = crate::read::Env1<T>;
    type WriteSlots = crate::read::Env1<T>;
    type Read = Column<T>;
    type Write = DeviceSliceMut<T>;

    fn len(&self) -> Result<usize, Error> {
        Ok(self.len())
    }
    fn truncate(&mut self, len: usize) {
        DeviceVec::truncate(self, len);
    }
    fn read(&self) -> Self::Read {
        self.column()
    }
    fn write(&self) -> Self::Write {
        self.slice_mut(..)
    }
    fn read_first(&self, exec: &Executor<R>) -> Result<Self::Item, Error> {
        exec.to_host(self)?
            .into_iter()
            .next()
            .ok_or(Error::LengthMismatch { left: 0, right: 1 })
    }
    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        self.slice(range)
    }
    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write {
        self.slice_mut(range)
    }

    fn read_item(&self, exec: &Executor<R>, index: usize) -> Result<Self::Item, Error> {
        if index >= self.len() {
            return Err(Error::LengthMismatch {
                left: index + 1,
                right: self.len(),
            });
        }
        Ok(exec.to_host(&self.slice(index..index + 1))?[0])
    }
}

impl<R, T> CopyStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: MStorageElement
        + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>
        + crate::WritableFrom<T>,
    Column<T>: ReadExpression<Item = T, ReadArity = crate::A1>
        + LowerReadExpression<Slots = crate::read::Env1<T>>
        + StageRead<R, Env0>,
    DeviceSliceMut<T>: OutputExpression<Item = T, StorageArity = S1>
        + LowerOutputExpression<Slots = crate::read::Env1<T>>
        + StageOutput<R, Env0>,
    Dispatch<crate::A13, crate::S12>: MaterializeDispatch<
            R,
            Column<T>,
            DeviceSliceMut<T>,
            crate::read::KernelReadSlots<crate::read::Env1<T>>,
            crate::output::KernelOutputSlots<crate::read::Env1<T>>,
        >,
{
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error> {
        materialize(exec, self.read(), output)
    }
}

impl<R, Left, Right> CanonicalStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: CanonicalStorage<R>,
    Right: CanonicalStorage<R>,
    (Left::Item, Right::Item): StorageLayout,
    Zip<Left::Read, Right::Read>:
        ReadExpression<Item = (Left::Item, Right::Item)> + LowerReadExpression + StageRead<R, Env0>,
    Zip<Left::Write, Right::Write>: OutputExpression<Item = (Left::Item, Right::Item)>
        + LowerOutputExpression
        + StageOutput<R, Env0>
        + SliceOutput
        + FillOutput<R>,
    <Zip<Left::Write, Right::Write> as LowerOutputExpression>::Slots:
        PaddedOutputSlots<Leaves = <(Left::Item, Right::Item) as StorageLayout>::StorageLeaves>,
{
    type Item = (Left::Item, Right::Item);
    type ReadSlots = <Zip<Left::Read, Right::Read> as LowerReadExpression>::Slots;
    type WriteSlots = <Zip<Left::Write, Right::Write> as LowerOutputExpression>::Slots;
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

    fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
        self.1.truncate(len);
    }

    fn read(&self) -> Self::Read {
        Zip::new(self.0.read(), self.1.read())
    }
    fn write(&self) -> Self::Write {
        Zip::new(self.0.write(), self.1.write())
    }
    fn read_first(&self, exec: &Executor<R>) -> Result<Self::Item, Error> {
        Ok((self.0.read_first(exec)?, self.1.read_first(exec)?))
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

    fn read_item(&self, exec: &Executor<R>, index: usize) -> Result<Self::Item, Error> {
        Ok((
            self.0.read_item(exec, index)?,
            self.1.read_item(exec, index)?,
        ))
    }
}

impl<R, Left, Right> CopyStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: CopyStorage<R>,
    Right: CopyStorage<R>,
    (Left::Item, Right::Item): StorageLayout,
    Zip<Left::Read, Right::Read>:
        ReadExpression<Item = (Left::Item, Right::Item)> + LowerReadExpression + StageRead<R, Env0>,
    Zip<Left::Write, Right::Write>: OutputExpression<Item = (Left::Item, Right::Item)>
        + LowerOutputExpression
        + StageOutput<R, Env0>
        + SliceOutput
        + FillOutput<R>,
    <Zip<Left::Write, Right::Write> as LowerOutputExpression>::Slots:
        PaddedOutputSlots<Leaves = <(Left::Item, Right::Item) as StorageLayout>::StorageLeaves>,
{
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error> {
        self.0.copy_storage(exec, output.0)?;
        self.1.copy_storage(exec, output.1)
    }
}

/// Allocates the canonical storage for a semantic item type.
pub trait CanonicalAlloc<R: Runtime>: StorageLayout + WritableFrom<Self::CanonicalItem> {
    type CanonicalItem: StorageLayout<StorageArity = Self::StorageArity, StorageLeaves = Self::StorageLeaves>
        + WritableFrom<Self>;
    type CanonicalStorage: CanonicalStorage<R, Item = Self::CanonicalItem> + CopyStorage<R>;
    fn alloc(exec: &Executor<R>, len: usize) -> Self::CanonicalStorage;
}

#[doc(hidden)]
pub trait AllocColumns<R: Runtime> {
    type Storage: CanonicalStorage<R>;
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

fn alloc8<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                    DeviceVec<R, L3>,
                >,
                DeviceVec<R, L4>,
            >,
            DeviceVec<R, L5>,
        >,
        DeviceVec<R, L6>,
    >,
    DeviceVec<R, L7>,
> {
    Zip::new(
        alloc7::<R, L0, L1, L2, L3, L4, L5, L6>(exec, len),
        exec.alloc_column::<L7>(len),
    )
}

fn alloc9<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                        DeviceVec<R, L3>,
                    >,
                    DeviceVec<R, L4>,
                >,
                DeviceVec<R, L5>,
            >,
            DeviceVec<R, L6>,
        >,
        DeviceVec<R, L7>,
    >,
    DeviceVec<R, L8>,
> {
    Zip::new(
        alloc8::<R, L0, L1, L2, L3, L4, L5, L6, L7>(exec, len),
        exec.alloc_column::<L8>(len),
    )
}

fn alloc10<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
    L9: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<
                            Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                            DeviceVec<R, L3>,
                        >,
                        DeviceVec<R, L4>,
                    >,
                    DeviceVec<R, L5>,
                >,
                DeviceVec<R, L6>,
            >,
            DeviceVec<R, L7>,
        >,
        DeviceVec<R, L8>,
    >,
    DeviceVec<R, L9>,
> {
    Zip::new(
        alloc9::<R, L0, L1, L2, L3, L4, L5, L6, L7, L8>(exec, len),
        exec.alloc_column::<L9>(len),
    )
}

fn alloc11<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
    L9: MStorageElement,
    L10: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<
                            Zip<
                                Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                                DeviceVec<R, L3>,
                            >,
                            DeviceVec<R, L4>,
                        >,
                        DeviceVec<R, L5>,
                    >,
                    DeviceVec<R, L6>,
                >,
                DeviceVec<R, L7>,
            >,
            DeviceVec<R, L8>,
        >,
        DeviceVec<R, L9>,
    >,
    DeviceVec<R, L10>,
> {
    Zip::new(
        alloc10::<R, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9>(exec, len),
        exec.alloc_column::<L10>(len),
    )
}

fn alloc12<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
    L9: MStorageElement,
    L10: MStorageElement,
    L11: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<
                            Zip<
                                Zip<
                                    Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                                    DeviceVec<R, L3>,
                                >,
                                DeviceVec<R, L4>,
                            >,
                            DeviceVec<R, L5>,
                        >,
                        DeviceVec<R, L6>,
                    >,
                    DeviceVec<R, L7>,
                >,
                DeviceVec<R, L8>,
            >,
            DeviceVec<R, L9>,
        >,
        DeviceVec<R, L10>,
    >,
    DeviceVec<R, L11>,
> {
    Zip::new(
        alloc11::<R, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10>(exec, len),
        exec.alloc_column::<L11>(len),
    )
}

alloc_left!(A2; More<L0,Last<L1>>; Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>; alloc2::<R,L0,L1>; L0,L1);
alloc_left!(A3; More<L0,More<L1,Last<L2>>>; Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>; alloc3::<R,L0,L1,L2>; L0,L1,L2);
alloc_left!(A4; More<L0,More<L1,More<L2,Last<L3>>>>; Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>; alloc4::<R,L0,L1,L2,L3>; L0,L1,L2,L3);
alloc_left!(A5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>; alloc5::<R,L0,L1,L2,L3,L4>; L0,L1,L2,L3,L4);
alloc_left!(A6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>; alloc6::<R,L0,L1,L2,L3,L4,L5>; L0,L1,L2,L3,L4,L5);
alloc_left!(A7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>,DeviceVec<R,L6>>; alloc7::<R,L0,L1,L2,L3,L4,L5,L6>; L0,L1,L2,L3,L4,L5,L6);
alloc_left!(A8; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,Last<L7>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>; alloc8::<R,L0,L1,L2,L3,L4,L5,L6,L7>; L0,L1,L2,L3,L4,L5,L6,L7);
alloc_left!(A9; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,Last<L8>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>; alloc9::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8>; L0,L1,L2,L3,L4,L5,L6,L7,L8);
alloc_left!(A10; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,Last<L9>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>; alloc10::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9);
alloc_left!(A11; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,Last<L10>>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>, DeviceVec<R, L10>>; alloc11::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10);
alloc_left!(A12; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,More<L10,Last<L11>>>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>, DeviceVec<R, L10>>, DeviceVec<R, L11>>; alloc12::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11);

macro_rules! impl_scalar_canonical_alloc {
    ($($item:ty),+ $(,)?) => {
        $(
            impl<R: Runtime> CanonicalAlloc<R> for $item {
                type CanonicalItem = $item;
                type CanonicalStorage = DeviceVec<R, $item>;

                fn alloc(exec: &Executor<R>, len: usize) -> Self::CanonicalStorage {
                    exec.alloc_column::<$item>(len)
                }
            }
        )+
    };
}

impl_scalar_canonical_alloc!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl<R, Left, Right> CanonicalAlloc<R> for (Left, Right)
where
    R: Runtime,
    Left: StorageLayout,
    Right: StorageLayout,
    (Left, Right): StorageLayout
        + WritableFrom<
            <<<(Left, Right) as StorageLayout>::StorageLeaves as AllocColumns<R>>::Storage as CanonicalStorage<R>>::Item,
        >,
    <(Left, Right) as StorageLayout>::StorageLeaves: AllocColumns<R>,
    <<(Left, Right) as StorageLayout>::StorageLeaves as AllocColumns<R>>::Storage: CopyStorage<R>,
    <<<(Left, Right) as StorageLayout>::StorageLeaves as AllocColumns<R>>::Storage as CanonicalStorage<R>>::Item:
        WritableFrom<(Left, Right)>
            + StorageLayout<
                StorageArity = <(Left, Right) as StorageLayout>::StorageArity,
                StorageLeaves = <(Left, Right) as StorageLayout>::StorageLeaves,
            >,
{
    type CanonicalItem =
        <<<(Left, Right) as StorageLayout>::StorageLeaves as AllocColumns<R>>::Storage as CanonicalStorage<R>>::Item;
    type CanonicalStorage =
        <<(Left, Right) as StorageLayout>::StorageLeaves as AllocColumns<R>>::Storage;

    fn alloc(exec: &Executor<R>, len: usize) -> Self::CanonicalStorage {
        <(Left, Right) as StorageLayout>::StorageLeaves::alloc_columns(exec, len)
    }
}

impl<Item> Canonicalizable for Item
where
    Item: StorageLayout,
    Item::StorageLeaves: CanonicalLeaves,
    <Item::StorageLeaves as CanonicalLeaves>::Item: CubeType,
{
    type Canonical = <Item::StorageLeaves as CanonicalLeaves>::Item;
}

impl<R, Item> Materializable<R> for Item
where
    R: Runtime,
    Item: crate::api::iter::MItem<R> + Canonicalizable,
    Item::Canonical: Allocable<R> + crate::WritableFrom<Item>,
    <Item::Canonical as Allocable<R>>::Storage: MStorage<R, Item = Item::Canonical>,
{
    type Materialized = Item::Canonical;
    type Storage = <Item::Canonical as Allocable<R>>::Storage;
}

#[doc(hidden)]
impl<R, Item> Allocable<R> for Item
where
    R: Runtime,
    Item: crate::api::iter::MItem<R>
        + MStorageElement
        + StorageLayout<StorageArity = S1, StorageLeaves = Last<Item>>,
    Last<Item>: crate::core::facade::KernelValue,
{
    type Storage = DeviceVec<R, Item>;

    fn alloc(exec: &Executor<R>, len: usize) -> <Self as Allocable<R>>::Storage {
        exec.alloc_column::<Item>(len)
    }
}

macro_rules! impl_public_alloc {
    ($item:ty; $arity:ty; $leaves:ty; $storage:ty; $value:expr; $( $leaf:ident ),+) => {
        #[doc(hidden)]
        impl<R, $( $leaf ),+> Allocable<R> for $item
        where
            R: Runtime,
            $(
                $leaf: MStorageElement
                    + CanonicalAlloc<
                        R,
                        CanonicalItem = $leaf,
                        CanonicalStorage = DeviceVec<R, $leaf>,
                    >,
            )+
            $item: crate::api::iter::MItem<R>
                + StorageLayout<StorageArity = $arity, StorageLeaves = $leaves>,
            $leaves: crate::core::facade::KernelValue,
        {
            type Storage = $storage;

            fn alloc(exec: &Executor<R>, len: usize) -> <Self as Allocable<R>>::Storage {
                $value(exec, len)
            }
        }
    };
}

impl_public_alloc!((L0,L1); crate::S2; More<L0,Last<L1>>; Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>; alloc2::<R,L0,L1>; L0,L1);
impl_public_alloc!(((L0,L1),L2); crate::S3; More<L0,More<L1,Last<L2>>>; Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>; alloc3::<R,L0,L1,L2>; L0,L1,L2);
impl_public_alloc!((((L0,L1),L2),L3); crate::S4; More<L0,More<L1,More<L2,Last<L3>>>>; Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>; alloc4::<R,L0,L1,L2,L3>; L0,L1,L2,L3);
impl_public_alloc!(((((L0,L1),L2),L3),L4); crate::S5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>; alloc5::<R,L0,L1,L2,L3,L4>; L0,L1,L2,L3,L4);
impl_public_alloc!((((((L0,L1),L2),L3),L4),L5); crate::S6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>; alloc6::<R,L0,L1,L2,L3,L4,L5>; L0,L1,L2,L3,L4,L5);
impl_public_alloc!(((((((L0,L1),L2),L3),L4),L5),L6); crate::S7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>,DeviceVec<R,L6>>; alloc7::<R,L0,L1,L2,L3,L4,L5,L6>; L0,L1,L2,L3,L4,L5,L6);
impl_public_alloc!((((((((L0,L1),L2),L3),L4),L5),L6),L7); crate::S8; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,Last<L7>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>; alloc8::<R,L0,L1,L2,L3,L4,L5,L6,L7>; L0,L1,L2,L3,L4,L5,L6,L7);
impl_public_alloc!(((((((((L0,L1),L2),L3),L4),L5),L6),L7),L8); crate::S9; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,Last<L8>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>; alloc9::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8>; L0,L1,L2,L3,L4,L5,L6,L7,L8);
impl_public_alloc!((((((((((L0,L1),L2),L3),L4),L5),L6),L7),L8),L9); crate::S10; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,Last<L9>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>; alloc10::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9);
impl_public_alloc!(((((((((((L0,L1),L2),L3),L4),L5),L6),L7),L8),L9),L10); crate::S11; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,Last<L10>>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>, DeviceVec<R, L10>>; alloc11::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10);
impl_public_alloc!((((((((((((L0,L1),L2),L3),L4),L5),L6),L7),L8),L9),L10),L11); crate::S12; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,More<L10,Last<L11>>>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>, DeviceVec<R, L10>>, DeviceVec<R, L11>>; alloc12::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11);

#[doc(hidden)]
impl<R, T> MStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: Allocable<R, Storage = Self>
        + MStorageElement
        + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>,
    Last<T>: crate::core::facade::KernelValue,
    Column<T>: crate::api::iter::MIter<R, Item = T>,
    DeviceSliceMut<T>: crate::api::iter::MIterMut<R, Item = T>,
{
    type Item = T;

    fn len(&self) -> Result<crate::MIndex, Error> {
        crate::MIndex::try_from(self.len()).map_err(|_| Error::LengthTooLarge { len: self.len() })
    }

    fn truncate(&mut self, len: crate::MIndex) {
        DeviceVec::truncate(self, len as usize);
    }

    fn slice<Bounds>(
        &self,
        range: Bounds,
    ) -> impl crate::api::iter::MIter<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let len = crate::MIndex::try_from(self.len()).expect("storage length exceeds MIndex");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        self.slice(start..start + count)
    }

    fn slice_mut<Bounds>(
        &self,
        range: Bounds,
    ) -> impl crate::api::iter::MIterMut<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let len = crate::MIndex::try_from(self.len()).expect("storage length exceeds MIndex");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        self.slice_mut(start..start + count)
    }
}

#[doc(hidden)]
impl<R, Left, Right> MStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: MStorage<R>,
    Right: MStorage<R>,
    Self: CanonicalStorage<R>,
    <Self as CanonicalStorage<R>>::Item: Allocable<R, Storage = Self>,
    <<Self as CanonicalStorage<R>>::Item as StorageLayout>::StorageLeaves:
        crate::core::facade::KernelValue,
    <Self as CanonicalStorage<R>>::Read:
        crate::api::iter::MIter<R, Item = <Self as CanonicalStorage<R>>::Item>,
    <Self as CanonicalStorage<R>>::Write:
        crate::api::iter::MIterMut<R, Item = <Self as CanonicalStorage<R>>::Item>,
{
    type Item = <Self as CanonicalStorage<R>>::Item;

    fn len(&self) -> Result<crate::MIndex, Error> {
        let len = CanonicalStorage::len(self)?;
        crate::MIndex::try_from(len).map_err(|_| Error::LengthTooLarge { len })
    }

    fn truncate(&mut self, len: crate::MIndex) {
        self.0.truncate(len);
        self.1.truncate(len);
    }

    fn slice<Bounds>(
        &self,
        range: Bounds,
    ) -> impl crate::api::iter::MIter<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let len = MStorage::len(self).expect("storage columns have equal lengths");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        CanonicalStorage::slice(self, start..start + count)
    }

    fn slice_mut<Bounds>(
        &self,
        range: Bounds,
    ) -> impl crate::api::iter::MIterMut<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let len = MStorage::len(self).expect("storage columns have equal lengths");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        CanonicalStorage::slice_mut(self, start..start + count)
    }
}

impl<R: Runtime> Executor<R> {
    /// Allocates canonical owned device storage for a semantic item type.
    ///
    /// Unlike [`Self::alloc`], this method accepts any supported tuple nesting
    /// and normalizes the returned [`crate::MVec`] to a left-associated SoA.
    pub fn alloc_mvec<Item>(&self, len: usize) -> crate::MVec<R, Item>
    where
        Item: Materializable<R>,
    {
        <Item::Materialized as Allocable<R>>::alloc(self, len)
    }

    /// Allocates uninitialized canonical device storage for `len` logical items.
    ///
    /// The storage must be completely written before it is read.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, lazy, vector::{fill, scatter}};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let output = exec.alloc::<u32>(4);
    /// let values = fill(&exec, 4, 7_u32).unwrap();
    /// scatter(
    ///     &exec,
    ///     values.slice(..),
    ///     lazy::counting(0).take(4),
    ///     output.slice_mut(..),
    /// ).unwrap();
    ///
    /// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7, 7]);
    /// ```
    pub fn alloc<Item: Allocable<R>>(&self, len: usize) -> <Item as Allocable<R>>::Storage {
        <Item as Allocable<R>>::alloc(self, len)
    }

    pub(crate) fn alloc_canonical<Item: CanonicalAlloc<R>>(
        &self,
        len: usize,
    ) -> <Item as CanonicalAlloc<R>>::CanonicalStorage {
        <Item as CanonicalAlloc<R>>::alloc(self, len)
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
    pub fn full<Item>(
        &self,
        len: usize,
        value: Item,
    ) -> Result<<Item as Allocable<R>>::Storage, Error>
    where
        Item: Allocable<R>,
        <Item as Allocable<R>>::Storage: CanonicalStorage<R>,
        <<Item as Allocable<R>>::Storage as CanonicalStorage<R>>::Item: WritableFrom<Item>,
        <<Item as Allocable<R>>::Storage as CanonicalStorage<R>>::Write: FillOutput<R>,
    {
        let storage = self.alloc::<Item>(len);
        let value =
            <<Item as Allocable<R>>::Storage as CanonicalStorage<R>>::Item::write_from(value);
        storage.write().fill_output(self, value)?;
        Ok(storage)
    }
}

pub(crate) fn singleton<R, Item>(
    exec: &Executor<R>,
    value: Item,
) -> Result<<Item as CanonicalAlloc<R>>::CanonicalStorage, Error>
where
    R: Runtime,
    Item: CanonicalAlloc<R>,
    <Item as CanonicalAlloc<R>>::CanonicalStorage: CanonicalStorage<R>,
    <<Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Item:
        WritableFrom<Item>,
    <<Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Write: FillOutput<R>,
{
    let storage = exec.alloc_canonical::<Item>(1);
    let value =
        <<Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Item::write_from(
            value,
        );
    storage.write().fill_output(exec, value)?;
    Ok(storage)
}

/// Normalizes an arbitrary read expression into canonical left-associated
/// storage without changing its semantic item type.
///
/// Consumers that need to add another read leaf (for example a permutation)
/// use this boundary to turn an `A13` expression back into at most twelve
/// physical payload leaves.  The semantic view is recovered with
/// [`Reassociate`], so input tuple association is not leaked into storage.
#[doc(hidden)]
pub trait NormalizeInput<R: Runtime>: ReadExpression + Sized {
    type Storage: CanonicalStorage<R>;
    type SemanticRead: ReadExpression<Item = Self::Item> + LowerReadExpression + StageRead<R, Env0>;

    fn normalize(self, exec: &Executor<R>) -> Result<Self::Storage, Error>;

    fn normalize_canonical(
        self,
        exec: &Executor<R>,
    ) -> Result<<Self::Item as CanonicalAlloc<R>>::CanonicalStorage, Error>
    where
        Self::Item: CanonicalAlloc<R>;

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
    Input::Item: CanonicalAlloc<R, CanonicalStorage = Input::Storage>,
    Input::Storage: CopyStorage<R>,
    <Input::Storage as CanonicalStorage<R>>::Item: crate::WritableFrom<Input::Item>,
    <Input::Storage as CanonicalStorage<R>>::Write: crate::selection::FillOutput<R>,
{
    fn prepend(self, exec: &Executor<R>, prefix: Self::Item) -> Result<Self::Storage, Error> {
        let values = self.normalize(exec)?;
        let len = values.len()?;
        let prefixed = exec.alloc_canonical::<Input::Item>(len + 1);
        let prefix = <Input::Storage as CanonicalStorage<R>>::Item::write_from(prefix);
        prefixed.slice_mut(..1).fill_output(exec, prefix)?;
        values.copy_storage(exec, prefixed.slice_mut(1..))?;
        Ok(prefixed)
    }
}

impl<R, Input> NormalizeInput<R> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: CanonicalAlloc<R>
        + crate::WritableFrom<
            <<Input::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Item,
        >,
    <Input::Item as StorageLayout>::StorageLeaves: crate::storage::StorePadded12,
    <Input::Item as CanonicalAlloc<R>>::CanonicalStorage: CanonicalStorage<R>,
    <<<Input::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Write as OutputExpression>::Item:
        crate::WritableFrom<Input::Item>,
    Dispatch<crate::A13, crate::S12>: MaterializeDispatch<
            R,
            Input,
            <<Input::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Write,
            crate::read::KernelReadSlots<Input::Slots>,
            crate::output::KernelOutputSlots<
                <<Input::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::WriteSlots,
            >,
        >,
{
    type Storage = <Input::Item as CanonicalAlloc<R>>::CanonicalStorage;
    type SemanticRead = crate::read::FixedReassociate<
        <<Input::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Read,
        Input::Item,
    >;

    fn normalize(self, exec: &Executor<R>) -> Result<Self::Storage, Error> {
        let len = self.logical_len()?;
        let storage = exec.alloc_canonical::<Input::Item>(len);
        materialize(exec, self, storage.write())?;
        Ok(storage)
    }

    fn normalize_canonical(
        self,
        exec: &Executor<R>,
    ) -> Result<<Self::Item as CanonicalAlloc<R>>::CanonicalStorage, Error>
    where
        Self::Item: CanonicalAlloc<R>,
    {
        self.normalize(exec)
    }

    fn semantic_read(storage: &Self::Storage) -> Self::SemanticRead {
        crate::read::FixedReassociate::new(storage.read())
    }
}

/// Materializes a fixed-ABI read expression into its canonical owned storage.
///
/// Unlike [`NormalizeInput`], this path derives the output layout from the
/// semantic item's physical leaves.  It therefore needs no arity-specific
/// algorithm capability on the iterator type.
pub(crate) fn normalize_lowered<R, Input>(
    exec: &Executor<R>,
    input: Input,
) -> Result<<Input::Item as CanonicalAlloc<R>>::CanonicalStorage, Error>
where
    R: Runtime,
    Input: crate::core::facade::KernelInput<R>,
    Input::Item: crate::api::iter::MItem<R>,
    <Input::Item as StorageLayout>::StorageLeaves: crate::core::facade::KernelValue,
    <Input::Item as CanonicalAlloc<R>>::CanonicalStorage: CanonicalStorage<R>,
    <<Input::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Item:
        crate::WritableFrom<Input::Item>,
{
    let len = input.logical_len()?;
    let storage = exec.alloc_canonical::<Input::Item>(len);
    let output = crate::output::ReassociatedOutput::<
        _,
        Input::Item,
        <<Input::Item as StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots,
    >::new(storage.write());
    crate::transform::materialize_fixed(exec, &input, &output)?;
    Ok(storage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn canonical_shape_is_runtime_independent() {
        fn assert_canonical<Item, Canonical>()
        where
            Item: Canonicalizable<Canonical = Canonical>,
            Canonical: CubeType,
        {
        }

        assert_canonical::<u32, u32>();
        assert_canonical::<(u32, (f32, i32)), ((u32, f32), i32)>();
    }

    #[test]
    fn alloc_normalizes_right_associated_item_to_left_storage() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0]);
        let c = exec.to_device(&[-1_i32, -2, -3]);
        let storage = exec.alloc_canonical::<(u32, (f32, i32))>(3);

        materialize(
            &exec,
            Zip::new(a.column(), Zip::new(b.column(), c.column())),
            storage.write(),
        )
        .unwrap();

        assert_eq!(exec.to_host(&storage.0.0).unwrap(), vec![1, 2, 3]);
        assert_eq!(exec.to_host(&storage.0.1).unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(exec.to_host(&storage.1).unwrap(), vec![-1, -2, -3]);
        let sliced = CanonicalStorage::slice(&storage, 1..);
        assert_eq!(sliced.0.0.len(), 2);
    }

    #[test]
    fn mvec_maps_semantic_shape_and_truncates_every_column() {
        type Semantic = (u32, (f32, i32));

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let mut storage: crate::MVec<WgpuRuntime, Semantic> = exec.alloc_mvec::<Semantic>(4);

        let _: &Zip<
            Zip<DeviceVec<WgpuRuntime, u32>, DeviceVec<WgpuRuntime, f32>>,
            DeviceVec<WgpuRuntime, i32>,
        > = &storage;
        MStorage::truncate(&mut storage, 2);

        assert_eq!(MStorage::len(&storage).unwrap(), 2);
        assert_eq!(storage.0.0.len(), 2);
        assert_eq!(storage.0.1.len(), 2);
        assert_eq!(storage.1.len(), 2);
    }
}
