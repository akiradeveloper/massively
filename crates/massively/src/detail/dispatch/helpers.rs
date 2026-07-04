use super::*;

pub(crate) fn array_from_inner<R, Item, Output>(inner: <Item as MAlloc<R>>::Inner) -> Output
where
    R: Runtime,
    Item: MAlloc<R>,
    Output: StorageFromInner<R, Item = Item>,
{
    <Output as StorageFromInner<R>>::from_inner(inner)
}
