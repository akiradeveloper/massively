use super::*;

pub(crate) fn array_from_inner<R, Item, Output>(inner: <Item as MItem<R>>::Inner) -> Output
where
    R: Runtime,
    Item: MItem<R>,
    Output: MVec<R, Item = Item>,
{
    <Output as MVec<R>>::from_inner(inner)
}
