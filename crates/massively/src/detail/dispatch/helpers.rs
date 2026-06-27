use super::*;

pub(crate) fn array_from_inner<R, Item, Output>(inner: <Item as MItem<R>>::Inner) -> Output
where
    R: Runtime,
    Item: MItem<R>,
    Output: MVec<R, Item = Item>,
{
    <Output as MVec<R>>::from_inner(inner)
}

pub(crate) fn column_view_at<R, Iter, T>(
    iter: &Iter,
    index: usize,
    algorithm: &str,
) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error>
where
    R: Runtime,
    Iter: MIter<R>,
    T: Scalar + 'static,
{
    <Iter as MIterDispatch<R>>::column_view_by_index_inner::<T>(iter, index)?.ok_or_else(|| {
        Error::Launch {
            message: format!("{algorithm} is not supported for this iterator shape"),
        }
    })
}
