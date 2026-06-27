use super::*;

pub(crate) fn array_from_inner<B, Item, Output>(inner: <Item as MItem<B>>::Inner) -> Output
where
    B: Runtime,
    Item: MItem<B>,
    Output: MVec<B, Item = Item>,
{
    <Output as MVec<B>>::from_inner(inner)
}

pub(crate) fn column_view_at<B, Iter, T>(
    iter: &Iter,
    index: usize,
    algorithm: &str,
) -> Result<crate::detail::device::DeviceColumnView<B, T>, Error>
where
    B: Runtime,
    Iter: MIter<B>,
    T: Scalar + 'static,
{
    <Iter as MIterDispatch<B>>::column_view_by_index_inner::<T>(iter, index)?.ok_or_else(|| {
        Error::Launch {
            message: format!("{algorithm} is not supported for this iterator shape"),
        }
    })
}
