use super::*;

pub trait MIterMutDispatch<R: Runtime>: Sized {
    fn validate_executor(&self, _exec: &Executor<R>) -> Result<(), Error> {
        Ok(())
    }

    fn column_mut_view_inner<T: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, T>>, Error>
    where
        T: MStorageElement,
    {
        Ok(None)
    }

    fn column_mut_view_by_index_inner<T: 'static>(
        &self,
        index: usize,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, T>>, Error>
    where
        T: MStorageElement,
    {
        if index == 0 {
            self.column_mut_view_inner::<T>()
        } else {
            Ok(None)
        }
    }
}
