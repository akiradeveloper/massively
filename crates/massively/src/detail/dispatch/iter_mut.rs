use super::*;

pub trait MIterMutDispatch<B: Runtime>: Sized {
    fn validate_executor(&self, _exec: &Executor<B>) -> Result<(), Error> {
        Ok(())
    }

    fn column_mut_view_inner<T: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<B, T>>, Error>
    where
        T: Scalar,
    {
        Ok(None)
    }

    fn column_mut_view_by_index_inner<T: 'static>(
        &self,
        index: usize,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<B, T>>, Error>
    where
        T: Scalar,
    {
        if index == 0 {
            self.column_mut_view_inner::<T>()
        } else {
            Ok(None)
        }
    }
}
