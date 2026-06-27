use cubecl::prelude::Runtime;

use crate::Error;
use crate::runtime::{DeviceSlice, Executor, Scalar};
use crate::slice::MSlice;

impl<'a, B, T> MSlice<B> for DeviceSlice<'a, B, T>
where
    B: Runtime,
    T: Scalar + 'static,
{
    type Item = T;
    type Read = crate::detail::device::DeviceColumnView<B, T>;

    fn len(&self) -> usize {
        DeviceSlice::len(self)
    }

    fn validate_executor(&self, exec: &Executor<B>) -> Result<(), Error> {
        exec.ensure_policy_id(self.policy_id())
    }

    fn into_read(self) -> Self::Read {
        self.column_view()
    }

    fn column_view<U: Scalar + 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnView<B, U>>, Error> {
        Ok(self.column_view_as::<U>())
    }
}
