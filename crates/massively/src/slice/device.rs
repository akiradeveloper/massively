use cubecl::prelude::Runtime;

use crate::Error;
use crate::runtime::{DeviceSlice, Executor, Scalar};
use crate::slice::MSlice;

impl<'a, R, T> MSlice<R> for DeviceSlice<'a, R, T>
where
    R: Runtime,
    T: Scalar + 'static,
{
    type Item = T;
    type Read = crate::detail::device::DeviceColumnView<R, T>;

    fn len(&self) -> usize {
        DeviceSlice::len(self)
    }

    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        exec.ensure_policy_id(self.policy_id())
    }

    fn into_read(self, _policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        Ok(self.column_view())
    }
}
