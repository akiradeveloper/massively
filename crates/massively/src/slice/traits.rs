use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::CubePolicy;
use crate::detail::device::{KernelColumn, KernelColumnAt, S0};
use crate::runtime::{Executor, Scalar};

/// Read-only logical slice that maps an index `i` to an item `T`.
///
/// `MSlice` is the public abstraction that lets algorithms accept storage
/// backed slices today and lazy slices in later versions without changing the
/// algorithm surface for every slice kind.
pub trait MSlice<R: Runtime>: Sized {
    type Item: Scalar + 'static;

    #[doc(hidden)]
    type Read: KernelColumn<Runtime = R, Item = Self::Item> + KernelColumnAt<S0>;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[doc(hidden)]
    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error>;

    #[doc(hidden)]
    fn into_read(self, policy: &CubePolicy<R>) -> Result<Self::Read, Error>;

    #[doc(hidden)]
    fn column_view<T: Scalar + 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnView<R, T>>, Error> {
        Ok(None)
    }
}
