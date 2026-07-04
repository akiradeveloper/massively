use cubecl::prelude::*;

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct PermutationControl<R: Runtime> {
    pub(crate) source_indices: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime> PermutationControl<R> {
    pub(crate) fn from_indices<T>(
        indices: &crate::detail::device::DeviceVec<R, T>,
    ) -> Result<Self, crate::Error> {
        let len = indices.len();
        let len_u32 = u32::try_from(len).map_err(|_| crate::Error::LengthTooLarge { len })?;
        Ok(Self {
            source_indices: indices.handle.clone(),
            len,
            len_u32,
            _runtime: std::marker::PhantomData,
        })
    }

    pub(crate) fn indices(
        &self,
        policy: &crate::policy::CubePolicy<R>,
    ) -> crate::detail::device::DeviceVec<R, u32> {
        crate::detail::device::DeviceVec::from_handle(
            policy.id(),
            self.source_indices.clone(),
            self.len,
        )
    }
}

#[derive(Clone)]
pub(crate) struct OrderingControl<R: Runtime> {
    permutation: PermutationControl<R>,
}

impl<R: Runtime> OrderingControl<R> {
    pub(crate) fn from_sorted_indices<T>(
        indices: &crate::detail::device::DeviceVec<R, T>,
    ) -> Result<Self, crate::Error> {
        Ok(Self {
            permutation: PermutationControl::from_indices(indices)?,
        })
    }

    pub(crate) fn permutation(&self) -> &PermutationControl<R> {
        &self.permutation
    }
}

#[allow(dead_code)]
pub(crate) struct MergeControl<R: Runtime> {
    pub(crate) source_side: cubecl::server::Handle,
    pub(crate) source_index: cubecl::server::Handle,
    pub(crate) left_len: usize,
    pub(crate) right_len: usize,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime> MergeControl<R> {
    pub(crate) fn as_merge_by_key_control(&self) -> MergeByKeyControl {
        MergeByKeyControl {
            source_sides: self.source_side.clone(),
            source_indices: self.source_index.clone(),
            left_len: self.left_len,
            right_len: self.right_len,
            len: self.len,
        }
    }
}

#[derive(Clone)]
pub(crate) struct MergeByKeyControl {
    pub(crate) source_sides: cubecl::server::Handle,
    pub(crate) source_indices: cubecl::server::Handle,
    pub(crate) left_len: usize,
    pub(crate) right_len: usize,
    pub(crate) len: usize,
}
