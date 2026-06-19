use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::BinaryPredicateOp,
    policy::CubePolicy,
    primitives::select::{self, SelectionControl},
};
use cubecl::prelude::*;

const BLOCK_SEGMENTED_SIZE: u32 = 256;

fn segmented_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SEGMENTED_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

#[derive(Clone)]
pub(crate) struct SegmentControl {
    selection: SelectionControl,
    count: usize,
}

impl SegmentControl {
    pub(crate) fn empty<R>(policy: &CubePolicy<R>) -> Result<Self, Error>
    where
        R: Runtime,
    {
        let handles =
            select::handles_from_flags(policy, 0, 0, policy.empty_handle(), policy.empty_handle())?;
        Ok(Self {
            selection: handles,
            count: 0,
        })
    }

    pub(crate) fn from_end_flags<R>(
        policy: &CubePolicy<R>,
        len: usize,
        len_u32: u32,
        flag_handle: cubecl::server::Handle,
        first_value_handle: cubecl::server::Handle,
    ) -> Result<Self, Error>
    where
        R: Runtime,
    {
        let handles =
            select::handles_from_flags(policy, len, len_u32, flag_handle, first_value_handle)?;
        let count = select::selected_count(policy, &handles)?;
        Ok(Self {
            selection: handles,
            count,
        })
    }

    pub(crate) fn compact_first<R, T>(
        &self,
        policy: &CubePolicy<R>,
    ) -> Result<DeviceVec<R, T>, Error>
    where
        R: Runtime,
        T: CubePrimitive + CubeElement,
    {
        select::compact_with_count(policy, self.selection.clone(), self.count)
    }

    pub(crate) fn compact_value<R, T>(
        &self,
        policy: &CubePolicy<R>,
        value_handle: cubecl::server::Handle,
    ) -> Result<DeviceVec<R, T>, Error>
    where
        R: Runtime,
        T: CubePrimitive + CubeElement,
    {
        let handles = self.selection.for_value(value_handle);
        select::compact_with_count(policy, handles, self.count)
    }

    pub(crate) fn compact_pair<R, A, B>(
        &self,
        policy: &CubePolicy<R>,
        first_value_handle: cubecl::server::Handle,
        second_value_handle: cubecl::server::Handle,
    ) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>), Error>
    where
        R: Runtime,
        A: CubePrimitive + CubeElement,
        B: CubePrimitive + CubeElement,
    {
        select::compact_pair_with_count(
            policy,
            &self.selection,
            first_value_handle,
            second_value_handle,
            self.count,
        )
    }
}

pub(crate) fn key_run_control<R, K, Eq>(keys: &DeviceVec<R, K>) -> Result<SegmentControl, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Eq: BinaryPredicateOp<K>,
{
    if keys.len() == 0 {
        return SegmentControl::empty(keys.policy());
    }

    let len = keys.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = keys.policy().client();
    let block_count_u32 = segmented_block_count(len)?;
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());

    unsafe {
        unique_by_key_flags_kernel::launch_unchecked::<K, Eq, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEGMENTED_SIZE),
            unsafe { BufferArg::from_raw_parts(keys.handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    SegmentControl::from_end_flags(
        keys.policy(),
        len,
        len_u32,
        flag_handle,
        keys.handle.clone(),
    )
}
