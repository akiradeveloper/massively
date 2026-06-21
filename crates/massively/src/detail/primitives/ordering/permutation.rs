use crate::{
    device::DeviceVec,
    error::Error,
    op::{BinaryPredicateOp, GpuOp},
    policy::CubePolicy,
    primitives::range,
};
use cubecl::prelude::*;

use super::sort_by_key_input_with_policy;

pub(crate) struct Permutation<R: Runtime> {
    indices: DeviceVec<R, u32>,
}

impl<R: Runtime> Permutation<R> {
    pub(crate) fn indices(&self) -> &DeviceVec<R, u32> {
        &self.indices
    }
}

pub(crate) fn sort_by_key_permutation<R, K, Less>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, K>, Permutation<R>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<K>,
{
    let indices = range::indices_u32(policy, keys.len())?;
    let (keys, indices) =
        sort_by_key_input_with_policy(policy, keys, &indices, GpuOp::<Less>::new())?;
    Ok((keys, Permutation { indices }))
}
