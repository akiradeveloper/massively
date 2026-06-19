use crate::{
    device::DeviceVec,
    error::Error,
    op::{BinaryPredicateOp, GpuOp},
    primitives::range,
};
use cubecl::prelude::*;

use super::sort_by_key;

pub(crate) struct Permutation<R: Runtime> {
    indices: DeviceVec<R, u32>,
}

impl<R: Runtime> Permutation<R> {
    pub(crate) fn indices(&self) -> &DeviceVec<R, u32> {
        &self.indices
    }
}

pub(crate) fn sort_by_key_permutation<R, K, Less>(
    keys: &DeviceVec<R, K>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, K>, Permutation<R>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<K>,
{
    let indices = range::indices_u32(keys.policy(), keys.len())?;
    let (keys, indices) = sort_by_key(keys, &indices, GpuOp::<Less>::new())?;
    Ok((keys, Permutation { indices }))
}
