use crate::policy::CubePolicy;
use cubecl::prelude::*;

pub(crate) struct Workspace<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
}

impl<'a, R: Runtime> Workspace<'a, R> {
    pub(crate) fn new(policy: &'a CubePolicy<R>) -> Self {
        Self {
            client: policy.client(),
        }
    }

    pub(crate) fn from_client(client: &'a ComputeClient<R>) -> Self {
        Self { client }
    }

    pub(crate) fn alloc<T>(&self, len: usize) -> cubecl::server::Handle
    where
        T: CubeElement,
    {
        self.client.empty(len * std::mem::size_of::<T>())
    }

    pub(crate) fn alloc_pair<T>(
        &self,
        len: usize,
    ) -> (cubecl::server::Handle, cubecl::server::Handle)
    where
        T: CubeElement,
    {
        (self.alloc::<T>(len), self.alloc::<T>(len))
    }
}
