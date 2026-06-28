use cubecl::prelude::*;

#[allow(dead_code)]
pub(crate) struct PermutationControl<R: Runtime> {
    pub(crate) source_indices: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

#[allow(dead_code)]
pub(crate) struct MergeControl<R: Runtime> {
    pub(crate) source_side: cubecl::server::Handle,
    pub(crate) source_index: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}
