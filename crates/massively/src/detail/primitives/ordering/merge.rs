#[derive(Clone)]
pub(crate) struct MergeByKeyControl {
    pub(crate) source_sides: cubecl::server::Handle,
    pub(crate) source_indices: cubecl::server::Handle,
    pub(crate) left_len: usize,
    pub(crate) right_len: usize,
    pub(crate) len: usize,
}
