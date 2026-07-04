#[derive(Clone)]
pub(crate) struct SearchControl {
    pub(crate) flag: cubecl::server::Handle,
    pub(crate) storage_len: usize,
    pub(crate) logical_len: usize,
}

impl SearchControl {
    pub(crate) fn from_flags(
        flag: cubecl::server::Handle,
        storage_len: usize,
        logical_len: usize,
    ) -> Self {
        Self {
            flag,
            storage_len,
            logical_len,
        }
    }
}
