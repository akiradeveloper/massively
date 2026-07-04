use cubecl::prelude::*;

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct MaskControl {
    pub(crate) flag: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
}

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct SelectedRankControl {
    pub(crate) flag: cubecl::server::Handle,
    pub(crate) position: cubecl::server::Handle,
    pub(crate) count: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
}

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct SplitRankControl {
    pub(crate) flag: cubecl::server::Handle,
    // Inclusive selected-rank prefix. Rejected payload application derives its
    // output rank from this prefix and the input index.
    pub(crate) position: cubecl::server::Handle,
    pub(crate) selected_count: cubecl::server::Handle,
    pub(crate) rejected_count: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
}

#[derive(Clone)]
pub(crate) struct UniqueByKeyControl {
    pub(crate) selection: SelectedRankControl,
    pub(crate) count: usize,
}

impl MaskControl {
    #[allow(dead_code)]
    pub(crate) fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            flag: crate::policy::empty_handle(client),
            len: 0,
            len_u32: 0,
        }
    }

    pub(crate) fn from_flags(flag: cubecl::server::Handle, len: usize, len_u32: u32) -> Self {
        Self { flag, len, len_u32 }
    }
}

impl SelectedRankControl {
    pub(crate) fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            flag: crate::policy::empty_handle(client),
            position: crate::policy::empty_handle(client),
            count: client.empty(std::mem::size_of::<u32>()),
            len: 0,
            len_u32: 0,
        }
    }

    pub(crate) fn from_parts(
        flag: cubecl::server::Handle,
        position: cubecl::server::Handle,
        count: cubecl::server::Handle,
        len: usize,
        len_u32: u32,
    ) -> Self {
        Self {
            flag,
            position,
            count,
            len,
            len_u32,
        }
    }

    pub(crate) fn from_mask_only<R: Runtime>(client: &ComputeClient<R>, mask: MaskControl) -> Self {
        Self {
            flag: mask.flag,
            position: crate::policy::empty_handle(client),
            count: crate::policy::empty_handle(client),
            len: mask.len,
            len_u32: mask.len_u32,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn mask(&self) -> MaskControl {
        MaskControl::from_flags(self.flag.clone(), self.len, self.len_u32)
    }
}

impl SplitRankControl {
    pub(crate) fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            flag: crate::policy::empty_handle(client),
            position: crate::policy::empty_handle(client),
            selected_count: client.empty(std::mem::size_of::<u32>()),
            rejected_count: client.empty(std::mem::size_of::<u32>()),
            len: 0,
            len_u32: 0,
        }
    }

    pub(crate) fn from_selected_rank(
        selected: SelectedRankControl,
        rejected_count: cubecl::server::Handle,
    ) -> Self {
        Self {
            flag: selected.flag,
            position: selected.position,
            selected_count: selected.count,
            rejected_count,
            len: selected.len,
            len_u32: selected.len_u32,
        }
    }
}
