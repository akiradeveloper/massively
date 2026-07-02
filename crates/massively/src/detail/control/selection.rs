use cubecl::prelude::*;

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct SelectionControl {
    pub(crate) flag: cubecl::server::Handle,
    pub(crate) position: cubecl::server::Handle,
    pub(crate) count: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
}

#[derive(Clone)]
pub(crate) struct SelectionHandles {
    pub(crate) control: SelectionControl,
    pub(crate) value: cubecl::server::Handle,
}

#[derive(Clone)]
pub(crate) struct UniqueByKeyControl {
    pub(crate) selection: SelectionControl,
    pub(crate) count: usize,
}

impl std::ops::Deref for SelectionHandles {
    type Target = SelectionControl;

    fn deref(&self) -> &Self::Target {
        &self.control
    }
}

impl SelectionControl {
    pub(crate) fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            flag: crate::policy::empty_handle(client),
            position: crate::policy::empty_handle(client),
            count: client.empty(std::mem::size_of::<u32>()),
            len: 0,
            len_u32: 0,
        }
    }

    pub(crate) fn flags_only<R: Runtime>(
        client: &ComputeClient<R>,
        flag: cubecl::server::Handle,
        len: usize,
        len_u32: u32,
    ) -> Self {
        Self {
            flag,
            position: crate::policy::empty_handle(client),
            count: crate::policy::empty_handle(client),
            len,
            len_u32,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn for_value(&self, value: cubecl::server::Handle) -> SelectionHandles {
        SelectionHandles {
            control: self.clone(),
            value,
        }
    }
}

impl SelectionHandles {
    pub(crate) fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        SelectionControl::empty(client).for_value(crate::policy::empty_handle(client))
    }
}
