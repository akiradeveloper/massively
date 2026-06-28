use cubecl::prelude::*;

use crate::detail::device::KernelColumnBindings;

#[allow(dead_code)]
pub(crate) struct SegmentControl<R: Runtime> {
    pub(crate) head_flags: cubecl::server::Handle,
    pub(crate) end_flags: Option<cubecl::server::Handle>,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

#[allow(dead_code)]
pub(crate) struct ScanByKeyControl<R: Runtime, K, KeyExpr, KeyPred> {
    pub(crate) key_bindings: KernelColumnBindings,
    pub(crate) head_flags: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _marker: std::marker::PhantomData<fn() -> (R, K, KeyExpr, KeyPred)>,
}

pub(crate) struct ReduceByKeyControl<R: Runtime, K, KeyExpr, KeyPred> {
    pub(crate) key_bindings: KernelColumnBindings,
    pub(crate) head_flags: cubecl::server::Handle,
    pub(crate) end_flags: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _marker: std::marker::PhantomData<fn() -> (R, K, KeyExpr, KeyPred)>,
}

impl<R: Runtime, K, KeyExpr, KeyPred> From<&ReduceByKeyControl<R, K, KeyExpr, KeyPred>>
    for ScanByKeyControl<R, K, KeyExpr, KeyPred>
{
    fn from(control: &ReduceByKeyControl<R, K, KeyExpr, KeyPred>) -> Self {
        Self {
            key_bindings: control.key_bindings.clone(),
            head_flags: control.head_flags.clone(),
            len: control.len,
            len_u32: control.len_u32,
            _marker: std::marker::PhantomData,
        }
    }
}
