use cubecl::prelude::*;

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct SegmentControl<R: Runtime> {
    pub(crate) head_flags: cubecl::server::Handle,
    pub(crate) end_flags: Option<cubecl::server::Handle>,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct ScanByKeyControl<R: Runtime> {
    pub(crate) head_flags: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

pub(crate) struct ReduceByKeyControl<R: Runtime> {
    pub(crate) head_flags: cubecl::server::Handle,
    pub(crate) end_flags: cubecl::server::Handle,
    pub(crate) output_selection: super::SelectedRankControl,
    pub(crate) output_count: usize,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime> From<&ReduceByKeyControl<R>> for ScanByKeyControl<R> {
    fn from(control: &ReduceByKeyControl<R>) -> Self {
        Self {
            head_flags: control.head_flags.clone(),
            len: control.len,
            len_u32: control.len_u32,
            _runtime: std::marker::PhantomData,
        }
    }
}

impl<R: Runtime> From<&SegmentControl<R>> for ScanByKeyControl<R> {
    fn from(control: &SegmentControl<R>) -> Self {
        Self {
            head_flags: control.head_flags.clone(),
            len: control.len,
            len_u32: control.len_u32,
            _runtime: std::marker::PhantomData,
        }
    }
}

impl<R: Runtime> From<&ScanByKeyControl<R>> for SegmentControl<R> {
    fn from(control: &ScanByKeyControl<R>) -> Self {
        Self {
            head_flags: control.head_flags.clone(),
            end_flags: None,
            len: control.len,
            len_u32: control.len_u32,
            _runtime: std::marker::PhantomData,
        }
    }
}
