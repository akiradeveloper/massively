use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, lazy, op::PredicateOp, vector::count_if};

struct Even;

#[cubecl::cube]
impl PredicateOp<usize> for Even {
    fn apply(value: usize) -> bool {
        value % 2usize == 0usize
    }
}

/// Guards both u32 logical lengths above i32::MAX and reduction block-count
/// dispatch without allocating a 16 GiB input buffer.
#[test]
fn count_if_lazy_four_billion_elements() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = lazy::counting(0).take(4_000_000_000usize);

    assert_eq!(count_if(&exec, input, Even).unwrap(), 2_000_000_000usize);
}
