use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, PredicateOp, count_if, lazy};

struct Even;

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2u32 == 0u32
    }
}

/// Guards both u32 logical lengths above i32::MAX and reduction block-count
/// dispatch without allocating a 16 GiB input buffer.
#[test]
fn count_if_lazy_four_billion_elements() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = lazy::counting(0).take(4_000_000_000u32);

    assert_eq!(count_if(&exec, input, Even).unwrap(), 2_000_000_000u32);
}
