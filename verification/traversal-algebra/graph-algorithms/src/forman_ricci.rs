//! Unweighted Forman–Ricci curvature emitted in CSR edge order.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, graph, op::UnaryOp, zip2};

use super::common::{self, DeviceCsr};

struct Curvature;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Curvature {
    type Output = i32;

    fn apply(input: (u32, u32)) -> i32 {
        4i32 - i32::cast_from(input.0) - i32::cast_from(input.1)
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, i32>> {
    let degree = common::resident_degrees(exec, graph)?;
    graph::traverse(
        exec,
        graph.csr(),
        common::counting_u32(0, graph.vertex_count() as usize),
    )?
    .map(
        zip2(
            graph::source(degree.slice(..)),
            graph::destination(degree.slice(..)),
        ),
        Curvature,
    )
    .emit(exec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn curvature_matches_endpoint_degrees() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        let output = solve(&exec, &graph).unwrap();
        assert_eq!(
            exec.to_host(&output).unwrap(),
            vec![-1, -1, -1, -2, -1, -1, -2, -1, -1, -1]
        );
    }
}
