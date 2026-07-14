//! Iterative geolocation with device-resident coordinates and known flags.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MVec, graph, lazy, op::Identity, op::ReductionOp, op::UnaryOp, vector,
    zip2,
};

use super::common::{self, DeviceCsr};

struct SumCoordinates;

#[cubecl::cube]
impl ReductionOp<(f32, f32)> for SumCoordinates {
    fn apply(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

struct UpdateCoordinates;

#[cubecl::cube]
impl UnaryOp<(((f32, f32), (f32, f32)), (u32, u32))> for UpdateCoordinates {
    type Output = (f32, f32);

    fn apply(input: (((f32, f32), (f32, f32)), (u32, u32))) -> (f32, f32) {
        let coordinates = input.0.0;
        let sums = input.0.1;
        let degree = input.1.0;
        let known = input.1.1;
        if known != 0u32 || degree == 0u32 {
            coordinates
        } else {
            (sums.0 / degree as f32, sums.1 / degree as f32)
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    initial: &MVec<R, (f32, f32)>,
    known: &DeviceVec<R, u32>,
    iterations: usize,
) -> common::Result<MVec<R, (f32, f32)>> {
    let n = graph.vertex_count();
    assert_eq!(initial.0.len(), n as usize);
    assert_eq!(initial.1.len(), n as usize);
    assert_eq!(known.len(), n as usize);
    let degree = common::resident_degrees(exec, graph)?;
    let mut coordinates = initial.clone();

    for _ in 0..iterations {
        let sums = graph::traverse(exec, graph.csr(), lazy::counting(0).take(n))?
            .map(
                graph::destination(zip2(coordinates.0.slice(..), coordinates.1.slice(..))),
                Identity,
            )
            .reduce_by_source(exec, (0.0, 0.0), SumCoordinates)?;

        coordinates = vector::transform(
            exec,
            zip2(
                zip2(
                    zip2(coordinates.0.slice(..), coordinates.1.slice(..)),
                    zip2(sums.0.slice(..), sums.1.slice(..)),
                ),
                zip2(degree.slice(..), known.slice(..)),
            ),
            UpdateCoordinates,
        )?;
    }

    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn unknown_center_is_neighbor_mean() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = common::CsrGraph::new(vec![0, 2, 3, 4], vec![1, 2, 0, 0]);
        let graph = DeviceCsr::from_host(&exec, &graph).unwrap();
        let initial = zip2(
            exec.to_device(&[0.0f32, 0.0, 2.0]),
            exec.to_device(&[0.0f32, 0.0, 2.0]),
        );
        let known = exec.to_device(&[0u32, 1, 1]);
        let result = solve(&exec, &graph, &initial, &known, 1).unwrap();
        assert_eq!(exec.to_host(&result.0).unwrap(), vec![1.0, 0.0, 2.0]);
        assert_eq!(exec.to_host(&result.1).unwrap(), vec![1.0, 0.0, 2.0]);
    }
}
