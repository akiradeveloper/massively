use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MStorage, op::UnaryOp, vector::transform};

type Twelve = (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);

struct AscendingTuple;

#[cubecl::cube]
impl UnaryOp<u32> for AscendingTuple {
    type Output = Twelve;

    fn apply(input: u32) -> Self::Output {
        (
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
            input + 6,
            input + 7,
            input + 8,
            input + 9,
            input + 10,
            input + 11,
        )
    }
}

struct ReversedTuple;

#[cubecl::cube]
impl UnaryOp<u32> for ReversedTuple {
    type Output = Twelve;

    fn apply(input: u32) -> Self::Output {
        let (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = (
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
            input + 6,
            input + 7,
            input + 8,
            input + 9,
            input + 10,
            input + 11,
        );
        (a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0)
    }
}

#[test]
fn flat_tuples_can_be_destructured_directly_inside_cube_ops() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20]);
    let outputs = transform(&exec, input.slice(..), ReversedTuple).unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = MStorage::into_columns(outputs);
    let outputs = [&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l];

    for (column, offset) in outputs
        .into_iter()
        .zip([11_u32, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![10 + offset, 20 + offset]
        );
    }
}

#[test]
fn tuple_outputs_expose_flat_owned_columns() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20]);
    let outputs = transform(&exec, input.slice(..), AscendingTuple).unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = MStorage::into_columns(outputs);
    let outputs = [&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l];

    for (column, offset) in outputs.into_iter().zip(0_u32..) {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![10 + offset, 20 + offset]
        );
    }
}
