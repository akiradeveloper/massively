use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MIter, MIterMut, MStorage, lazy, op::Identity, op::UnaryOp, vector::gather,
    vector::transform, zip2, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12,
};

struct AddThree;
struct IdentityTriple;
struct SumFour;
struct AddPair;
struct EncodeBoolIndex;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for AddThree {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 + input.1 + input.2
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for IdentityTriple {
    type Output = (u32, u32, u32);

    fn apply(input: (u32, u32, u32)) -> (u32, u32, u32) {
        input
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32, u32)> for SumFour {
    type Output = u32;

    fn apply(input: (u32, u32, u32, u32)) -> u32 {
        input.0 + input.1 + input.2 + input.3
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddPair {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

#[cubecl::cube]
impl UnaryOp<(bool, usize)> for EncodeBoolIndex {
    type Output = u32;

    fn apply(input: (bool, usize)) -> u32 {
        if input.0 { input.1 as u32 } else { 0u32 }
    }
}

#[test]
fn zip_flattens_read_only_semantic_scalars() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = zip2(lazy::constant(true).take(3), lazy::counting(4).take(3));
    let output = transform(&exec, input, EncodeBoolIndex).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![4, 5, 6]);
}

#[test]
fn zip_helpers_expose_flat_public_iterators() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let columns: Vec<_> = (0_u32..12)
        .map(|base| exec.to_device(&[base + 1, base + 2]))
        .collect();

    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip2(columns[0].slice(..), columns[1].slice(..))).unwrap(),
        2
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip3(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip4(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip5(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip6(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
            columns[5].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip7(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip8(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
            columns[7].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip9(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
            columns[7].slice(..),
            columns[8].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip10(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
            columns[7].slice(..),
            columns[8].slice(..),
            columns[9].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip11(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
            columns[7].slice(..),
            columns[8].slice(..),
            columns[9].slice(..),
            columns[10].slice(..),
        ))
        .unwrap(),
        2,
    );
    assert_eq!(
        MIter::<WgpuRuntime>::len(&zip12(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
            columns[7].slice(..),
            columns[8].slice(..),
            columns[9].slice(..),
            columns[10].slice(..),
            columns[11].slice(..),
        ))
        .unwrap(),
        2,
    );

    let output = transform(
        &exec,
        zip3(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
        ),
        AddThree,
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![6, 9]);
}

#[test]
fn zip_grouping_does_not_change_the_logical_row() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let a = exec.to_device(&[1_u32, 2]);
    let b = exec.to_device(&[3_u32, 4]);
    let c = exec.to_device(&[5_u32, 6]);

    let left_grouped = transform(
        &exec,
        zip2(zip2(a.slice(..), b.slice(..)), c.slice(..)),
        IdentityTriple,
    )
    .unwrap();
    let right_grouped = transform(
        &exec,
        zip2(a.slice(..), zip2(b.slice(..), c.slice(..))),
        IdentityTriple,
    )
    .unwrap();

    let (left_a, left_b, left_c) = MStorage::into_columns(left_grouped);
    let (right_a, right_b, right_c) = MStorage::into_columns(right_grouped);
    assert_eq!(exec.to_host(&left_a).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&left_b).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&left_c).unwrap(), vec![5, 6]);
    assert_eq!(exec.to_host(&right_a).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&right_b).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&right_c).unwrap(), vec![5, 6]);
}

#[test]
fn read_slice_adapters_compose_on_binary_zip_trees() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let a = exec.to_device(&[1_u32, 2, 3, 4]);
    let b = exec.to_device(&[10_u32, 20, 30, 40]);
    let c = exec.to_device(&[100_u32, 200, 300, 400]);

    let sliced = zip3(a.slice(..), b.slice(..), c.slice(..))
        .slice(1..4)
        .slice(1..2);
    assert_eq!(MIter::<WgpuRuntime>::len(&sliced).unwrap(), 1);

    let output = transform(&exec, sliced, AddThree).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![333]);
}

#[test]
fn mutable_slice_adapters_compose_and_can_be_read_back() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let out_a = exec.to_device(&[0_u32; 5]);
    let out_b = exec.to_device(&[0_u32; 5]);
    let output = zip2(out_a.slice_mut(..), out_b.slice_mut(..));

    massively::vector::replace_where(
        &exec,
        (7_u32, 9_u32),
        lazy::constant(true).take(1),
        output.slice_mut(1..4).slice_mut(1..2),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0, 0, 7, 0, 0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 0, 9, 0, 0]);

    let read = output.slice(1..4).slice(1..2);
    let copy = transform(&exec, read, Identity).unwrap();
    let (first, second) = MStorage::into_columns(copy);
    assert_eq!(exec.to_host(&first).unwrap(), vec![7]);
    assert_eq!(exec.to_host(&second).unwrap(), vec![9]);
}

#[test]
fn gather_keeps_an_eval8_value_expression_lazy() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let columns: Vec<_> = (0_u32..8)
        .map(|column| exec.to_device(&[column, column + 10, column + 20]))
        .collect();
    let indices = exec.to_device(&[2_u32, 0]);
    let left = lazy::transform(
        zip4(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
        ),
        SumFour,
    );
    let right = lazy::transform(
        zip4(
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
            columns[7].slice(..),
        ),
        SumFour,
    );
    let values = lazy::transform(zip2(left, right), AddPair);
    let indices = lazy::transform(indices.slice(..), massively::op::U32ToUsize);

    let output = gather(&exec, values, indices).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![188, 28]);
}
