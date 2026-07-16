use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MIter, MIterMut, lazy, op::Identity, op::UnaryOp, unzip2, unzip3, unzip4, unzip5,
    unzip6, unzip7, unzip8, unzip9, unzip10, unzip11, unzip12, vector::gather, vector::transform,
    zip2, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12,
};

struct AddThree;
struct IdentityRightAssociated;
struct SumFour;
struct AddPair;

#[test]
fn unzip_helpers_are_the_inverse_of_zip_helpers() {
    assert_eq!(unzip2(zip2(1, 2)), (1, 2));
    assert_eq!(unzip3(zip3(1, 2, 3)), (1, 2, 3));
    assert_eq!(unzip4(zip4(1, 2, 3, 4)), (1, 2, 3, 4));
    assert_eq!(unzip5(zip5(1, 2, 3, 4, 5)), (1, 2, 3, 4, 5));
    assert_eq!(unzip6(zip6(1, 2, 3, 4, 5, 6)), (1, 2, 3, 4, 5, 6));
    assert_eq!(unzip7(zip7(1, 2, 3, 4, 5, 6, 7)), (1, 2, 3, 4, 5, 6, 7));
    assert_eq!(
        unzip8(zip8(1, 2, 3, 4, 5, 6, 7, 8)),
        (1, 2, 3, 4, 5, 6, 7, 8)
    );
    assert_eq!(
        unzip9(zip9(1, 2, 3, 4, 5, 6, 7, 8, 9)),
        (1, 2, 3, 4, 5, 6, 7, 8, 9)
    );
    assert_eq!(
        unzip10(zip10(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    );
    assert_eq!(
        unzip11(zip11(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    );
    assert_eq!(
        unzip12(zip12(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    );
}

#[cubecl::cube]
impl UnaryOp<((u32, u32), u32)> for AddThree {
    type Output = u32;

    fn apply(input: ((u32, u32), u32)) -> u32 {
        input.0.0 + input.0.1 + input.1
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, (u32, u32))> for IdentityRightAssociated {
    type Output = (u32, (u32, u32));

    fn apply(input: (u32, (u32, u32))) -> (u32, (u32, u32)) {
        input
    }
}

#[cubecl::cube]
impl UnaryOp<(((u32, u32), u32), u32)> for SumFour {
    type Output = u32;

    fn apply(input: (((u32, u32), u32), u32)) -> u32 {
        input.0.0.0 + input.0.0.1 + input.0.1 + input.1
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddPair {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

#[test]
fn zip_helpers_are_left_associated_public_iterators() {
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
fn write_reassociates_equal_ordered_leaves_at_the_output_boundary() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let a = exec.to_device(&[1_u32, 2]);
    let b = exec.to_device(&[3_u32, 4]);
    let c = exec.to_device(&[5_u32, 6]);
    let output = transform(
        &exec,
        zip2(a.slice(..), zip2(b.slice(..), c.slice(..))),
        IdentityRightAssociated,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output.0.0).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&output.0.1).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&output.1).unwrap(), vec![5, 6]);
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
    assert_eq!(exec.to_host(&copy.0).unwrap(), vec![7]);
    assert_eq!(exec.to_host(&copy.1).unwrap(), vec![9]);
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
