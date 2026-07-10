use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MIter, MIterMut, UnaryOp, fill, op::Identity, transform, zip2, zip3, zip4, zip5,
    zip6, zip7,
};

struct AddThree;
struct IdentityRightAssociated;

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

#[test]
fn zip_helpers_are_left_associated_public_iterators() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let columns: Vec<_> = (0_u32..7)
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

    let output = exec.to_device(&[0_u32; 2]);
    transform(
        &exec,
        zip3(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
        ),
        AddThree,
        output.slice_mut(..),
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
    let out_a = exec.to_device(&[0_u32; 2]);
    let out_b = exec.to_device(&[0_u32; 2]);
    let out_c = exec.to_device(&[0_u32; 2]);

    transform(
        &exec,
        zip2(a.slice(..), zip2(b.slice(..), c.slice(..))),
        IdentityRightAssociated,
        zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![5, 6]);
}

#[test]
fn read_slice_adapters_compose_on_binary_zip_trees() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let a = exec.to_device(&[1_u32, 2, 3, 4]);
    let b = exec.to_device(&[10_u32, 20, 30, 40]);
    let c = exec.to_device(&[100_u32, 200, 300, 400]);
    let output = exec.to_device(&[0_u32; 1]);

    let sliced = zip3(a.slice(..), b.slice(..), c.slice(..))
        .slice(1..4)
        .slice(1..2);
    assert_eq!(MIter::<WgpuRuntime>::len(&sliced).unwrap(), 1);

    transform(&exec, sliced, AddThree, output.slice_mut(..)).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![333]);
}

#[test]
fn mutable_slice_adapters_compose_and_can_be_read_back() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let out_a = exec.to_device(&[0_u32; 5]);
    let out_b = exec.to_device(&[0_u32; 5]);
    let output = zip2(out_a.slice_mut(..), out_b.slice_mut(..));

    fill(
        &exec,
        (7_u32, 9_u32),
        output.slice_mut(1..4).slice_mut(1..2),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0, 0, 7, 0, 0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 0, 9, 0, 0]);

    let read = output.slice(1..4).slice(1..2);
    let copy_a = exec.to_device(&[0_u32; 1]);
    let copy_b = exec.to_device(&[0_u32; 1]);
    transform(
        &exec,
        read,
        Identity,
        zip2(copy_a.slice_mut(..), copy_b.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&copy_a).unwrap(), vec![7]);
    assert_eq!(exec.to_host(&copy_b).unwrap(), vec![9]);
}
