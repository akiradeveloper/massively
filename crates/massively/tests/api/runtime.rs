use cubecl::prelude::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::*;
use static_assertions::{assert_impl_all, assert_not_impl_any};

type LeftAssociatedOutput = Zip<Zip<DeviceSliceMut<u32>, DeviceSliceMut<u32>>, DeviceSliceMut<u32>>;
type RightAssociatedOutput =
    Zip<DeviceSliceMut<u32>, Zip<DeviceSliceMut<u32>, DeviceSliceMut<u32>>>;

assert_impl_all!(LeftAssociatedOutput: MIterMut<WgpuRuntime>);
assert_not_impl_any!(RightAssociatedOutput: MIterMut<WgpuRuntime>);

fn fill_through_allocated_output_item<R, Output>(
    exec: &Executor<R>,
    value: Output::Item,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    let len = output.len()? as usize;
    let temporary = exec.alloc::<Output::Item>(len);
    fill(exec, value, temporary.slice_mut(..))?;
    transform(exec, temporary.slice(..), op::Identity, output)
}

#[test]
fn public_device_slice_methods_return_public_view_types() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values: DeviceVec<WgpuRuntime, u32> = exec.to_device(&[1_u32, 2, 3, 4, 5]);

    let read: DeviceSlice<u32> = values.slice(1..5);
    let nested_read: DeviceSlice<u32> = read.slice(1..3);
    assert_eq!(exec.to_host(&nested_read).unwrap(), vec![3, 4]);

    let write: DeviceSliceMut<u32> = values.slice_mut(1..5);
    let read_from_write: DeviceSlice<u32> = write.slice(1..3);
    let nested_write: DeviceSliceMut<u32> = write.slice_mut(1..3);
    assert_eq!(exec.to_host(&read_from_write).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&nested_write).unwrap(), vec![3, 4]);
}

#[test]
fn generic_output_item_can_be_allocated_as_writable_storage() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[0_u32; 3]);
    let second = exec.to_device(&[0_u32; 3]);
    let third = exec.to_device(&[0_u32; 3]);

    fill_through_allocated_output_item(
        &exec,
        tuple3(7_u32, 11_u32, 13_u32),
        zip3(
            first.slice_mut(..),
            second.slice_mut(..),
            third.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&first).unwrap(), vec![7, 7, 7]);
    assert_eq!(exec.to_host(&second).unwrap(), vec![11, 11, 11]);
    assert_eq!(exec.to_host(&third).unwrap(), vec![13, 13, 13]);
}
