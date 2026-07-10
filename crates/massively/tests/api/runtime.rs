use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor};

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
