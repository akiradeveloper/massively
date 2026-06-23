use crate::common::*;

#[test]
fn unique_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 1]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let (keys, values) = unique_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 300.0]);
}

#[test]
fn unique_by_key_accepts_tuple_values_with_multiple_runs() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 0, 2, 3, 3]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();

    let (keys, values) = unique_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 2, 3]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 4.0, 5.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 40, 50]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 400.0, 500.0]);
}
