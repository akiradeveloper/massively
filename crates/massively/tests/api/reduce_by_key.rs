use crate::common::*;

#[test]
fn reduce_by_key_uses_supplied_key_equality() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 2, 4, 1, 3]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        SameParityU32,
        (0.0,),
        Sum,
    )
    .unwrap();
    let (keys,) = keys;
    let (values,) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![4, 3]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![6.0, 9.0]);
}

#[test]
fn reduce_by_key_handles_singleton_runs() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 1, 2, 3]).unwrap();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let ((out_keys,), (out_values,)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn reduce_by_key_handles_one_run() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();

    let ((out_keys,), (out_values,)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10]);
}

#[test]
fn reduce_by_key_handles_all_same_key_long_run() {
    let exec = exec();
    let len = 512;
    let keys = vec![7_u32; len];
    let values = vec![1_u32; len];

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let ((out_keys,), (out_values,)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![7]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![len as u32]);
}

#[test]
fn reduce_by_key_handles_block_boundary_runs() {
    let exec = exec();
    let mut keys = vec![0_u32; 300];
    keys.extend(vec![1_u32; 20]);
    keys.extend(vec![0_u32; 10]);
    let values = vec![1_u32; keys.len()];

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let ((out_keys,), (out_values,)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0, 1, 0]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![300, 20, 10]);
}

#[test]
fn reduce_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA2(values.slice(..), ids.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();
    let (keys,) = keys;
    let (values, ids) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![3.0, 12.0]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![30, 120]);
}

#[test]
fn reduce_by_key_accepts_three_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0])
        .unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![3.0, 12.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 120]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![300.0, 1200.0]);
}
