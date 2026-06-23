use crate::common::*;

#[test]
fn sort_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let a = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let b = exec.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = exec.to_device(&[300_u32, 100, 200]).unwrap();

    let (keys, values) = sort_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        LessU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 200, 300]);
}
