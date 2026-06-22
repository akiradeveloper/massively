use crate::common::*;

#[test]
fn gather_accepts_soa12_values() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[2_u32, 0]).unwrap();
    let output = gather(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        indices.slice(..),
    )
    .unwrap();
    let (a, b) = output;
    assert_eq!(exec.to_host(&a).unwrap(), vec![3.0, 1.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 10]);
}
