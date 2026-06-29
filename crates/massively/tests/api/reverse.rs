use crate::common::*;

#[test]
fn reverse_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let reversed = reverse(&exec, massively::SoA2(a.slice(..), b.slice(..))).unwrap();
    let massively::SoA2(a, b) = reversed;

    assert_eq!(exec.to_host(&a).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 20, 10]);
}

#[test]
fn reverse_accepts_borrowed_three_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let reversed = reverse(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
    )
    .unwrap();
    let massively::SoA3(a, b, c) = reversed;

    assert_eq!(exec.to_host(&a).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 20, 10]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![300.0, 200.0, 100.0]);
}
