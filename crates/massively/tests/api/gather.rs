use crate::common::*;

#[test]
fn gather_accepts_lazy_counting_indices() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let output = exec.to_device(&[0_u32; 3]).unwrap();

    gather(
        &exec,
        massively::Zip1(values.slice(..)),
        massively::lazy::counting(1).take(3),
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 40]);
}
