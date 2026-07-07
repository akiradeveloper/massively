use crate::common::*;

#[test]
fn partition_puts_everything_on_matching_side_when_all_values_match() {
    let exec = exec();
    let values = exec.to_device(&[2.0_f32, 3.0, 4.0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 3]).unwrap();

    let split = partition(
        &exec,
        massively::Zip1(values.slice(..)),
        F32GreaterThanOne,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&output.slice(..split)).unwrap(),
        vec![2.0, 3.0, 4.0]
    );
    assert_eq!(
        exec.to_host(&output.slice(split..)).unwrap(),
        Vec::<f32>::new()
    );
}

#[test]
fn partition_puts_everything_on_failing_side_when_no_values_match() {
    let exec = exec();
    let values = exec.to_device(&[-1.0_f32, 0.0, 1.0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 3]).unwrap();

    let split = partition(
        &exec,
        massively::Zip1(values.slice(..)),
        F32GreaterThanOne,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&output.slice(..split)).unwrap(),
        Vec::<f32>::new()
    );
    assert_eq!(
        exec.to_host(&output.slice(split..)).unwrap(),
        vec![-1.0, 0.0, 1.0]
    );
}
