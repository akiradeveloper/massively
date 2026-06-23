use crate::common::*;

#[test]
fn executor_tabulate_initializes_u32_from_index() {
    let exec = exec();

    let output = exec.tabulate(6, SquareIndex).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 4, 9, 16, 25]);
}

#[test]
fn executor_tabulate_initializes_f32_from_index() {
    let exec = exec();

    let output = exec.tabulate(5, HalfIndex).unwrap();

    assert_eq!(
        exec.to_host(&output).unwrap(),
        vec![0.0, 0.5, 1.0, 1.5, 2.0]
    );
}

#[test]
fn executor_tabulate_accepts_empty_length() {
    let exec = exec();

    let output = exec.tabulate(0, SquareIndex).unwrap();

    assert!(output.is_empty());
    assert_eq!(exec.to_host(&output).unwrap(), Vec::<u32>::new());
}
