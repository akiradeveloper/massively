use oracle_ref::vector as oracle;

use super::common::*;

#[test]
fn reduce() {
    let input = scale_input();
    let exec = exec();
    let device = exec.to_device(&input);
    assert_eq!(
        massively::vector::reduce(&exec, lazify(device.slice(..)), 0, MaxU32).unwrap(),
        oracle::reduce(&input, 0, MaxU32),
    );
}

#[test]
fn count_if() {
    let input = scale_input();
    let exec = exec();
    let device = exec.to_device(&input);
    assert_eq!(
        massively::vector::count_if(&exec, lazify(device.slice(..)), NonZero).unwrap() as usize,
        oracle::count_if(&input, NonZero),
    );
}

#[test]
fn inclusive_scan() {
    let input = scale_input();
    let exec = exec();
    let device = exec.to_device(&input);
    let output = exec.to_device(&vec![0_u32; input.len()]);
    massively::vector::inclusive_scan(
        &exec,
        lazify(device.slice(..)),
        MaxU32,
        output.slice_mut(..),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        oracle::inclusive_scan(&input, MaxU32),
    );
}
