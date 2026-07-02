use crate::common::*;

#[test]
fn unique_keeps_one_value_when_all_values_are_equal() {
    let exec = exec();
    let values = exec.to_device(&[7.0_f32, 7.0, 7.0, 7.0]).unwrap();

    let massively::SoA1(output) =
        unique(&exec, massively::SoA1(values.slice(..)), EqualF32).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![7.0]);
}

#[test]
fn unique_keeps_all_values_when_no_adjacent_values_are_equal() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let massively::SoA1(output) =
        unique(&exec, massively::SoA1(values.slice(..)), EqualF32).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn unique_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 100.0, 200.0, 201.0, 300.0])
        .unwrap();
    let d = exec.to_device(&[1000_u32, 1000, 2000, 2001, 3000]).unwrap();
    let e = exec
        .to_device(&[10000.0_f32, 10000.0, 20000.0, 20001.0, 30000.0])
        .unwrap();
    let f = exec
        .to_device(&[100000_u32, 100000, 200000, 200001, 300000])
        .unwrap();
    let g = exec
        .to_device(&[1000000.0_f32, 1000000.0, 2000000.0, 2000001.0, 3000000.0])
        .unwrap();

    let output = unique(
        &exec,
        massively::SoA7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        Tuple7MixedEqual,
    )
    .unwrap();
    let massively::SoA7(a, b, c, d, e, f, g) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 200.0, 201.0, 300.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![1000, 2000, 2001, 3000]);
    assert_eq!(
        exec.to_host(&e).unwrap(),
        vec![10000.0, 20000.0, 20001.0, 30000.0]
    );
    assert_eq!(
        exec.to_host(&f).unwrap(),
        vec![100000, 200000, 200001, 300000]
    );
    assert_eq!(
        exec.to_host(&g).unwrap(),
        vec![1000000.0, 2000000.0, 2000001.0, 3000000.0]
    );
}
