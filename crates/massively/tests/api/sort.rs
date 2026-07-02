use crate::common::*;

#[test]
fn sort_returns_device_storage() {
    let exec = exec();
    let x = exec.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();

    let sorted = sort(&exec, massively::SoA1(x.slice(..)), Less).unwrap();
    let massively::SoA1(sorted) = sorted;

    assert_eq!(exec.to_host(&sorted).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&x).unwrap(), vec![3.0, 1.0, 2.0]);
}

#[test]
fn tuple_sort_preserves_soa_components() {
    let exec = exec();
    let x = exec.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let y = exec.to_device(&[30_u32, 10, 20]).unwrap();

    let sorted = sort(
        &exec,
        massively::SoA2(x.slice(..), y.slice(..)),
        MixedTupleLess,
    )
    .unwrap();
    let massively::SoA2(x, y) = sorted;

    assert_eq!(exec.to_host(&x).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&y).unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_accepts_heterogeneous_tuple_comparators_for_two_and_three_columns() {
    let exec = exec();
    let values = exec.to_device(&[2.0_f32, 1.0, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[20_u32, 30, 10, 40]).unwrap();

    let sorted = sort(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        MixedTupleLess,
    )
    .unwrap();
    let massively::SoA2(values, tags) = sorted;
    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![30, 10, 20, 40]);

    let values = exec.to_device(&[2.0_f32, 1.0, 4.0, 3.0]).unwrap();
    let tags = exec.to_device(&[20_u32, 10, 20, 10]).unwrap();
    let payload = exec.to_device(&[200.0_f32, 100.0, 400.0, 300.0]).unwrap();

    let sorted = sort(
        &exec,
        massively::SoA3(values.slice(..), tags.slice(..), payload.slice(..)),
        MixedTuple3Less,
    )
    .unwrap();
    let massively::SoA3(values, tags, payload) = sorted;
    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(
        exec.to_host(&payload).unwrap(),
        vec![100.0, 300.0, 200.0, 400.0]
    );
}

#[test]
fn sort_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[2.0_f32, 1.0, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[20_u32, 30, 10, 40]).unwrap();
    let c = exec.to_device(&[200.0_f32, 100.0, 201.0, 300.0]).unwrap();
    let d = exec.to_device(&[2000_u32, 1000, 2010, 3000]).unwrap();
    let e = exec.to_device(&[2.2_f32, 1.1, 2.1, 3.3]).unwrap();
    let f = exec.to_device(&[22_u32, 11, 21, 33]).unwrap();
    let g = exec.to_device(&[220.0_f32, 110.0, 210.0, 330.0]).unwrap();

    let sorted = sort(
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
        Tuple7MixedLess,
    )
    .unwrap();
    let massively::SoA7(a, b, c, d, e, f, g) = sorted;

    assert_eq!(exec.to_host(&a).unwrap(), vec![2.0, 2.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30, 40]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![201.0, 200.0, 100.0, 300.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![2010, 2000, 1000, 3000]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![2.1, 2.2, 1.1, 3.3]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![21, 22, 11, 33]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![210.0, 220.0, 110.0, 330.0]);
}

#[test]
fn stable_sort_accepts_seven_tuple_columns_and_preserves_equal_order() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 1.0, 1.0, 0.0]).unwrap();
    let b = exec.to_device(&[2_u32, 1, 1, 1]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 201.0, 300.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000, 2010, 3000]).unwrap();
    let e = exec.to_device(&[10.0_f32, 20.0, 21.0, 30.0]).unwrap();
    let f = exec.to_device(&[100_u32, 200, 201, 300]).unwrap();
    let g = exec
        .to_device(&[1000.0_f32, 2000.0, 2010.0, 3000.0])
        .unwrap();

    let sorted = massively::stable_sort(
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
        Tuple7MixedLess,
    )
    .unwrap();
    let massively::SoA7(a, b, c, d, e, f, g) = sorted;

    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![1, 1, 1, 2]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![300.0, 200.0, 201.0, 100.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![3000, 2000, 2010, 1000]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![30.0, 20.0, 21.0, 10.0]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![300, 200, 201, 100]);
    assert_eq!(
        exec.to_host(&g).unwrap(),
        vec![3000.0, 2000.0, 2010.0, 1000.0]
    );
}

#[test]
fn tuple_sort_accepts_wide_borrowed_soas() {
    let exec = exec();
    let a = exec.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let b = exec.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = exec.to_device(&[300.0_f32, 100.0, 200.0]).unwrap();
    let d = exec.to_device(&[3000.0_f32, 1000.0, 2000.0]).unwrap();

    let sorted = sort(
        &exec,
        massively::SoA4(a.slice(..), b.slice(..), c.slice(..), d.slice(..)),
        Tuple4Less,
    )
    .unwrap();
    let massively::SoA4(a, b, c, d) = sorted;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![1000.0, 2000.0, 3000.0]);
}
