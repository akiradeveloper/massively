use crate::common::*;

#[test]
fn sort_returns_device_storage() {
    let exec = exec();
    let x = exec.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let sorted = exec.to_device(&[0.0_f32; 3]).unwrap();

    sort(
        &exec,
        massively::SoA1(x.slice(..)),
        Less,
        massively::SoA1(sorted.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&sorted).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&x).unwrap(), vec![3.0, 1.0, 2.0]);
}

#[test]
fn sort_accepts_builtin_less() {
    let exec = exec();
    let x = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let sorted = exec.to_device(&[0_u32; 3]).unwrap();

    sort(
        &exec,
        massively::SoA1(x.slice(..)),
        massively::op::Less,
        massively::SoA1(sorted.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&sorted).unwrap(), vec![1, 2, 3]);
}

#[test]
fn tuple_sort_accepts_builtin_lexicographical_less() {
    let exec = exec();
    let a = exec.to_device(&[2_u32, 1, 2, 1]).unwrap();
    let b = exec.to_device(&[20_u32, 40, 10, 30]).unwrap();
    let c = exec.to_device(&[200_u32, 400, 100, 300]).unwrap();
    let out_a = exec.to_device(&[0_u32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();
    let out_c = exec.to_device(&[0_u32; 4]).unwrap();

    sort(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        massively::op::Less,
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1, 1, 2, 2]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![30, 40, 10, 20]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![300, 400, 100, 200]);
}

#[test]
fn tuple_sort_preserves_soa_components() {
    let exec = exec();
    let x = exec.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let y = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let out_x = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_y = exec.to_device(&[0_u32; 3]).unwrap();

    sort(
        &exec,
        massively::SoA2(x.slice(..), y.slice(..)),
        MixedTupleLess,
        massively::SoA2(out_x.slice_mut(..), out_y.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_x).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&out_y).unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_accepts_heterogeneous_tuple_comparators_for_two_and_three_columns() {
    let exec = exec();
    let values = exec.to_device(&[2.0_f32, 1.0, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[20_u32, 30, 10, 40]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 4]).unwrap();

    sort(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        MixedTupleLess,
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![30, 10, 20, 40]);

    let values = exec.to_device(&[2.0_f32, 1.0, 4.0, 3.0]).unwrap();
    let tags = exec.to_device(&[20_u32, 10, 20, 10]).unwrap();
    let payload = exec.to_device(&[200.0_f32, 100.0, 400.0, 300.0]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 4]).unwrap();
    let out_payload = exec.to_device(&[0.0_f32; 4]).unwrap();

    sort(
        &exec,
        massively::SoA3(values.slice(..), tags.slice(..), payload.slice(..)),
        MixedTuple3Less,
        massively::SoA3(
            out_values.slice_mut(..),
            out_tags.slice_mut(..),
            out_payload.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(
        exec.to_host(&out_payload).unwrap(),
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
    let out_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_d = exec.to_device(&[0_u32; 4]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_f = exec.to_device(&[0_u32; 4]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 4]).unwrap();

    sort(
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
        massively::SoA7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![2.0, 2.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 20, 30, 40]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![201.0, 200.0, 100.0, 300.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![2010, 2000, 1000, 3000]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![2.1, 2.2, 1.1, 3.3]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![21, 22, 11, 33]);
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
        vec![210.0, 220.0, 110.0, 330.0]
    );
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
    let out_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_d = exec.to_device(&[0_u32; 4]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_f = exec.to_device(&[0_u32; 4]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 4]).unwrap();

    massively::stable_sort(
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
        massively::SoA7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![1, 1, 1, 2]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![300.0, 200.0, 201.0, 100.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![3000, 2000, 2010, 1000]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![30.0, 20.0, 21.0, 10.0]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![300, 200, 201, 100]);
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
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
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_d = exec.to_device(&[0.0_f32; 3]).unwrap();

    sort(
        &exec,
        massively::SoA4(a.slice(..), b.slice(..), c.slice(..), d.slice(..)),
        Tuple4Less,
        massively::SoA4(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![1000.0, 2000.0, 3000.0]);
}
