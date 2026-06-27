use crate::common::*;

#[test]
fn transform_zip_output_returns_storage() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        PairMixedSplit,
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_returns_device_storage() {
    let exec = exec();
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        PairMixedSplit,
        massively::SoA2(values.slice_mut(..), tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_tuple_output_maps_to_mitem_storage() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let values_out = exec.to_device(&[0.0_f32; 3]).unwrap();
    let flags = exec.to_device(&[0_u32; 3]).unwrap();
    let bias_out = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA3(values.slice(..), tags.slice(..), bias.slice(..)),
        Tuple3MixedSplit,
        massively::SoA3(
            values_out.slice_mut(..),
            flags.slice_mut(..),
            bias_out.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&values_out).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
    assert_eq!(exec.to_host(&flags).unwrap(), vec![11, 21, 31]);
    assert_eq!(exec.to_host(&bias_out).unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn tuple1_transform_returns_soa1_storage() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA1(input.slice(..)),
        Double,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 4.0, 6.0]);
}

#[cfg(any())]
#[test]
fn unary_transform_accepts_wide_tuple_outputs() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = transform(&exec, massively::SoA1(input.slice(..)), ScalarToTuple5Mixed).unwrap();
    let (a, b, c, d, e) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![2.0, 3.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![3, 4, 5]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![4.0, 5.0, 6.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![5, 6, 7]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![6.0, 7.0, 8.0]);
}

#[cfg(any())]
#[test]
fn unary_transform_accepts_tuple12_output_and_checks_every_column() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20]).unwrap();

    let output = transform(
        &exec,
        massively::SoA1(input.slice(..)),
        ScalarToTuple12Mixed,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![11.0, 21.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![12, 22]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![13.0, 23.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![14, 24]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![15.0, 25.0]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![16, 26]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![17.0, 27.0]);
    assert_eq!(exec.to_host(&h).unwrap(), vec![18, 28]);
    assert_eq!(exec.to_host(&i).unwrap(), vec![19.0, 29.0]);
    assert_eq!(exec.to_host(&j).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&k).unwrap(), vec![21.0, 31.0]);
    assert_eq!(exec.to_host(&l).unwrap(), vec![22, 32]);
}

#[test]
fn tuple_transform_uses_flat_soa_input() {
    let exec = exec();
    let lhs = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let rhs = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tags = exec.to_device(&[0_u32; 3]).unwrap();
    let adjusted_bias = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA3(lhs.slice(..), rhs.slice(..), bias.slice(..)),
        Tuple3MixedSplit,
        massively::SoA3(
            values.slice_mut(..),
            tags.slice_mut(..),
            adjusted_bias.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![11, 21, 31]);
    assert_eq!(
        exec.to_host(&adjusted_bias).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
}

#[test]
fn transform_accepts_heterogeneous_tuple_inputs() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let pair_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let pair_tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        PairMixedSplit,
        massively::SoA2(pair_values.slice_mut(..), pair_tags.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&pair_values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&pair_tags).unwrap(), vec![11, 21, 31]);

    let tuple_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tuple_tags = exec.to_device(&[0_u32; 3]).unwrap();
    let tuple_bias = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA3(values.slice(..), tags.slice(..), bias.slice(..)),
        Tuple3MixedSplit,
        massively::SoA3(
            tuple_values.slice_mut(..),
            tuple_tags.slice_mut(..),
            tuple_bias.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&tuple_values).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
    assert_eq!(exec.to_host(&tuple_tags).unwrap(), vec![11, 21, 31]);
    assert_eq!(
        exec.to_host(&tuple_bias).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
}

#[cfg(any())]
#[test]
fn transform_accepts_soa4_heterogeneous_inputs_and_checks_every_column() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000, 3000]).unwrap();

    let output = transform(
        &exec,
        zip4(a.slice(..), b.slice(..), c.slice(..), d.slice(..)),
        Tuple4MixedSplit,
    )
    .unwrap();
    let (a, b, c, d) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![12, 22, 32]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![1004, 2004, 3004]);
}

#[cfg(any())]
#[test]
fn transform_accepts_mismatched_input_and_output_tuple_widths() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000]).unwrap();
    let e = exec.to_device(&[10000.0_f32, 20000.0]).unwrap();

    let out_5_to_3 = transform(
        &exec,
        zip5(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
        ),
        Tuple5To3MixedSplit,
    )
    .unwrap();
    let (x, y, z) = out_5_to_3;
    assert_eq!(exec.to_host(&x).unwrap(), vec![10101.0, 20202.0]);
    assert_eq!(exec.to_host(&y).unwrap(), vec![1010, 2020]);
    assert_eq!(exec.to_host(&z).unwrap(), vec![9999.0, 19998.0]);

    let out_3_to_5 = transform(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        Tuple3To5MixedSplit,
    )
    .unwrap();
    let (x, y, z, w, v) = out_3_to_5;
    assert_eq!(exec.to_host(&x).unwrap(), vec![101.0, 202.0]);
    assert_eq!(exec.to_host(&y).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&z).unwrap(), vec![99.0, 198.0]);
    assert_eq!(exec.to_host(&w).unwrap(), vec![30, 40]);
    assert_eq!(exec.to_host(&v).unwrap(), vec![100.0, 400.0]);
}

#[cfg(any())]
#[test]
fn transform_accepts_extreme_mismatched_tuple_widths() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();

    let expanded = transform(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        Tuple2To12MixedExpand,
    )
    .unwrap();
    let (a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6) = expanded;
    assert_eq!(exec.to_host(&a1).unwrap(), vec![2.0, 3.0]);
    assert_eq!(exec.to_host(&b1).unwrap(), vec![12, 22]);
    assert_eq!(exec.to_host(&a2).unwrap(), vec![4.0, 5.0]);
    assert_eq!(exec.to_host(&b2).unwrap(), vec![14, 24]);
    assert_eq!(exec.to_host(&a3).unwrap(), vec![6.0, 7.0]);
    assert_eq!(exec.to_host(&b3).unwrap(), vec![16, 26]);
    assert_eq!(exec.to_host(&a4).unwrap(), vec![8.0, 9.0]);
    assert_eq!(exec.to_host(&b4).unwrap(), vec![18, 28]);
    assert_eq!(exec.to_host(&a5).unwrap(), vec![10.0, 11.0]);
    assert_eq!(exec.to_host(&b5).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&a6).unwrap(), vec![12.0, 13.0]);
    assert_eq!(exec.to_host(&b6).unwrap(), vec![22, 32]);

    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000]).unwrap();
    let e = exec.to_device(&[10000.0_f32, 20000.0]).unwrap();
    let f = exec.to_device(&[100000_u32, 200000]).unwrap();
    let g = exec.to_device(&[1000000.0_f32, 2000000.0]).unwrap();
    let h = exec.to_device(&[7_u32, 8]).unwrap();
    let i = exec.to_device(&[70.0_f32, 80.0]).unwrap();
    let j = exec.to_device(&[700_u32, 800]).unwrap();
    let k = exec.to_device(&[7000.0_f32, 8000.0]).unwrap();
    let l = exec.to_device(&[70000_u32, 80000]).unwrap();

    let projected = transform(
        &exec,
        zip12(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
            h.slice(..),
            i.slice(..),
            j.slice(..),
            k.slice(..),
            l.slice(..),
        ),
        Tuple12To2MixedProject,
    )
    .unwrap();
    let (x, y) = projected;
    assert_eq!(exec.to_host(&x).unwrap(), vec![7101.0, 8202.0]);
    assert_eq!(exec.to_host(&y).unwrap(), vec![170010, 280020]);
}

#[cfg(any())]
#[test]
fn transform_accepts_soa5_to_soa11_heterogeneous_tuple_outputs() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32]).unwrap();
    let b = exec.to_device(&[10_u32]).unwrap();
    let c = exec.to_device(&[100.0_f32]).unwrap();
    let d = exec.to_device(&[1000_u32]).unwrap();
    let e = exec.to_device(&[10000.0_f32]).unwrap();
    let f = exec.to_device(&[100000_u32]).unwrap();
    let g = exec.to_device(&[1000000.0_f32]).unwrap();
    let h = exec.to_device(&[7_u32]).unwrap();
    let i = exec.to_device(&[70.0_f32]).unwrap();
    let j = exec.to_device(&[700_u32]).unwrap();
    let k = exec.to_device(&[7000.0_f32]).unwrap();

    let (a5, b5, c5, d5, e5) = transform(
        &exec,
        zip5(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
        ),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(exec.to_host(&a5).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b5).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c5).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d5).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e5).unwrap(), vec![10005.0]);

    let (a6, b6, c6, d6, e6, f6) = transform(
        &exec,
        zip6(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
        ),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(exec.to_host(&a6).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b6).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c6).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d6).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e6).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f6).unwrap(), vec![100006]);

    let (a7, b7, c7, d7, e7, f7, g7) = transform(
        &exec,
        zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(exec.to_host(&a7).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b7).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c7).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d7).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e7).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f7).unwrap(), vec![100006]);
    assert_eq!(exec.to_host(&g7).unwrap(), vec![1000007.0]);

    let (a8, b8, c8, d8, e8, f8, g8, h8) = transform(
        &exec,
        zip8(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
            h.slice(..),
        ),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(exec.to_host(&a8).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b8).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c8).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d8).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e8).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f8).unwrap(), vec![100006]);
    assert_eq!(exec.to_host(&g8).unwrap(), vec![1000007.0]);
    assert_eq!(exec.to_host(&h8).unwrap(), vec![15]);

    let (a9, b9, c9, d9, e9, f9, g9, h9, i9) = transform(
        &exec,
        zip9(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
            h.slice(..),
            i.slice(..),
        ),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(exec.to_host(&a9).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b9).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c9).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d9).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e9).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f9).unwrap(), vec![100006]);
    assert_eq!(exec.to_host(&g9).unwrap(), vec![1000007.0]);
    assert_eq!(exec.to_host(&h9).unwrap(), vec![15]);
    assert_eq!(exec.to_host(&i9).unwrap(), vec![79.0]);

    let (a10, b10, c10, d10, e10, f10, g10, h10, i10, j10) = transform(
        &exec,
        zip10(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
            h.slice(..),
            i.slice(..),
            j.slice(..),
        ),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(exec.to_host(&a10).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b10).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c10).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d10).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e10).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f10).unwrap(), vec![100006]);
    assert_eq!(exec.to_host(&g10).unwrap(), vec![1000007.0]);
    assert_eq!(exec.to_host(&h10).unwrap(), vec![15]);
    assert_eq!(exec.to_host(&i10).unwrap(), vec![79.0]);
    assert_eq!(exec.to_host(&j10).unwrap(), vec![710]);

    let (a11, b11, c11, d11, e11, f11, g11, h11, i11, j11, k11) = transform(
        &exec,
        zip11(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
            h.slice(..),
            i.slice(..),
            j.slice(..),
            k.slice(..),
        ),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(exec.to_host(&a11).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b11).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c11).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d11).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e11).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f11).unwrap(), vec![100006]);
    assert_eq!(exec.to_host(&g11).unwrap(), vec![1000007.0]);
    assert_eq!(exec.to_host(&h11).unwrap(), vec![15]);
    assert_eq!(exec.to_host(&i11).unwrap(), vec![79.0]);
    assert_eq!(exec.to_host(&j11).unwrap(), vec![710]);
    assert_eq!(exec.to_host(&k11).unwrap(), vec![7011.0]);
}

#[cfg(any())]
#[test]
fn transform_accepts_soa12_heterogeneous_inputs_and_checks_every_column() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000]).unwrap();
    let e = exec.to_device(&[10000.0_f32, 20000.0]).unwrap();
    let f = exec.to_device(&[100000_u32, 200000]).unwrap();
    let g = exec.to_device(&[1000000.0_f32, 2000000.0]).unwrap();
    let h = exec.to_device(&[7_u32, 8]).unwrap();
    let i = exec.to_device(&[70.0_f32, 80.0]).unwrap();
    let j = exec.to_device(&[700_u32, 800]).unwrap();
    let k = exec.to_device(&[7000.0_f32, 8000.0]).unwrap();
    let l = exec.to_device(&[70000_u32, 80000]).unwrap();

    let output = transform(
        &exec,
        zip12(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
            h.slice(..),
            i.slice(..),
            j.slice(..),
            k.slice(..),
            l.slice(..),
        ),
        Tuple12MixedSplit,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![2.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![12, 22]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![103.0, 203.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![1004, 2004]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![10005.0, 20005.0]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![100006, 200006]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![1000007.0, 2000007.0]);
    assert_eq!(exec.to_host(&h).unwrap(), vec![15, 16]);
    assert_eq!(exec.to_host(&i).unwrap(), vec![79.0, 89.0]);
    assert_eq!(exec.to_host(&j).unwrap(), vec![710, 810]);
    assert_eq!(exec.to_host(&k).unwrap(), vec![7011.0, 8011.0]);
    assert_eq!(exec.to_host(&l).unwrap(), vec![70012, 80012]);
}

#[cfg(any())]
#[test]
fn transform_accepts_soa12_heterogeneous_inputs_to_tuple1_output() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000]).unwrap();
    let e = exec.to_device(&[3.0_f32, 4.0]).unwrap();
    let f = exec.to_device(&[30_u32, 40]).unwrap();
    let g = exec.to_device(&[300.0_f32, 400.0]).unwrap();
    let h = exec.to_device(&[3000_u32, 4000]).unwrap();
    let i = exec.to_device(&[5.0_f32, 6.0]).unwrap();
    let j = exec.to_device(&[50_u32, 60]).unwrap();
    let k = exec.to_device(&[500.0_f32, 600.0]).unwrap();
    let l = exec.to_device(&[5000_u32, 6000]).unwrap();

    let mut output = exec.to_device(&[0.0_f32; 2]).unwrap();
    transform(
        &exec,
        zip12(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
            h.slice(..),
            i.slice(..),
            j.slice(..),
            k.slice(..),
            l.slice(..),
        ),
        Tuple12MixedChecksum,
        (output.slice_mut(..),),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![9999.0, 13332.0]);
}

#[test]
fn transform_zip_flattens_soa1_columns() {
    let exec = exec();
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        PairMixedSplit,
        massively::SoA2(values.slice_mut(..), tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![11, 21, 31]);
}
