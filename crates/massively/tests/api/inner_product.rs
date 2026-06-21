use crate::common::*;

#[test]
fn inner_product_accepts_scalar_column() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();

    let result =
        inner_product((left.slice(..),), (right.slice(..),), Sum, (0.0_f32,), Sum).unwrap();

    assert_eq!(result, (21.0,));
}

#[cfg(any())]
#[test]
fn inner_product_accepts_tuple_columns() {
    let policy = policy();
    let left_values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let left_ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let right_values = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let right_ids = policy.to_device(&[40_u32, 50, 60]).unwrap();

    let result = inner_product(
        (left_values.slice(..), left_ids.slice(..)),
        (right_values.slice(..), right_ids.slice(..)),
        Sum,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();

    assert_eq!(result, (21.0, 210));
}

#[cfg(any())]
#[test]
fn inner_product_accepts_heterogeneous_soas() {
    let policy = policy();
    let left_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let left_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let right_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let right_b = policy.to_device(&[40_u32, 50, 60]).unwrap();

    let pair = inner_product(
        zip(left_a.slice(..), left_b.slice(..)),
        zip(right_a.slice(..), right_b.slice(..)),
        Sum,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    assert_eq!(pair, (21.0, 210));

    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[1_u32, 2]).unwrap();
    let c = policy.to_device(&[3.0_f32, 4.0]).unwrap();
    let d = policy.to_device(&[3_u32, 4]).unwrap();
    let e = policy.to_device(&[5.0_f32, 6.0]).unwrap();
    let f = policy.to_device(&[5_u32, 6]).unwrap();
    let g = policy.to_device(&[7.0_f32, 8.0]).unwrap();
    let h = policy.to_device(&[7_u32, 8]).unwrap();
    let i = policy.to_device(&[9.0_f32, 10.0]).unwrap();
    let j = policy.to_device(&[9_u32, 10]).unwrap();
    let k = policy.to_device(&[11.0_f32, 12.0]).unwrap();
    let l = policy.to_device(&[11_u32, 12]).unwrap();
    let ra = policy.to_device(&[2.0_f32, 3.0]).unwrap();
    let rb = policy.to_device(&[2_u32, 3]).unwrap();
    let rc = policy.to_device(&[4.0_f32, 5.0]).unwrap();
    let rd = policy.to_device(&[4_u32, 5]).unwrap();
    let re = policy.to_device(&[6.0_f32, 7.0]).unwrap();
    let rf = policy.to_device(&[6_u32, 7]).unwrap();
    let rg = policy.to_device(&[8.0_f32, 9.0]).unwrap();
    let rh = policy.to_device(&[8_u32, 9]).unwrap();
    let ri = policy.to_device(&[10.0_f32, 11.0]).unwrap();
    let rj = policy.to_device(&[10_u32, 11]).unwrap();
    let rk = policy.to_device(&[12.0_f32, 13.0]).unwrap();
    let rl = policy.to_device(&[12_u32, 13]).unwrap();

    let wide = inner_product(
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
        zip12(
            ra.slice(..),
            rb.slice(..),
            rc.slice(..),
            rd.slice(..),
            re.slice(..),
            rf.slice(..),
            rg.slice(..),
            rh.slice(..),
            ri.slice(..),
            rj.slice(..),
            rk.slice(..),
            rl.slice(..),
        ),
        Sum,
        (
            0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32,
            0.0_f32, 0_u32,
        ),
        Sum,
    )
    .unwrap();
    assert_eq!(
        wide,
        (8.0, 8, 16.0, 16, 24.0, 24, 32.0, 32, 40.0, 40, 48.0, 48)
    );

    let lhs_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let lhs_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let rhs_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let rhs_b = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let zipped = inner_product(
        zip(lhs_a.slice(..), lhs_b.slice(..)),
        zip(rhs_a.slice(..), rhs_b.slice(..)),
        Sum,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    assert_eq!(zipped, (21.0, 210));

    let mixed_left_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let mixed_left_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let mixed_right_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let mixed_right_b = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let mixed = inner_product(
        zip(mixed_left_a.slice(..), mixed_left_b.slice(..)),
        zip(mixed_right_a.slice(..), mixed_right_b.slice(..)),
        Sum,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    assert_eq!(mixed, (21.0, 210));

    let mixed_left_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let mixed_left_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let mixed_right_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let mixed_right_b = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let mixed = inner_product(
        zip(mixed_left_a.slice(..), mixed_left_b.slice(..)),
        zip(mixed_right_a.slice(..), mixed_right_b.slice(..)),
        Sum,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    assert_eq!(mixed, (21.0, 210));
}
