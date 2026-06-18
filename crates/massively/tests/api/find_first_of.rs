use crate::common::*;

#[test]
fn pair_search_accepts_borrowed_heterogeneous_soa12_patterns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 20, 30]).unwrap();
    let c = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0])
        .unwrap();
    let d = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let e = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let f = policy.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let g = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let h = policy.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let i = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let j = policy.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let k = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 200.0, 300.0])
        .unwrap();
    let l = policy
        .to_device(&[1000_u32, 2000, 3000, 2000, 3000])
        .unwrap();

    let na = policy.to_device(&[9.0_f32, 3.0]).unwrap();
    let nb = policy.to_device(&[90_u32, 30]).unwrap();
    let nc = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nd = policy.to_device(&[0_u32, 0]).unwrap();
    let ne = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nf = policy.to_device(&[0_u32, 0]).unwrap();
    let ng = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nh = policy.to_device(&[0_u32, 0]).unwrap();
    let ni = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nj = policy.to_device(&[0_u32, 0]).unwrap();
    let nk = policy.to_device(&[900.0_f32, 300.0]).unwrap();
    let nl = policy.to_device(&[9000_u32, 3000]).unwrap();

    assert_eq!(
        find_first_of(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(&na, &nb, &nc, &nd, &ne, &nf, &ng, &nh, &ni, &nj, &nk, &nl),
            Tuple12MixedEqual,
        )
        .unwrap(),
        Some(2)
    );
}
