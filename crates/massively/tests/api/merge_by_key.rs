use crate::common::*;

#[allow(unused_macros)]
macro_rules! soa12_rows {
    ($exec:expr; [$( $x:expr ),+ $(,)?]) => {{
        let a = $exec.to_device(&[$(($x as f32)),+]).unwrap();
        let b = $exec.to_device(&[$(($x as u32) * 10),+]).unwrap();
        let c = $exec.to_device(&[$(($x as f32) * 100.0),+]).unwrap();
        let d = $exec.to_device(&[$(($x as u32) * 1000),+]).unwrap();
        let e = $exec.to_device(&[$(($x as f32) + 10.0),+]).unwrap();
        let f = $exec.to_device(&[$(($x as u32) + 100),+]).unwrap();
        let g = $exec.to_device(&[$(($x as f32) + 1000.0),+]).unwrap();
        let h = $exec.to_device(&[$(($x as u32) + 10000),+]).unwrap();
        let i = $exec.to_device(&[$(($x as f32) + 20.0),+]).unwrap();
        let j = $exec.to_device(&[$(($x as u32) + 200),+]).unwrap();
        let k = $exec.to_device(&[$(($x as f32) + 2000.0),+]).unwrap();
        let l = $exec.to_device(&[$(($x as u32) + 20000),+]).unwrap();
        (a, b, c, d, e, f, g, h, i, j, k, l)
    }};
}

#[allow(unused_macros)]
macro_rules! assert_soa12_rows {
    ($output:expr; [$( $x:expr ),* $(,)?]) => {{
        let (a, b, c, d, e, f, g, h, i, j, k, l) = $output;
        assert_eq!(exec.to_host(&a).unwrap(), vec![$(($x as f32)),*]);
        assert_eq!(exec.to_host(&b).unwrap(), vec![$(($x as u32) * 10),*]);
        assert_eq!(exec.to_host(&c).unwrap(), vec![$(($x as f32) * 100.0),*]);
        assert_eq!(exec.to_host(&d).unwrap(), vec![$(($x as u32) * 1000),*]);
        assert_eq!(exec.to_host(&e).unwrap(), vec![$(($x as f32) + 10.0),*]);
        assert_eq!(exec.to_host(&f).unwrap(), vec![$(($x as u32) + 100),*]);
        assert_eq!(exec.to_host(&g).unwrap(), vec![$(($x as f32) + 1000.0),*]);
        assert_eq!(exec.to_host(&h).unwrap(), vec![$(($x as u32) + 10000),*]);
        assert_eq!(exec.to_host(&i).unwrap(), vec![$(($x as f32) + 20.0),*]);
        assert_eq!(exec.to_host(&j).unwrap(), vec![$(($x as u32) + 200),*]);
        assert_eq!(exec.to_host(&k).unwrap(), vec![$(($x as f32) + 2000.0),*]);
        assert_eq!(exec.to_host(&l).unwrap(), vec![$(($x as u32) + 20000),*]);
    }};
}

#[test]
fn merge_by_key_accepts_tuple_values() {
    let exec = exec();
    let left_keys = exec.to_device(&[0_u32, 2, 2, 5]).unwrap();
    let right_keys = exec.to_device(&[1_u32, 2, 4]).unwrap();
    let left_values = exec.to_device(&[0.0_f32, 20.0, 21.0, 50.0]).unwrap();
    let left_ids = exec.to_device(&[0_u32, 20, 21, 50]).unwrap();
    let right_values = exec.to_device(&[10.0_f32, 22.0, 40.0]).unwrap();
    let right_ids = exec.to_device(&[10_u32, 22, 40]).unwrap();

    let (keys, values) = merge_by_key(
        &exec,
        massively::SoA1(left_keys.slice(..)),
        massively::SoA2(left_values.slice(..), left_ids.slice(..)),
        massively::SoA1(right_keys.slice(..)),
        massively::SoA2(right_values.slice(..), right_ids.slice(..)),
        LessU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (values, ids) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        exec.to_host(&values).unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(exec.to_host(&ids).unwrap(), vec![0, 10, 20, 21, 22, 40, 50]);
}
