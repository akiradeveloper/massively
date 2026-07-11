use oracle_ref::vector as oracle;
use proptest::prelude::*;

use super::common::*;

macro_rules! by_key_case {
    ($name:ident, |$exec:ident, $keys:ident, $values:ident, $key_gpu:ident, $value_gpu:ident| $body:block) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(
                pairs in oracle_vec((0_u32..12, 0_u32..50)),
            ) {
                let (raw_keys, raw_values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                let $keys = raw_keys.as_slice();
                let $values = raw_values.as_slice();
                let $exec = exec();
                let $key_gpu = $exec.to_device($keys);
                let $value_gpu = $exec.to_device($values);
                $body
            }
        }
    };
}

by_key_case!(
    inclusive_scan_by_key,
    |exec, keys, values, key_gpu, value_gpu| {
        let output = exec.to_device(&vec![0_u32; keys.len()]);
        massively::vector::inclusive_scan_by_key(
            &exec,
            lazify(key_gpu.slice(..)),
            lazify(value_gpu.slice(..)),
            Equal,
            Sum,
            output.slice_mut(..),
        )
        .unwrap();
        prop_assert_eq!(
            exec.to_host(&output).unwrap(),
            oracle::inclusive_scan_by_key(keys, values, Equal, Sum),
        );
    }
);

by_key_case!(
    exclusive_scan_by_key,
    |exec, keys, values, key_gpu, value_gpu| {
        let output = exec.to_device(&vec![0_u32; keys.len()]);
        massively::vector::exclusive_scan_by_key(
            &exec,
            lazify(key_gpu.slice(..)),
            lazify(value_gpu.slice(..)),
            Equal,
            7,
            Sum,
            output.slice_mut(..),
        )
        .unwrap();
        prop_assert_eq!(
            exec.to_host(&output).unwrap(),
            oracle::exclusive_scan_by_key(keys, values, Equal, 7, Sum),
        );
    }
);

by_key_case!(reduce_by_key, |exec, keys, values, key_gpu, value_gpu| {
    let out_keys = exec.to_device(&vec![0_u32; keys.len()]);
    let out_values = exec.to_device(&vec![0_u32; keys.len()]);
    let len = massively::vector::reduce_by_key(
        &exec,
        lazify(key_gpu.slice(..)),
        lazify(value_gpu.slice(..)),
        Equal,
        7,
        Sum,
        out_keys.slice_mut(..),
        out_values.slice_mut(..),
    )
    .unwrap();
    let (expected_keys, expected_values) = oracle::reduce_by_key(keys, values, Equal, 7, Sum);
    prop_assert_eq!(len as usize, expected_keys.len());
    prop_assert_eq!(
        exec.to_host(&out_keys.slice(..len as usize)).unwrap(),
        expected_keys,
    );
    prop_assert_eq!(
        exec.to_host(&out_values.slice(..len as usize)).unwrap(),
        expected_values,
    );
});

by_key_case!(unique_by_key, |exec, keys, values, key_gpu, value_gpu| {
    let out_keys = exec.to_device(&vec![0_u32; keys.len()]);
    let out_values = exec.to_device(&vec![0_u32; keys.len()]);
    let len = massively::vector::unique_by_key(
        &exec,
        lazify(key_gpu.slice(..)),
        lazify(value_gpu.slice(..)),
        Equal,
        out_keys.slice_mut(..),
        out_values.slice_mut(..),
    )
    .unwrap();
    let (expected_keys, expected_values) = oracle::unique_by_key(keys, values, Equal);
    prop_assert_eq!(len as usize, expected_keys.len());
    prop_assert_eq!(
        exec.to_host(&out_keys.slice(..len as usize)).unwrap(),
        expected_keys,
    );
    prop_assert_eq!(
        exec.to_host(&out_values.slice(..len as usize)).unwrap(),
        expected_values,
    );
});

by_key_case!(sort_by_key, |exec, keys, values, key_gpu, value_gpu| {
    let out_keys = exec.to_device(&vec![0_u32; keys.len()]);
    let out_values = exec.to_device(&vec![0_u32; keys.len()]);
    massively::vector::sort_by_key(
        &exec,
        lazify(key_gpu.slice(..)),
        lazify(value_gpu.slice(..)),
        Less,
        out_keys.slice_mut(..),
        out_values.slice_mut(..),
    )
    .unwrap();
    let (expected_keys, expected_values) = oracle::sort_by_key(keys, values, Less);
    prop_assert_eq!(exec.to_host(&out_keys).unwrap(), expected_keys);
    prop_assert_eq!(exec.to_host(&out_values).unwrap(), expected_values);
});

by_key_case!(merge_by_key, |exec, keys, values, _key_gpu, _value_gpu| {
    let split = keys.len() / 2;
    let (left_keys, left_values) = oracle::sort_by_key(&keys[..split], &values[..split], Less);
    let (right_keys, right_values) = oracle::sort_by_key(&keys[split..], &values[split..], Less);
    let left_keys_gpu = exec.to_device(&left_keys);
    let left_values_gpu = exec.to_device(&left_values);
    let right_keys_gpu = exec.to_device(&right_keys);
    let right_values_gpu = exec.to_device(&right_values);
    let out_keys = exec.to_device(&vec![0_u32; keys.len()]);
    let out_values = exec.to_device(&vec![0_u32; keys.len()]);
    massively::vector::merge_by_key(
        &exec,
        lazify(left_keys_gpu.slice(..)),
        lazify(left_values_gpu.slice(..)),
        lazify(right_keys_gpu.slice(..)),
        lazify(right_values_gpu.slice(..)),
        Less,
        out_keys.slice_mut(..),
        out_values.slice_mut(..),
    )
    .unwrap();
    let (expected_keys, expected_values) =
        oracle::merge_by_key(&left_keys, &left_values, &right_keys, &right_values, Less);
    prop_assert_eq!(exec.to_host(&out_keys).unwrap(), expected_keys);
    prop_assert_eq!(exec.to_host(&out_values).unwrap(), expected_values);
});
