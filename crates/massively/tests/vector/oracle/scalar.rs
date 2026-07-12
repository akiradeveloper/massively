use oracle_ref::vector as oracle;
use proptest::prelude::*;

use super::common::*;

macro_rules! unary_case {
    ($name:ident, |$exec:ident, $input:ident, $gpu:ident| $body:block) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(
                $input in oracle_vec(0_u32..100),
            ) {
                let $exec = exec();
                let $gpu = $exec.to_device(&$input);
                $body
            }
        }
    };
}

macro_rules! pair_case {
    ($name:ident, |$exec:ident, $left:ident, $right:ident, $left_gpu:ident, $right_gpu:ident| $body:block) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(
                $left in oracle_vec(0_u32..100),
                $right in oracle_vec(0_u32..100),
            ) {
                let $exec = exec();
                let $left_gpu = $exec.to_device(&$left);
                let $right_gpu = $exec.to_device(&$right);
                $body
            }
        }
    };
}

unary_case!(transform, |exec, input, gpu| {
    let output = massively::vector::transform(&exec, lazify(gpu.slice(..)), AddOne).unwrap();
    let mut expected = vec![0; input.len()];
    oracle::transform(&input, AddOne, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(transform_where, |exec, input, gpu| {
    let flags = flags_for(&input);
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&vec![777_u32; input.len()]);
    massively::vector::transform_where(
        &exec,
        lazify(gpu.slice(..)),
        AddOne,
        lazify(flags_gpu.slice(..)),
        output.slice_mut(..),
    )
    .unwrap();
    let mut expected = vec![777; input.len()];
    oracle::transform_where(&input, AddOne, &flags, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(reduce, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::reduce(&exec, lazify(gpu.slice(..)), 7, Sum).unwrap(),
        oracle::reduce(&input, 7, Sum),
    );
});

unary_case!(inclusive_scan, |exec, input, gpu| {
    let output = massively::vector::inclusive_scan(&exec, lazify(gpu.slice(..)), Sum).unwrap();
    prop_assert_eq!(
        exec.to_host(&output).unwrap(),
        oracle::inclusive_scan(&input, Sum)
    );
});

unary_case!(exclusive_scan, |exec, input, gpu| {
    let output = massively::vector::exclusive_scan(&exec, lazify(gpu.slice(..)), 7, Sum).unwrap();
    prop_assert_eq!(
        exec.to_host(&output).unwrap(),
        oracle::exclusive_scan(&input, 7, Sum)
    );
});

unary_case!(adjacent_difference, |exec, input, gpu| {
    let output = massively::vector::adjacent_difference(&exec, lazify(gpu.slice(..)), Sum).unwrap();
    prop_assert_eq!(
        exec.to_host(&output).unwrap(),
        oracle::adjacent_difference(&input, Sum)
    );
});

unary_case!(count_if, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::count_if(&exec, lazify(gpu.slice(..)), Even).unwrap() as usize,
        oracle::count_if(&input, Even),
    );
});

unary_case!(all_of, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::all_of(&exec, lazify(gpu.slice(..)), Even).unwrap(),
        oracle::all_of(&input, Even)
    );
});

unary_case!(any_of, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::any_of(&exec, lazify(gpu.slice(..)), Even).unwrap(),
        oracle::any_of(&input, Even)
    );
});

unary_case!(none_of, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::none_of(&exec, lazify(gpu.slice(..)), Even).unwrap(),
        oracle::none_of(&input, Even)
    );
});

unary_case!(find_if, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::find_if(&exec, lazify(gpu.slice(..)), Even)
            .unwrap()
            .map(|x| x as usize),
        oracle::find_if(&input, Even),
    );
});

unary_case!(is_partitioned, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::is_partitioned(&exec, lazify(gpu.slice(..)), Even).unwrap(),
        oracle::is_partitioned(&input, Even),
    );
});

unary_case!(copy_where, |exec, input, gpu| {
    let flags = flags_for(&input);
    let flags_gpu = exec.to_device(&flags);
    let output =
        massively::vector::copy_where(&exec, lazify(gpu.slice(..)), lazify(flags_gpu.slice(..)))
            .unwrap();
    let expected = oracle::copy_where(&input, &flags);
    prop_assert_eq!(output.len(), expected.len());
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(remove_where, |exec, input, gpu| {
    let flags = flags_for(&input);
    let flags_gpu = exec.to_device(&flags);
    let output =
        massively::vector::remove_where(&exec, lazify(gpu.slice(..)), lazify(flags_gpu.slice(..)))
            .unwrap();
    let expected = oracle::remove_where(&input, &flags);
    prop_assert_eq!(output.len(), expected.len());
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(partition, |exec, input, gpu| {
    let (output, boundary) =
        massively::vector::partition(&exec, lazify(gpu.slice(..)), Even).unwrap();
    let (mut passing, failing) = oracle::partition(&input, Even);
    let expected_boundary = passing.len();
    passing.extend(failing);
    prop_assert_eq!(boundary as usize, expected_boundary);
    prop_assert_eq!(exec.to_host(&output).unwrap(), passing);
});

unary_case!(fill, |exec, input, _gpu| {
    let output = massively::vector::fill(&exec, input.len(), 42_u32).unwrap();
    let mut expected = vec![0; input.len()];
    oracle::fill(42, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(replace_where, |exec, input, _gpu| {
    let flags = flags_for(&input);
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&input);
    massively::vector::replace_where(&exec, 42, lazify(flags_gpu.slice(..)), output.slice_mut(..))
        .unwrap();
    let mut expected = input.clone();
    oracle::replace_where(42, &flags, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(gather, |exec, input, gpu| {
    let indices = indices_for(input.len());
    let indices_gpu = exec.to_device(&indices);
    let output =
        massively::vector::gather(&exec, lazify(gpu.slice(..)), lazify(indices_gpu.slice(..)))
            .unwrap();
    let mut expected = vec![0; input.len()];
    oracle::gather(&input, &indices, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(gather_where, |exec, input, gpu| {
    let indices = indices_for(input.len());
    let indices_gpu = exec.to_device(&indices);
    let flags = flags_for(&input);
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&input);
    massively::vector::gather_where(
        &exec,
        lazify(gpu.slice(..)),
        lazify(indices_gpu.slice(..)),
        lazify(flags_gpu.slice(..)),
        output.slice_mut(..),
    )
    .unwrap();
    let mut expected = input.clone();
    oracle::gather_where(&input, &indices, &flags, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(scatter, |exec, input, gpu| {
    let indices = indices_for(input.len());
    let indices_gpu = exec.to_device(&indices);
    let output = exec.to_device(&vec![0_u32; input.len()]);
    massively::vector::scatter(
        &exec,
        lazify(gpu.slice(..)),
        lazify(indices_gpu.slice(..)),
        output.slice_mut(..),
    )
    .unwrap();
    let mut expected = vec![0; input.len()];
    oracle::scatter(&input, &indices, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(scatter_where, |exec, input, gpu| {
    let indices = indices_for(input.len());
    let indices_gpu = exec.to_device(&indices);
    let flags = flags_for(&input);
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&input);
    massively::vector::scatter_where(
        &exec,
        lazify(gpu.slice(..)),
        lazify(indices_gpu.slice(..)),
        lazify(flags_gpu.slice(..)),
        output.slice_mut(..),
    )
    .unwrap();
    let mut expected = input.clone();
    oracle::scatter_where(&input, &indices, &flags, &mut expected);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

unary_case!(reverse, |exec, input, gpu| {
    let output = massively::vector::reverse(&exec, lazify(gpu.slice(..))).unwrap();
    prop_assert_eq!(exec.to_host(&output).unwrap(), oracle::reverse(&input));
});

pair_case!(equal, |exec, left, right, left_gpu, right_gpu| {
    prop_assert_eq!(
        massively::vector::equal(
            &exec,
            lazify(left_gpu.slice(..)),
            lazify(right_gpu.slice(..)),
            Equal
        )
        .unwrap(),
        oracle::equal(&left, &right, Equal),
    );
});

pair_case!(mismatch, |exec, left, right, left_gpu, right_gpu| {
    prop_assert_eq!(
        massively::vector::mismatch(
            &exec,
            lazify(left_gpu.slice(..)),
            lazify(right_gpu.slice(..)),
            Equal
        )
        .unwrap()
        .map(|x| x as usize),
        oracle::mismatch(&left, &right, Equal),
    );
});

unary_case!(adjacent_find, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::adjacent_find(&exec, lazify(gpu.slice(..)), Equal)
            .unwrap()
            .map(|x| x as usize),
        oracle::adjacent_find(&input, Equal),
    );
});

pair_case!(find_first_of, |exec, left, right, left_gpu, right_gpu| {
    prop_assert_eq!(
        massively::vector::find_first_of(
            &exec,
            lazify(left_gpu.slice(..)),
            lazify(right_gpu.slice(..)),
            Equal
        )
        .unwrap()
        .map(|x| x as usize),
        oracle::find_first_of(&left, &right, Equal),
    );
});

pair_case!(
    lexicographical_compare,
    |exec, left, right, left_gpu, right_gpu| {
        prop_assert_eq!(
            massively::vector::lexicographical_compare(
                &exec,
                lazify(left_gpu.slice(..)),
                lazify(right_gpu.slice(..)),
                Less
            )
            .unwrap(),
            oracle::lexicographical_compare(&left, &right, Less),
        );
    }
);

unary_case!(min_element, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::min_element(&exec, lazify(gpu.slice(..)), Less)
            .unwrap()
            .map(|x| x as usize),
        oracle::min_element(&input, Less),
    );
});

unary_case!(max_element, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::max_element(&exec, lazify(gpu.slice(..)), Less)
            .unwrap()
            .map(|x| x as usize),
        oracle::max_element(&input, Less),
    );
});

unary_case!(minmax_element, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::minmax_element(&exec, lazify(gpu.slice(..)), Less)
            .unwrap()
            .map(|(a, b)| (a as usize, b as usize)),
        oracle::minmax_element(&input, Less),
    );
});

unary_case!(is_sorted, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::is_sorted(&exec, lazify(gpu.slice(..)), Less).unwrap(),
        oracle::is_sorted(&input, Less)
    );
});

unary_case!(is_sorted_until, |exec, input, gpu| {
    prop_assert_eq!(
        massively::vector::is_sorted_until(&exec, lazify(gpu.slice(..)), Less).unwrap() as usize,
        oracle::is_sorted_until(&input, Less),
    );
});

unary_case!(sort, |exec, input, gpu| {
    let output = massively::vector::sort(&exec, lazify(gpu.slice(..)), Less).unwrap();
    prop_assert_eq!(exec.to_host(&output).unwrap(), oracle::sort(&input, Less));
});

unary_case!(unique, |exec, input, gpu| {
    let output = massively::vector::unique(&exec, lazify(gpu.slice(..)), Equal).unwrap();
    let expected = oracle::unique(&input, Equal);
    prop_assert_eq!(output.len(), expected.len());
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

pair_case!(merge, |exec, left, right, _left_gpu, _right_gpu| {
    let mut left = left;
    let mut right = right;
    left.sort();
    right.sort();
    let left_gpu = exec.to_device(&left);
    let right_gpu = exec.to_device(&right);
    let output = massively::vector::merge(
        &exec,
        lazify(left_gpu.slice(..)),
        lazify(right_gpu.slice(..)),
        Less,
    )
    .unwrap();
    prop_assert_eq!(
        exec.to_host(&output).unwrap(),
        oracle::merge(&left, &right, Less)
    );
});

pair_case!(set_union, |exec, left, right, _left_gpu, _right_gpu| {
    let mut left = left;
    let mut right = right;
    left.sort();
    right.sort();
    let left_gpu = exec.to_device(&left);
    let right_gpu = exec.to_device(&right);
    let output = massively::vector::set_union(
        &exec,
        lazify(left_gpu.slice(..)),
        lazify(right_gpu.slice(..)),
        Less,
    )
    .unwrap();
    let expected = oracle::set_union(&left, &right, Less);
    prop_assert_eq!(output.len(), expected.len());
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

pair_case!(set_intersection, |exec,
                              left,
                              right,
                              _left_gpu,
                              _right_gpu| {
    let mut left = left;
    let mut right = right;
    left.sort();
    right.sort();
    let left_gpu = exec.to_device(&left);
    let right_gpu = exec.to_device(&right);
    let output = massively::vector::set_intersection(
        &exec,
        lazify(left_gpu.slice(..)),
        lazify(right_gpu.slice(..)),
        Less,
    )
    .unwrap();
    let expected = oracle::set_intersection(&left, &right, Less);
    prop_assert_eq!(output.len(), expected.len());
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

pair_case!(set_difference, |exec,
                            left,
                            right,
                            _left_gpu,
                            _right_gpu| {
    let mut left = left;
    let mut right = right;
    left.sort();
    right.sort();
    let left_gpu = exec.to_device(&left);
    let right_gpu = exec.to_device(&right);
    let output = massively::vector::set_difference(
        &exec,
        lazify(left_gpu.slice(..)),
        lazify(right_gpu.slice(..)),
        Less,
    )
    .unwrap();
    let expected = oracle::set_difference(&left, &right, Less);
    prop_assert_eq!(output.len(), expected.len());
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected);
});

pair_case!(lower_bound, |exec, left, values, _left_gpu, values_gpu| {
    let mut left = left;
    left.sort();
    let left_gpu = exec.to_device(&left);
    let output = massively::vector::lower_bound(
        &exec,
        lazify(left_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        Less,
    )
    .unwrap();
    prop_assert_eq!(
        exec.to_host(&output).unwrap(),
        oracle::lower_bound(&left, &values, Less)
    );
});

pair_case!(upper_bound, |exec, left, values, _left_gpu, values_gpu| {
    let mut left = left;
    left.sort();
    let left_gpu = exec.to_device(&left);
    let output = massively::vector::upper_bound(
        &exec,
        lazify(left_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        Less,
    )
    .unwrap();
    prop_assert_eq!(
        exec.to_host(&output).unwrap(),
        oracle::upper_bound(&left, &values, Less)
    );
});
