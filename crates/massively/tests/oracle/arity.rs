use cubecl::prelude::*;
use massively::*;
use oracle_ref as oracle;
use proptest::prelude::*;

use super::common::*;

type Seven = ((((((u32, u32), u32), u32), u32), u32), u32);

type Two = (u32, u32);
type Three = (Two, u32);
type Four = (Three, u32);
type Five = (Four, u32);
type Six = (Five, u32);

struct IdentitySeven;
struct MaxSeven;
struct Project1;
struct Project2;
struct Project3;
struct Project4;
struct Project5;
struct Project6;
struct Project7;

macro_rules! impl_project_ops {
    ($input:ty, $value:ident => $seed:expr) => {
        #[cubecl::cube]
        impl UnaryOp<$input> for Project1 {
            type Output = u32;

            fn apply($value: $input) -> u32 {
                let seed = $seed;
                seed ^ 0x5a5a_5a5a
            }
        }

        #[cubecl::cube]
        impl UnaryOp<$input> for Project2 {
            type Output = Two;

            fn apply($value: $input) -> Two {
                let seed = $seed;
                (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5)
            }
        }

        #[cubecl::cube]
        impl UnaryOp<$input> for Project3 {
            type Output = Three;

            fn apply($value: $input) -> Three {
                let seed = $seed;
                (
                    (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5),
                    (seed >> 1) ^ 0x3c3c_3c3c,
                )
            }
        }

        #[cubecl::cube]
        impl UnaryOp<$input> for Project4 {
            type Output = Four;

            fn apply($value: $input) -> Four {
                let seed = $seed;
                (
                    (
                        (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5),
                        (seed >> 1) ^ 0x3c3c_3c3c,
                    ),
                    (seed << 2) ^ 0xc3c3_c3c3,
                )
            }
        }

        #[cubecl::cube]
        impl UnaryOp<$input> for Project5 {
            type Output = Five;

            fn apply($value: $input) -> Five {
                let seed = $seed;
                (
                    (
                        (
                            (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5),
                            (seed >> 1) ^ 0x3c3c_3c3c,
                        ),
                        (seed << 2) ^ 0xc3c3_c3c3,
                    ),
                    (seed >> 2) ^ 0x0f0f_0f0f,
                )
            }
        }

        #[cubecl::cube]
        impl UnaryOp<$input> for Project6 {
            type Output = Six;

            fn apply($value: $input) -> Six {
                let seed = $seed;
                (
                    (
                        (
                            (
                                (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5),
                                (seed >> 1) ^ 0x3c3c_3c3c,
                            ),
                            (seed << 2) ^ 0xc3c3_c3c3,
                        ),
                        (seed >> 2) ^ 0x0f0f_0f0f,
                    ),
                    (seed << 3) ^ 0xf0f0_f0f0,
                )
            }
        }

        #[cubecl::cube]
        impl UnaryOp<$input> for Project7 {
            type Output = Seven;

            fn apply($value: $input) -> Seven {
                let seed = $seed;
                (
                    (
                        (
                            (
                                (
                                    (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5),
                                    (seed >> 1) ^ 0x3c3c_3c3c,
                                ),
                                (seed << 2) ^ 0xc3c3_c3c3,
                            ),
                            (seed >> 2) ^ 0x0f0f_0f0f,
                        ),
                        (seed << 3) ^ 0xf0f0_f0f0,
                    ),
                    (seed >> 3) ^ 0x9696_9696,
                )
            }
        }
    };
}

impl_project_ops!(u32, input => input);
impl_project_ops!(Two, input => input.0 ^ (input.1 << 1));
impl_project_ops!(Three, input => input.0.0 ^ (input.0.1 << 1) ^ (input.1 << 2));
impl_project_ops!(Four, input => input.0.0.0 ^ (input.0.0.1 << 1) ^ (input.0.1 << 2) ^ (input.1 << 3));
impl_project_ops!(Five, input => input.0.0.0.0 ^ (input.0.0.0.1 << 1) ^ (input.0.0.1 << 2) ^ (input.0.1 << 3) ^ (input.1 << 4));
impl_project_ops!(Six, input => input.0.0.0.0.0 ^ (input.0.0.0.0.1 << 1) ^ (input.0.0.0.1 << 2) ^ (input.0.0.1 << 3) ^ (input.0.1 << 4) ^ (input.1 << 5));
impl_project_ops!(Seven, input => input.0.0.0.0.0.0 ^ (input.0.0.0.0.0.1 << 1) ^ (input.0.0.0.0.1 << 2) ^ (input.0.0.0.1 << 3) ^ (input.0.0.1 << 4) ^ (input.0.1 << 5) ^ (input.1 << 6));

#[cubecl::cube]
impl UnaryOp<Seven> for IdentitySeven {
    type Output = Seven;

    fn apply(input: Seven) -> Seven {
        input
    }
}

impl oracle::op::UnaryOp<Seven> for IdentitySeven {
    type Output = Seven;

    fn apply(input: Seven) -> Seven {
        input
    }
}

fn max_seven(lhs: Seven, rhs: Seven) -> Seven {
    (
        (
            (
                (
                    (
                        (
                            lhs.0.0.0.0.0.0.max(rhs.0.0.0.0.0.0),
                            lhs.0.0.0.0.0.1.max(rhs.0.0.0.0.0.1),
                        ),
                        lhs.0.0.0.0.1.max(rhs.0.0.0.0.1),
                    ),
                    lhs.0.0.0.1.max(rhs.0.0.0.1),
                ),
                lhs.0.0.1.max(rhs.0.0.1),
            ),
            lhs.0.1.max(rhs.0.1),
        ),
        lhs.1.max(rhs.1),
    )
}

#[cubecl::cube]
impl ReductionOp<Seven> for MaxSeven {
    fn apply(lhs: Seven, rhs: Seven) -> Seven {
        (
            (
                (
                    (
                        (
                            (
                                lhs.0.0.0.0.0.0.max(rhs.0.0.0.0.0.0),
                                lhs.0.0.0.0.0.1.max(rhs.0.0.0.0.0.1),
                            ),
                            lhs.0.0.0.0.1.max(rhs.0.0.0.0.1),
                        ),
                        lhs.0.0.0.1.max(rhs.0.0.0.1),
                    ),
                    lhs.0.0.1.max(rhs.0.0.1),
                ),
                lhs.0.1.max(rhs.0.1),
            ),
            lhs.1.max(rhs.1),
        )
    }
}

impl oracle::op::ReductionOp<Seven> for MaxSeven {
    fn apply(lhs: Seven, rhs: Seven) -> Seven {
        max_seven(lhs, rhs)
    }
}

fn seven_columns(seed: &[u32]) -> [Vec<u32>; 7] {
    core::array::from_fn(|column| seed.iter().map(|value| value + column as u32).collect())
}

fn seven_aos(columns: &[Vec<u32>; 7]) -> Vec<Seven> {
    (0..columns[0].len())
        .map(|index| {
            (
                (
                    (
                        (
                            ((columns[0][index], columns[1][index]), columns[2][index]),
                            columns[3][index],
                        ),
                        columns[4][index],
                    ),
                    columns[5][index],
                ),
                columns[6][index],
            )
        })
        .collect()
}

fn project(seed: u32, column: usize) -> u32 {
    match column {
        0 => seed ^ 0x5a5a_5a5a,
        1 => (seed << 1) ^ 0xa5a5_a5a5,
        2 => (seed >> 1) ^ 0x3c3c_3c3c,
        3 => (seed << 2) ^ 0xc3c3_c3c3,
        4 => (seed >> 2) ^ 0x0f0f_0f0f,
        5 => (seed << 3) ^ 0xf0f0_f0f0,
        6 => (seed >> 3) ^ 0x9696_9696,
        _ => unreachable!(),
    }
}

macro_rules! raw_input_expr {
    (1, $d:ident) => {
        $d[0].slice(..)
    };
    (2, $d:ident) => {
        zip2($d[0].slice(..), $d[1].slice(..))
    };
    (3, $d:ident) => {
        zip3($d[0].slice(..), $d[1].slice(..), $d[2].slice(..))
    };
    (4, $d:ident) => {
        zip4(
            $d[0].slice(..),
            $d[1].slice(..),
            $d[2].slice(..),
            $d[3].slice(..),
        )
    };
    (5, $d:ident) => {
        zip5(
            $d[0].slice(..),
            $d[1].slice(..),
            $d[2].slice(..),
            $d[3].slice(..),
            $d[4].slice(..),
        )
    };
    (6, $d:ident) => {
        zip6(
            $d[0].slice(..),
            $d[1].slice(..),
            $d[2].slice(..),
            $d[3].slice(..),
            $d[4].slice(..),
            $d[5].slice(..),
        )
    };
    (7, $d:ident) => {
        zip7(
            $d[0].slice(..),
            $d[1].slice(..),
            $d[2].slice(..),
            $d[3].slice(..),
            $d[4].slice(..),
            $d[5].slice(..),
            $d[6].slice(..),
        )
    };
}

macro_rules! input_expr {
    ($arity:tt, $device:ident) => {
        lazify(raw_input_expr!($arity, $device))
    };
}

macro_rules! output_expr {
    (1, $o:ident) => {
        $o[0].slice_mut(..)
    };
    (2, $o:ident) => {
        zip2($o[0].slice_mut(..), $o[1].slice_mut(..))
    };
    (3, $o:ident) => {
        zip3(
            $o[0].slice_mut(..),
            $o[1].slice_mut(..),
            $o[2].slice_mut(..),
        )
    };
    (4, $o:ident) => {
        zip4(
            $o[0].slice_mut(..),
            $o[1].slice_mut(..),
            $o[2].slice_mut(..),
            $o[3].slice_mut(..),
        )
    };
    (5, $o:ident) => {
        zip5(
            $o[0].slice_mut(..),
            $o[1].slice_mut(..),
            $o[2].slice_mut(..),
            $o[3].slice_mut(..),
            $o[4].slice_mut(..),
        )
    };
    (6, $o:ident) => {
        zip6(
            $o[0].slice_mut(..),
            $o[1].slice_mut(..),
            $o[2].slice_mut(..),
            $o[3].slice_mut(..),
            $o[4].slice_mut(..),
            $o[5].slice_mut(..),
        )
    };
    (7, $o:ident) => {
        zip7(
            $o[0].slice_mut(..),
            $o[1].slice_mut(..),
            $o[2].slice_mut(..),
            $o[3].slice_mut(..),
            $o[4].slice_mut(..),
            $o[5].slice_mut(..),
            $o[6].slice_mut(..),
        )
    };
}

macro_rules! input_seed {
    (1, $columns:ident, $index:ident) => {
        $columns[0][$index]
    };
    (2, $columns:ident, $index:ident) => {
        $columns[0][$index] ^ ($columns[1][$index] << 1)
    };
    (3, $columns:ident, $index:ident) => {
        input_seed!(2, $columns, $index) ^ ($columns[2][$index] << 2)
    };
    (4, $columns:ident, $index:ident) => {
        input_seed!(3, $columns, $index) ^ ($columns[3][$index] << 3)
    };
    (5, $columns:ident, $index:ident) => {
        input_seed!(4, $columns, $index) ^ ($columns[4][$index] << 4)
    };
    (6, $columns:ident, $index:ident) => {
        input_seed!(5, $columns, $index) ^ ($columns[5][$index] << 5)
    };
    (7, $columns:ident, $index:ident) => {
        input_seed!(6, $columns, $index) ^ ($columns[6][$index] << 6)
    };
}

macro_rules! transform_arity_case {
    ($name:ident, $input_arity:tt, $output_arity:tt, $op:ident) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(seed in oracle_vec(0_u32..100)) {
                let exec = exec();
                let columns = seven_columns(&seed);
                let device: Vec<_> = columns
                    .iter()
                    .map(|column| exec.to_device(column))
                    .collect();
                let output: Vec<_> = (0..$output_arity)
                    .map(|_| exec.to_device(&vec![0_u32; seed.len()]))
                    .collect();

                massively::transform(&exec, input_expr!($input_arity, device),
                    $op,
                    output_expr!($output_arity, output),
                )
                .unwrap();

                let seeds: Vec<_> = (0..seed.len())
                    .map(|index| input_seed!($input_arity, columns, index))
                    .collect();
                for (column, actual) in output.iter().enumerate() {
                    let expected: Vec<_> = seeds
                        .iter()
                        .map(|seed| project(*seed, column))
                        .collect();
                    prop_assert_eq!(exec.to_host(actual).unwrap(), expected);
                }
            }
        }
    };
}

macro_rules! transform_arity_row {
    ($input:tt; $($name:ident, $output:tt, $op:ident);+ $(;)?) => {
        $(transform_arity_case!($name, $input, $output, $op);)+
    };
}

transform_arity_row!(1;
    transform_1_to_1, 1, Project1;
    transform_1_to_2, 2, Project2;
    transform_1_to_3, 3, Project3;
    transform_1_to_4, 4, Project4;
    transform_1_to_5, 5, Project5;
    transform_1_to_6, 6, Project6;
    transform_1_to_7, 7, Project7;
);
transform_arity_row!(2;
    transform_2_to_1, 1, Project1;
    transform_2_to_2, 2, Project2;
    transform_2_to_3, 3, Project3;
    transform_2_to_4, 4, Project4;
    transform_2_to_5, 5, Project5;
    transform_2_to_6, 6, Project6;
    transform_2_to_7, 7, Project7;
);
transform_arity_row!(3;
    transform_3_to_1, 1, Project1;
    transform_3_to_2, 2, Project2;
    transform_3_to_3, 3, Project3;
    transform_3_to_4, 4, Project4;
    transform_3_to_5, 5, Project5;
    transform_3_to_6, 6, Project6;
    transform_3_to_7, 7, Project7;
);
transform_arity_row!(4;
    transform_4_to_1, 1, Project1;
    transform_4_to_2, 2, Project2;
    transform_4_to_3, 3, Project3;
    transform_4_to_4, 4, Project4;
    transform_4_to_5, 5, Project5;
    transform_4_to_6, 6, Project6;
    transform_4_to_7, 7, Project7;
);
transform_arity_row!(5;
    transform_5_to_1, 1, Project1;
    transform_5_to_2, 2, Project2;
    transform_5_to_3, 3, Project3;
    transform_5_to_4, 4, Project4;
    transform_5_to_5, 5, Project5;
    transform_5_to_6, 6, Project6;
    transform_5_to_7, 7, Project7;
);
transform_arity_row!(6;
    transform_6_to_1, 1, Project1;
    transform_6_to_2, 2, Project2;
    transform_6_to_3, 3, Project3;
    transform_6_to_4, 4, Project4;
    transform_6_to_5, 5, Project5;
    transform_6_to_6, 6, Project6;
    transform_6_to_7, 7, Project7;
);
transform_arity_row!(7;
    transform_7_to_1, 1, Project1;
    transform_7_to_2, 2, Project2;
    transform_7_to_3, 3, Project3;
    transform_7_to_4, 4, Project4;
    transform_7_to_5, 5, Project5;
    transform_7_to_6, 6, Project6;
    transform_7_to_7, 7, Project7;
);

macro_rules! transform_where_arity_case {
    ($name:ident, $input_arity:tt, $output_arity:tt, $op:ident) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(seed in oracle_vec(0_u32..100)) {
                let exec = exec();
                let columns = seven_columns(&seed);
                let device: Vec<_> = columns
                    .iter()
                    .map(|column| exec.to_device(column))
                    .collect();
                let flags = flags_for(&seed);
                let flags_device = exec.to_device(&flags);
                let output: Vec<_> = (0..$output_arity)
                    .map(|_| exec.to_device(&vec![777_u32; seed.len()]))
                    .collect();

                massively::transform_where(
                    &exec,
                    input_expr!($input_arity, device),
                    $op,
                    lazify(flags_device.slice(..)),
                    output_expr!($output_arity, output),
                )
                .unwrap();

                let seeds: Vec<_> = (0..seed.len())
                    .map(|index| input_seed!($input_arity, columns, index))
                    .collect();
                for (column, actual) in output.iter().enumerate() {
                    let expected: Vec<_> = seeds
                        .iter()
                        .zip(flags.iter())
                        .map(|(seed, flag)| if *flag != 0 { project(*seed, column) } else { 777 })
                        .collect();
                    prop_assert_eq!(exec.to_host(actual).unwrap(), expected);
                }
            }
        }
    };
}

macro_rules! transform_where_arity_row {
    ($input:tt; $($name:ident, $output:tt, $op:ident);+ $(;)?) => {
        $(transform_where_arity_case!($name, $input, $output, $op);)+
    };
}

transform_where_arity_row!(1;
    transform_where_1_to_1, 1, Project1;
    transform_where_1_to_2, 2, Project2;
    transform_where_1_to_3, 3, Project3;
    transform_where_1_to_4, 4, Project4;
    transform_where_1_to_5, 5, Project5;
    transform_where_1_to_6, 6, Project6;
    transform_where_1_to_7, 7, Project7;
);
transform_where_arity_row!(2;
    transform_where_2_to_1, 1, Project1;
    transform_where_2_to_2, 2, Project2;
    transform_where_2_to_3, 3, Project3;
    transform_where_2_to_4, 4, Project4;
    transform_where_2_to_5, 5, Project5;
    transform_where_2_to_6, 6, Project6;
    transform_where_2_to_7, 7, Project7;
);
transform_where_arity_row!(3;
    transform_where_3_to_1, 1, Project1;
    transform_where_3_to_2, 2, Project2;
    transform_where_3_to_3, 3, Project3;
    transform_where_3_to_4, 4, Project4;
    transform_where_3_to_5, 5, Project5;
    transform_where_3_to_6, 6, Project6;
    transform_where_3_to_7, 7, Project7;
);
transform_where_arity_row!(4;
    transform_where_4_to_1, 1, Project1;
    transform_where_4_to_2, 2, Project2;
    transform_where_4_to_3, 3, Project3;
    transform_where_4_to_4, 4, Project4;
    transform_where_4_to_5, 5, Project5;
    transform_where_4_to_6, 6, Project6;
    transform_where_4_to_7, 7, Project7;
);
transform_where_arity_row!(5;
    transform_where_5_to_1, 1, Project1;
    transform_where_5_to_2, 2, Project2;
    transform_where_5_to_3, 3, Project3;
    transform_where_5_to_4, 4, Project4;
    transform_where_5_to_5, 5, Project5;
    transform_where_5_to_6, 6, Project6;
    transform_where_5_to_7, 7, Project7;
);
transform_where_arity_row!(6;
    transform_where_6_to_1, 1, Project1;
    transform_where_6_to_2, 2, Project2;
    transform_where_6_to_3, 3, Project3;
    transform_where_6_to_4, 4, Project4;
    transform_where_6_to_5, 5, Project5;
    transform_where_6_to_6, 6, Project6;
    transform_where_6_to_7, 7, Project7;
);
transform_where_arity_row!(7;
    transform_where_7_to_1, 1, Project1;
    transform_where_7_to_2, 2, Project2;
    transform_where_7_to_3, 3, Project3;
    transform_where_7_to_4, 4, Project4;
    transform_where_7_to_5, 5, Project5;
    transform_where_7_to_6, 6, Project6;
    transform_where_7_to_7, 7, Project7;
);

proptest! {
    #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]

    /// Seven value leaves plus the permutation index consume all eight read slots.
    #[test]
    fn lazify_dispatches_through_eval8(
        seed in oracle_vec(0_u32..100),
    ) {
        let exec = exec();
        let columns = seven_columns(&seed);
        let device: Vec<_> = columns.iter().map(|column| exec.to_device(column)).collect();
        let input = || {
            lazy::identity(lazy::permute(
                zip7(
                    device[0].slice(..),
                    device[1].slice(..),
                    device[2].slice(..),
                    device[3].slice(..),
                    device[4].slice(..),
                    device[5].slice(..),
                    device[6].slice(..),
                ),
                lazy::counting(0).take(seed.len() as MIndex),
            ))
        };

        let output: Vec<_> = (0..7)
            .map(|_| exec.to_device(&vec![0_u32; seed.len()]))
            .collect();
        massively::transform(&exec, input(),
            IdentitySeven,
            zip7(
                output[0].slice_mut(..),
                output[1].slice_mut(..),
                output[2].slice_mut(..),
                output[3].slice_mut(..),
                output[4].slice_mut(..),
                output[5].slice_mut(..),
                output[6].slice_mut(..),
            ),
        )
        .unwrap();
        for (actual, expected) in output.iter().zip(columns.iter()) {
            prop_assert_eq!(exec.to_host(actual).unwrap(), expected.clone());
        }

        let zero: Seven = ((((((0, 0), 0), 0), 0), 0), 0);
        prop_assert_eq!(
            massively::reduce(&exec, input(), zero, MaxSeven).unwrap(),
            oracle::reduce(&seven_aos(&columns), zero, MaxSeven),
        );
    }
}
