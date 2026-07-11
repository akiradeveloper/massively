use cubecl::prelude::*;
use massively::{op::*, *};
use oracle_ref::{op, vector as oracle};

use super::common::exec;

const A8_SCALE_LEN: usize = 65_537;

type Seven = ((((((u32, u32), u32), u32), u32), u32), u32);

struct MaxSeven;

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

impl op::ReductionOp<Seven> for MaxSeven {
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

#[test]
fn lazify_reduce_at_read_arity_8() {
    let columns: [Vec<u32>; 7] = core::array::from_fn(|column| {
        (0..A8_SCALE_LEN)
            .map(|index| ((index * (column + 3) + column * 17) % 100_003) as u32)
            .collect()
    });
    let aos: Vec<Seven> = (0..A8_SCALE_LEN)
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
        .collect();
    let exec = exec();
    let device: Vec<_> = columns
        .iter()
        .map(|column| exec.to_device(column))
        .collect();
    let lazified = lazy::identity(lazy::permute(
        zip7(
            device[0].slice(..),
            device[1].slice(..),
            device[2].slice(..),
            device[3].slice(..),
            device[4].slice(..),
            device[5].slice(..),
            device[6].slice(..),
        ),
        lazy::counting(0).take(A8_SCALE_LEN as MIndex),
    ));
    let zero: Seven = ((((((0, 0), 0), 0), 0), 0), 0);

    assert_eq!(
        massively::vector::reduce(&exec, lazified, zero, MaxSeven).unwrap(),
        oracle::reduce(&aos, zero, MaxSeven),
    );
}
