mod merge;
mod permutation;
mod radix;
mod set;
mod sort;

#[allow(unused_imports)]
pub(crate) use merge::MergeByKeyControl;
pub(crate) use merge::{
    merge, merge_by_key, merge_by_key_control, merge_by_key_values_with_control,
};
#[allow(unused_imports)]
pub(crate) use permutation::Permutation;
pub(crate) use permutation::sort_by_key_permutation;
#[allow(unused_imports)]
pub(crate) use radix::{radix_sort_by_key_u32, radix_sort_u32};
pub(crate) use set::{set_difference, set_intersection, set_union};
pub(crate) use sort::{
    sort, sort_by_key, sort_tuple2, sort_tuple2_by_key, sort_tuple3, sort_tuple3_by_key,
};

pub(super) const BLOCK_ORDERING_SIZE: u32 = 256;
pub(super) const RADIX_DIGITS: usize = 16;
