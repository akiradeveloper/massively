mod merge;
mod radix;
mod set;
mod sort;

#[allow(unused_imports)]
pub(crate) use merge::MergeByKeyControl;
pub(crate) use merge::{
    merge_by_key_control_with_policy, merge_by_key_values_with_control_with_policy,
    merge_by_key_with_policy, merge_with_policy,
};
#[allow(unused_imports)]
pub(crate) use radix::{radix_sort_by_key_u32, radix_sort_u32};
pub(crate) use set::{
    set_difference_with_policy, set_intersection_with_policy, set_union_with_policy,
};
pub(crate) use sort::{
    sort_by_key_input_with_policy, sort_input_with_policy, sort_tuple2, sort_tuple2_by_key,
    sort_tuple2_by_key_input, sort_tuple2_input, sort_tuple3, sort_tuple3_by_key,
    sort_tuple3_input,
};

pub(super) const BLOCK_ORDERING_SIZE: u32 = 256;
pub(super) const RADIX_DIGITS: usize = 16;
