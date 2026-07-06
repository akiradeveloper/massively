//! Public algorithm API implementation for `massively`.

use cubecl::prelude::Runtime;

use crate::index::MIndex;
use crate::iter::{MIter, MIterMut, MIterReduce};
use crate::op;
use crate::runtime::Executor;

pub use crate::Error;

fn validate_input<R, Input>(exec: &Executor<R>, input: &Input) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
{
    input.validate_executor(exec)
}

fn validate_output<R, Output>(exec: &Executor<R>, output: &Output) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    output.validate_executor(exec)
}

mod indexed;
mod ordering;
mod predicate;
mod reduce;
mod scan;
mod search;
mod selection;
mod set;
mod transform;
mod unique;

pub use indexed::{gather, gather_where, scatter, scatter_where};
pub use ordering::{
    merge, merge_by_key, reverse, sort, sort_by_key, stable_sort, stable_sort_by_key,
};
pub use predicate::{all_of, any_of, count_if, find_if, is_partitioned, none_of, partition};
pub use reduce::{reduce, reduce_by_key};
pub use scan::{
    adjacent_difference, exclusive_scan, exclusive_scan_by_key, inclusive_scan,
    inclusive_scan_by_key,
};
pub use search::{
    adjacent_find, equal, find_first_of, is_sorted, is_sorted_until, lexicographical_compare,
    lower_bound, max_element, min_element, minmax_element, mismatch, upper_bound,
};
pub use selection::{copy_where, fill, remove_where, replace_where};
pub use set::{set_difference, set_intersection, set_union};
pub use transform::{transform, transform_where};
pub use unique::{unique, unique_by_key};
