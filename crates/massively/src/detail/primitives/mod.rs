pub(crate) mod ordering;
pub(crate) mod range;
pub(crate) mod reduce;
pub(crate) mod scan;
pub(crate) mod search;
pub(crate) mod select;
pub(crate) mod workspace;

pub(crate) use crate::error::ensure_same_len;
pub(crate) use range::fill_slice_with_policy;
