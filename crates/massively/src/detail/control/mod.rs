//! Internal GPU control streams.
//!
//! A control stream describes where rows move or where logical segments begin.
//! Algorithms build a control stream once, then apply it to every carried Zip
//! column so row semantics stay synchronized.

mod ordering;
mod range;
mod search;
mod segment;
mod selection;

#[allow(unused_imports)]
pub(crate) use ordering::{MergeByKeyControl, MergeControl, OrderingControl, PermutationControl};
pub(crate) use range::{RangeControl, RangeMapping};
pub(crate) use search::SearchControl;
#[allow(unused_imports)]
pub(crate) use segment::{ReduceByKeyControl, ScanByKeyControl, SegmentControl};
#[allow(unused_imports)]
pub(crate) use selection::{
    MaskControl, SelectedRankControl, SplitRankControl, UniqueByKeyControl,
};
