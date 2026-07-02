//! Internal GPU control streams.
//!
//! A control stream describes where rows move or where logical segments begin.
//! Algorithms build a control stream once, then apply it to every carried SoA
//! column so row semantics stay synchronized.

mod ordering;
mod segment;
mod selection;

#[allow(unused_imports)]
pub(crate) use ordering::{MergeByKeyControl, MergeControl, PermutationControl};
#[allow(unused_imports)]
pub(crate) use segment::{ReduceByKeyControl, ScanByKeyControl, SegmentControl};
#[allow(unused_imports)]
pub(crate) use selection::{SelectionControl, SelectionHandles, UniqueByKeyControl};
