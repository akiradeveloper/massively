//! Logical index type used by massively device-side APIs.

use crate::Error;

/// Logical length, position, offset, and count type used by massively.
///
/// `MIndex` represents the index space used by device algorithms. Host
/// APIs such as Rust slices and byte allocations still use `usize` at their
/// boundaries, and conversions are kept explicit at those boundaries.
pub type MIndex = u32;

pub(crate) fn mindex_from_usize(len: usize) -> Result<MIndex, Error> {
    MIndex::try_from(len).map_err(|_| Error::LengthTooLarge { len })
}

pub(crate) fn usize_from_mindex(len: MIndex) -> usize {
    len as usize
}

pub(crate) trait IntoMIndex {
    fn into_mindex(self) -> MIndex;
}

impl IntoMIndex for MIndex {
    fn into_mindex(self) -> MIndex {
        self
    }
}

impl IntoMIndex for usize {
    fn into_mindex(self) -> MIndex {
        mindex_from_usize(self).expect("length exceeds MIndex")
    }
}
