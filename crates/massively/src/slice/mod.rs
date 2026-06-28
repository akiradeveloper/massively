//! Read-only slice abstractions.

mod device;
mod lazy;
pub(crate) mod lowering;
mod traits;

pub(crate) use lazy::{ConstantRead, TabulateRead, TransformRead};
pub use lazy::{constant_slice, tabulate_slice, transform_slice};
pub use traits::MSlice;
