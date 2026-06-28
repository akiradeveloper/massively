//! Read-only slice abstractions.

mod device;
mod lazy;
pub(crate) mod lowering;
mod traits;

pub(crate) use lazy::{ConstantRead, TransformRead};
pub use lazy::{constant_slice, transform_slice};
pub use traits::MSlice;
