//! Read-only slice abstractions.

mod device;
pub(crate) mod lowering;
pub mod op;
mod traits;

pub use op::{TabulateOp, TransformOp};
pub use traits::MSlice;
