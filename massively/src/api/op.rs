//! Compile-time operations embedded in GPU algorithms and lazy expressions.

pub use crate::core::op::{Identity, UnaryOp};
pub(crate) use crate::core::op::{IndexedBinaryOp, IndexedUnaryOp};
#[doc(hidden)]
pub use crate::core::op::{U32ToBool, U32ToUsize};
pub use crate::core::ordering::BinaryPredicateOp;
pub use crate::core::predicate::PredicateOp;
pub use crate::core::reduce::ReductionOp;
