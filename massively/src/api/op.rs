//! Compile-time operations embedded in GPU algorithms and lazy expressions.

pub use crate::core::op::{ExpandOp, Identity, NonZero, UnaryOp};
pub(crate) use crate::core::op::{IndexedBinaryOp, IndexedUnaryOp, bool_flag};
pub use crate::core::ordering::BinaryPredicateOp;
pub use crate::core::predicate::PredicateOp;
pub use crate::core::reduce::ReductionOp;
