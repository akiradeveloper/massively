//! By-key read lowering.
//!
//! By-key algorithms are split into two phases to avoid multiplying key arity
//! by value arity: keys build a control object, then values consume it.

mod ordering;
mod reduce;
pub(in crate::detail) mod scan;
mod selection;

#[allow(unused_imports)]
pub(crate) use ordering::*;
#[allow(unused_imports)]
pub(crate) use reduce::*;
#[allow(unused_imports)]
pub(crate) use scan::*;
#[allow(unused_imports)]
pub(crate) use selection::*;
