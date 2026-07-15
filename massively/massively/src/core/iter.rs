//! Iterator composition types.

/// A binary zip node.
///
/// Repeated zipping creates a binary tree.  No arity-specific public zip types
/// are needed; for example `Zip(a, Zip(b, c))` has the semantic item shape
/// `(A, (B, C))`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Zip<X, Y>(pub X, pub Y);

impl<X, Y> Zip<X, Y> {
    /// Creates a binary zip node.
    pub const fn new(left: X, right: Y) -> Self {
        Self(left, right)
    }

    /// Returns the two child nodes.
    pub fn into_parts(self) -> (X, Y) {
        (self.0, self.1)
    }
}
