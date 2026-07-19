//! Read-only segment values whose data is supplied by a lowered read expression.

use core::marker::PhantomData;
use std::rc::Rc;

use cubecl::prelude::*;

/// A read-only, dynamically bounded view into another logical iterator.
///
/// `Segment<T>` is a semantic GPU value, not materialized storage.  Its
/// backing reader is bound while a read expression is lowered, so individual
/// segment values only vary by `start` and `len`. Consequently this type does
/// not implement any allocation or write capability.
pub struct Segment<T: CubeType> {
    _item: PhantomData<fn() -> T>,
}

impl<T: CubeType> Clone for Segment<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: CubeType> Copy for Segment<T> {}

impl<T: CubeType> core::fmt::Debug for Segment<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Segment").finish_non_exhaustive()
    }
}

type RuntimeIndex = <u32 as CubeType>::ExpandType;
type Expanded<T> = <T as CubeType>::ExpandType;

/// Type-erased IR builder for the shared backing expression.
///
/// This callback exists only while CubeCL builds a kernel.  It contributes no
/// runtime field to the generated GPU value; the captured staged inputs are
/// the same kernel bindings shared by every `Segment` produced by one read.
pub(crate) type SegmentReader<T> = Rc<dyn Fn(&Scope, RuntimeIndex) -> Expanded<T>>;

/// CubeCL expansion of [`Segment`].
///
/// `start` and `len` are device registers. `reader` is compiler-side context
/// that emits a read from the shared backing expression.
#[doc(hidden)]
pub struct SegmentExpand<T: CubeType> {
    reader: SegmentReader<T>,
    start: RuntimeIndex,
    len: RuntimeIndex,
}

impl<T: CubeType> SegmentExpand<T> {
    pub(crate) fn from_bounds(
        scope: &Scope,
        reader: SegmentReader<T>,
        start: RuntimeIndex,
        end: RuntimeIndex,
    ) -> Self {
        let len = ExpandTypeClone::clone_unchecked(&end)
            .__expand_sub_method(scope, ExpandTypeClone::clone_unchecked(&start));
        Self { reader, start, len }
    }

    /// Expansion hook used for `segment.len()` inside cube functions.
    pub fn __expand_len_method(&self, _scope: &Scope) -> RuntimeIndex {
        ExpandTypeClone::clone_unchecked(&self.len)
    }

    /// Expansion hook used for `segment.is_empty()` inside cube functions.
    pub fn __expand_is_empty_method(&self, scope: &Scope) -> <bool as CubeType>::ExpandType {
        self.__expand_len_method(scope)
            .__expand_eq_method(scope, &NativeExpand::from_lit(scope, 0u32))
    }

    /// Expansion hook used for `segment.at(index)` inside cube functions.
    pub fn __expand_at_method(&self, scope: &Scope, index: RuntimeIndex) -> Expanded<T> {
        let absolute =
            ExpandTypeClone::clone_unchecked(&self.start).__expand_add_method(scope, index);
        (self.reader)(scope, absolute)
    }
}

impl<T: CubeType> ExpandTypeClone for SegmentExpand<T> {
    fn clone_unchecked(&self) -> Self {
        Self {
            reader: Rc::clone(&self.reader),
            start: ExpandTypeClone::clone_unchecked(&self.start),
            len: ExpandTypeClone::clone_unchecked(&self.len),
        }
    }
}

impl<T: CubeType> CubeType for Segment<T> {
    type ExpandType = SegmentExpand<T>;
}

impl<T: CubeType> IntoExpand for SegmentExpand<T> {
    type Expand = Self;

    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self
    }
}

impl<T: CubeType> IntoMut for SegmentExpand<T> {
    fn into_mut(self, scope: &Scope) -> Self {
        Self {
            reader: self.reader,
            start: self.start.into_mut(scope),
            len: self.len.into_mut(scope),
        }
    }
}

impl<T: CubeType> CubeDebug for SegmentExpand<T> {}

impl<T: CubeType> AsRefExpand for SegmentExpand<T> {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}

impl<T: CubeType> AsMutExpand for SegmentExpand<T> {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}

impl<T: CubeType> Segment<T> {
    /// Returns the number of items in this segment.
    pub fn len(&self) -> u32 {
        unreachable!("Segment::len is available only inside a CubeCL function")
    }

    /// Returns whether this segment contains no items.
    pub fn is_empty(&self) -> bool {
        unreachable!("Segment::is_empty is available only inside a CubeCL function")
    }

    /// Reads one item without performing a bounds check.
    ///
    /// Callers must ensure that `index < self.len()`.
    pub fn at(&self, _index: u32) -> T {
        unreachable!("Segment::at is available only inside a CubeCL function")
    }
}
