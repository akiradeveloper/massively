//! Internal dispatch for writing logical read expressions into physical output.

use cubecl::prelude::Runtime;

use crate::detail::api::PrecomputedSelection;
use crate::detail::control::SelectedRankControl;
use crate::detail::read::KernelReadBoundMany;
use crate::error::Error;
use crate::index::MIndex;
use crate::iter::MIterMut;
use crate::op;
use crate::value::MAlloc;

/// Write-side operations for logical read expressions.
///
/// This is intentionally internal. `MAlloc` describes allocatable storage for an
/// item, while this trait describes how an already-lowered read expression is
/// applied to an output iterator.
#[doc(hidden)]
#[allow(unused_variables)]
pub trait MItemWriteDispatch<R: Runtime>: Sized {
    fn copy_selected_from_read<Read, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Read,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "copy_where is not supported for this item write shape".to_string(),
        })
    }

    fn gather_from_read<Read, IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Read,
        indices: IndexSource,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        IndexSource: KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "gather is not supported for this item write shape".to_string(),
        })
    }

    fn gather_where_from_read<Read, IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Read,
        indices: IndexSource,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        IndexSource: KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "gather_where is not supported for this item write shape".to_string(),
        })
    }

    fn scatter_from_read<Read, IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Read,
        indices: IndexSource,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        IndexSource: KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "scatter is not supported for this item write shape".to_string(),
        })
    }

    fn scatter_where_from_read<Read, IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Read,
        indices: IndexSource,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        IndexSource: KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "scatter_where is not supported for this item write shape".to_string(),
        })
    }

    fn unique_from_read<Read, Pred, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Read,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        Pred: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "unique is not supported for this item write shape".to_string(),
        })
    }

    fn partition_from_read<Read, Pred, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Read,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        Pred: op::PredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "partition is not supported for this item write shape".to_string(),
        })
    }

    fn adjacent_difference_from_read<Read, Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Read,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "adjacent_difference is not supported for this item write shape".to_string(),
        })
    }

    fn inclusive_scan_from_read<Read, Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Read,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "inclusive_scan is not supported for this item write shape".to_string(),
        })
    }

    fn exclusive_scan_from_read<Read, Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Read,
        init: Self,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "exclusive_scan is not supported for this item write shape".to_string(),
        })
    }

    fn reduce_by_key_values_from_read<KeyEq, Op, Read, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Read,
        selection: &SelectedRankControl,
        output_count: usize,
        init: Self,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Read: KernelReadBoundMany<R, Item = Self>,
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "reduce_by_key values are not supported for this item write shape".to_string(),
        })
    }

    fn merge_from_read<Left, Right, Less, Output>(
        policy: &crate::detail::CubePolicy<R>,
        left: Left,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MAlloc<R>,
        Left: KernelReadBoundMany<R, Item = Self>,
        Right: KernelReadBoundMany<R, Item = Self>,
        Less: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "merge is not supported for this item write shape".to_string(),
        })
    }

    fn set_union_from_read<Left, Right, Less, Output>(
        policy: &crate::detail::CubePolicy<R>,
        left: Left,
        right: Right,
        right_only: &SelectedRankControl,
        less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MAlloc<R>,
        Left: KernelReadBoundMany<R, Item = Self>,
        Right: KernelReadBoundMany<R, Item = Self>,
        Less: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = ();
        Err(Error::Launch {
            message: "set_union is not supported for this item write shape".to_string(),
        })
    }
}
