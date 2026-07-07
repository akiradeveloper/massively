//! Massively logical item traits.

use cubecl::prelude::{CubeElement, CubePrimitive, CubeType, Runtime};

use crate::Error;
use crate::detail::dispatch;
use crate::index::MIndex;
use crate::iter::{MIterMut, MStorage};
use crate::op;
use crate::runtime::Executor;

/// Logical item handled by massively algorithms.
///
/// An `MItem` is one element of an [`crate::iter::MIter`]. It is a kernel value
/// shape, not a storage layout promise.
pub trait MItem<R: Runtime>: CubeType + Copy + Sized + Send + Sync + 'static {}

impl<R, T> MItem<R> for T
where
    R: Runtime,
    T: CubeType + Copy + Sized + Send + Sync + 'static,
{
}

/// Physical element that can be stored in a device column.
pub trait MStorageElement:
    CubePrimitive + CubeElement + crate::expr::LogicalItemPack7 + crate::expr::LogicalPackLeaf
{
}
impl<T> MStorageElement for T where
    T: CubePrimitive + CubeElement + crate::expr::LogicalItemPack7 + crate::expr::LogicalPackLeaf
{
}

/// Logical item that has an owned/writable Zip device storage shape.
pub trait MAlloc<R: Runtime>:
    MItem<R> + dispatch::MItemDispatch<R> + crate::detail::write::MItemWriteDispatch<R>
{
    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    type View;

    #[doc(hidden)]
    type Storage: MStorage<R, Item = Self>;

    #[doc(hidden)]
    fn storage_from_inner(inner: Self::Inner) -> Self::Storage;

    #[doc(hidden)]
    fn alloc_storage(exec: &Executor<R>, len: MIndex) -> Result<Self::Storage, Error>;

    #[doc(hidden)]
    fn reverse_from_view<Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, input, output);
        Err(Error::Launch {
            message: "reverse is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn sort_from_view<Less, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, input, less, output);
        Err(Error::Launch {
            message: "sort is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn sort_by_key_control_from_view<Less>(
        policy: &crate::detail::CubePolicy<R>,
        keys: Self::View,
        less: Less,
    ) -> Result<(Self::Inner, crate::detail::DeviceVec<R, MIndex>), Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
    {
        let _ = (policy, keys, less);
        Err(Error::Launch {
            message: "sort_by_key keys are not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn sort_by_key_values_from_view<Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        control: &crate::detail::control::PermutationControl<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, control, output);
        Err(Error::Launch {
            message: "sort_by_key values are not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn merge_by_key_control_from_views<Less>(
        policy: &crate::detail::CubePolicy<R>,
        left_keys: Self::View,
        right_keys: Self::View,
        less: Less,
    ) -> Result<(Self::Inner, crate::detail::control::MergeByKeyControl), Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
    {
        let _ = (policy, left_keys, right_keys, less);
        Err(Error::Launch {
            message: "merge_by_key keys are not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn merge_by_key_values_from_views<Output>(
        policy: &crate::detail::CubePolicy<R>,
        left_values: Self::View,
        right_values: Self::View,
        control: &crate::detail::control::MergeByKeyControl,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, left_values, right_values, control, output);
        Err(Error::Launch {
            message: "merge_by_key values are not supported for this item storage shape"
                .to_string(),
        })
    }

    #[doc(hidden)]
    fn unique_by_key_control_from_view<Eq>(
        policy: &crate::detail::CubePolicy<R>,
        keys: Self::View,
        eq: Eq,
    ) -> Result<(Self::Inner, crate::detail::control::UniqueByKeyControl), Error>
    where
        Eq: op::BinaryPredicateOp<R, Self>,
    {
        let _ = (policy, keys, eq);
        Err(Error::Launch {
            message: "unique_by_key keys are not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn unique_by_key_values_from_view<Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        control: &crate::detail::control::UniqueByKeyControl,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, control, output);
        Err(Error::Launch {
            message: "unique_by_key values are not supported for this item storage shape"
                .to_string(),
        })
    }

    #[doc(hidden)]
    fn reduce_by_key_control_from_view<KeyEq>(
        policy: &crate::detail::CubePolicy<R>,
        keys: Self::View,
        key_eq: KeyEq,
    ) -> Result<(Self::Inner, crate::detail::control::ReduceByKeyControl<R>), Error>
    where
        KeyEq: op::BinaryPredicateOp<R, Self>,
    {
        let _ = (policy, keys, key_eq);
        Err(Error::Launch {
            message: "reduce_by_key keys are not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn reduce_by_key_values_from_view<KeyEq, Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        control: &crate::detail::control::ReduceByKeyControl<R>,
        init: Self,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, control, init, op, output);
        Err(Error::Launch {
            message: "reduce_by_key values are not supported for this item storage shape"
                .to_string(),
        })
    }

    #[doc(hidden)]
    fn inclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::Inner,
        control: &crate::detail::control::ScanByKeyControl<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, control, op, output);
        Err(Error::Launch {
            message: "scan_by_key values are not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn exclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::Inner,
        control: &crate::detail::control::ScanByKeyControl<R>,
        init: Self,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, control, init, op, output);
        Err(Error::Launch {
            message: "scan_by_key values are not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn copy_selected_from_view<Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, stencil, output);
        Err(Error::Launch {
            message: "copy_where is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn gather_from_view<IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        indices: IndexSource,
        output: Output,
    ) -> Result<(), Error>
    where
        IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, indices, output);
        Err(Error::Launch {
            message: "gather is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn gather_where_from_view<IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        indices: IndexSource,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, indices, stencil, output);
        Err(Error::Launch {
            message: "gather_where is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn scatter_from_view<IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        indices: IndexSource,
        output: Output,
    ) -> Result<(), Error>
    where
        IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, indices, output);
        Err(Error::Launch {
            message: "scatter is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn scatter_where_from_view<IndexSource, Output>(
        policy: &crate::detail::CubePolicy<R>,
        values: Self::View,
        indices: IndexSource,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, values, indices, stencil, output);
        Err(Error::Launch {
            message: "scatter_where is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn transform_from_view<Output, Op>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: MAlloc<R> + dispatch::MItemDispatch<R>,
        Op: op::UnaryOp<R, Self, Output = Output::Item>,
    {
        let _ = (policy, input, op, output);
        Err(Error::Launch {
            message: "transform is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn transform_where_from_view<Output, Op>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        op: Op,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: MAlloc<R> + dispatch::MItemDispatch<R>,
        Op: op::UnaryOp<R, Self, Output = Output::Item>,
    {
        let _ = (policy, input, op, stencil, output);
        Err(Error::Launch {
            message: "transform_where is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn unique_from_view<Pred, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Pred: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, input, pred, output);
        Err(Error::Launch {
            message: "unique is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn reduce_from_view<Op>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        init: Self,
        op: Op,
    ) -> Result<Self, Error>
    where
        Op: op::ReductionOp<R, Self>,
    {
        let _ = (policy, input, init, op);
        Err(Error::Launch {
            message: "reduce is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn min_element_from_view<Less>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
    {
        let _ = (policy, input, less);
        Err(Error::Launch {
            message: "min_element is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn max_element_from_view<Less>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
    {
        let _ = (policy, input, less);
        Err(Error::Launch {
            message: "max_element is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn minmax_element_from_view<Less>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        less: Less,
    ) -> Result<Option<(MIndex, MIndex)>, Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
    {
        let _ = (policy, input, less);
        Err(Error::Launch {
            message: "minmax_element is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn partition_from_view<Pred, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Pred: op::PredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, input, pred, output);
        Err(Error::Launch {
            message: "partition is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn adjacent_difference_from_view<Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, input, op, output);
        Err(Error::Launch {
            message: "adjacent_difference is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn inclusive_scan_from_view<Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, input, op, output);
        Err(Error::Launch {
            message: "inclusive_scan is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn exclusive_scan_from_view<Op, Output>(
        policy: &crate::detail::CubePolicy<R>,
        input: Self::View,
        init: Self,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, input, init, op, output);
        Err(Error::Launch {
            message: "exclusive_scan is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn merge_from_views<Less, Output>(
        policy: &crate::detail::CubePolicy<R>,
        left: Self::View,
        right: Self::View,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, left, right, less, output);
        Err(Error::Launch {
            message: "merge is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn set_union_from_views<Less, Output>(
        policy: &crate::detail::CubePolicy<R>,
        left: Self::View,
        right: Self::View,
        less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, left, right, less, output);
        Err(Error::Launch {
            message: "set_union is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn set_intersection_from_views<Less, Output>(
        policy: &crate::detail::CubePolicy<R>,
        left: Self::View,
        right: Self::View,
        less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, left, right, less, output);
        Err(Error::Launch {
            message: "set_intersection is not supported for this item storage shape".to_string(),
        })
    }

    #[doc(hidden)]
    fn set_difference_from_views<Less, Output>(
        policy: &crate::detail::CubePolicy<R>,
        left: Self::View,
        right: Self::View,
        less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Less: op::BinaryPredicateOp<R, Self>,
        Output: MIterMut<R, Item = Self>,
    {
        let _ = (policy, left, right, less, output);
        Err(Error::Launch {
            message: "set_difference is not supported for this item storage shape".to_string(),
        })
    }
}

#[doc(hidden)]
pub trait StorageFromInner<R: Runtime>: Sized {
    type Item: MAlloc<R>;

    fn from_inner(inner: <Self::Item as MAlloc<R>>::Inner) -> Self;

    #[doc(hidden)]
    fn into_inner(self) -> <Self::Item as MAlloc<R>>::Inner;

    fn len(&self) -> MIndex;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
