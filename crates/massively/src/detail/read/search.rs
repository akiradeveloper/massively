use super::*;

pub(crate) trait KernelMinMaxInput<Less>: Sized {
    type Runtime: Runtime;

    fn min_element_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error>;

    fn max_element_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error>;

    fn minmax_element_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<(MIndex, MIndex)>, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelAdjacentFindInput<Pred>: Sized {
    type Runtime: Runtime;

    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Option<MIndex>, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelSortedSearchInput<Less>: Sized {
    type Runtime: Runtime;
    type Item;

    fn lower_bound_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<MIndex, Error>;

    fn upper_bound_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<MIndex, Error>;

    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<MIndex, Error>;

    fn is_sorted_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<bool, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelSortedSearchManyInput<Values, Less>: Sized {
    type Runtime: Runtime;

    fn lower_bound_many_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        less: GpuOp<Less>,
    ) -> Result<crate::detail::DeviceVec<Self::Runtime, MIndex>, Error>;

    fn upper_bound_many_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        less: GpuOp<Less>,
    ) -> Result<crate::detail::DeviceVec<Self::Runtime, MIndex>, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelPairSearchInput<Other, Op>: Sized {
    type Runtime: Runtime;

    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<bool, Error>;

    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error>;

    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error>;

    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<bool, Error>;
}
