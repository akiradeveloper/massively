//! Transitional internal algorithm/lowering traits.
//!
//! Public `MIter` values still lower through these traits, but v0.32 keeps
//! control construction and payload movement in `detail::api` and
//! `detail::control`. New code should keep this module as a compatibility
//! adapter instead of adding new control-stream ownership here.

use cubecl::prelude::*;

use crate::{
    Error, MItem,
    detail::{
        CubePolicy,
        device::{
            DeviceColumnMutView, DeviceVec, KernelColumn, KernelColumnAt, KernelColumnBindings,
            ReadOnlyKernelColumn, ReadOnlyZip, S0, Zip1 as DeviceZip1, Zip2 as DeviceZip2,
            Zip3 as DeviceZip3, ZipView1, ZipView2, ZipView3,
        },
        op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp},
        primitives::{
            ordering as primitive_ordering, range as primitive_range, scan as primitive_scan,
            select,
        },
    },
    error::ensure_same_len,
    expr::{DeviceGpuExpr, GpuExpr},
    index::{MIndex, mindex_from_usize},
    kernels::*,
    op::GpuOp,
    value::MStorageElement,
};

#[allow(dead_code)]
const BLOCK_GATHER_WHERE_SIZE: u32 = 256;
const BLOCK_REPLACE_WHERE_SIZE: u32 = 256;
const BLOCK_UNIQUE_SIZE: u32 = 256;

pub(in crate::detail) mod by_key;
mod core;
mod gather;
mod kernel;
mod ordering;
mod reduce;
mod scan;
mod scatter;
mod search;
mod selection;

#[allow(unused_imports)]
pub(crate) use core::*;
pub(crate) use kernel::{
    ColumnRead, ConstantRead, CountingRead, GatherRead, ScanByKeyValueItem, SliceRead,
    TransformRead, ZipRead1, ZipRead2, ZipRead3, ZipRead4, ZipRead5, ZipRead6, ZipRead7,
};
pub use kernel::{KernelRead, KernelReadBoundMany, KernelReadReduce};

#[allow(unused_imports)]
pub(crate) use by_key::*;
#[allow(unused_imports)]
pub(crate) use gather::*;
#[allow(unused_imports)]
pub(crate) use ordering::*;
#[allow(unused_imports)]
pub(crate) use reduce::*;
#[allow(unused_imports)]
pub(crate) use scan::*;
#[allow(unused_imports)]
pub(crate) use scatter::*;
#[allow(unused_imports)]
pub(crate) use search::*;
#[allow(unused_imports)]
pub(crate) use selection::*;

fn validate_columns2<A, C>(left: &A, right: &C) -> Result<(), Error>
where
    A: KernelColumn,
    C: KernelColumn<Runtime = A::Runtime>,
{
    A::validate(left)?;
    C::validate(right)?;
    ensure_same_len(C::len(right), A::len(left))
}

fn validate_columns3<A, C, D>(first: &A, second: &C, third: &D) -> Result<(), Error>
where
    A: KernelColumn,
    C: KernelColumn<Runtime = A::Runtime>,
    D: KernelColumn<Runtime = A::Runtime>,
{
    A::validate(first)?;
    C::validate(second)?;
    D::validate(third)?;
    ensure_same_len(C::len(second), A::len(first))?;
    ensure_same_len(D::len(third), A::len(first))
}
