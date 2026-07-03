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
            ReadOnlyKernelColumn, ReadOnlySoA, S0, SoA1 as DeviceSoA1, SoA2 as DeviceSoA2,
            SoA3 as DeviceSoA3, SoAView1, SoAView2, SoAView3,
        },
        op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp},
        primitives::{
            ordering as primitive_ordering, range as primitive_range, reduce as primitive_reduce,
            scan as primitive_scan, search as primitive_search, select,
        },
    },
    error::{LengthForCompare, ensure_same_len},
    expr::{DeviceGpuExpr, GpuExpr},
    index::{MIndex, mindex_from_usize},
    kernels::*,
    op::GpuOp,
    runtime::Scalar,
};

#[allow(dead_code)]
const BLOCK_GATHER_WHERE_SIZE: u32 = 256;
const BLOCK_SCATTER_WHERE_SIZE: u32 = 256;
const BLOCK_REPLACE_WHERE_SIZE: u32 = 256;
const BLOCK_UNIQUE_SIZE: u32 = 256;

mod by_key;
mod core;
mod gather;
mod ordering;
mod reduce;
mod scan;
mod scatter;
mod search;
mod selection;

#[allow(unused_imports)]
pub(crate) use core::*;

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

fn validate_key_column<Key, Len>(keys: &Key, len: Len) -> Result<(), Error>
where
    Key: KernelColumn,
    Len: LengthForCompare,
{
    Key::validate(keys)?;
    ensure_same_len(Key::len(keys), len)
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
