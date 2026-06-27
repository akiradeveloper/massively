use std::any::Any;

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch::{self as sealed, array_from_inner, column_view_at};
use crate::detail::op_adapter::{
    KernelOp, KernelTuple1InnerProductOp, KernelTuple1Op, StencilFlag,
};
use crate::error::ensure_same_len;
use crate::iter::{MIter, MIterMut, SoA1, SoA2, SoA3};
use crate::op;
use crate::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor, Scalar};
use crate::value::{MItem, MVec};

mod item;
mod iter;
