//! # Problem
//!
//! Return products whose stock is below the three-day sales target, sorted by
//! urgency descending.
//!
//! # Task
//!
//! Implement `solve(sku, stock, daily_sales) -> reordered rows`.
//!
//! # GPU Algorithm
//!
//! 1. Transform `(stock, daily_sales)` to urgency.
//! 2. Copy rows with non-zero urgency.
//! 3. Sort by urgency and reverse.

mod common;

use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{DeviceVec, Executor, SoA1, SoA2, copy_where, reverse, sort_by_key, transform};

struct InventoryUrgency;

#[cubecl::cube]
impl<B> UnaryOp<B, (u32, u32)> for InventoryUrgency
where
    B: cubecl::prelude::Runtime,
{
    type Env = ();
    type Output = (u32,);

    fn apply(_env: (), input: (u32, u32)) -> (u32,) {
        let stock = input.0;
        let daily_sales = input.1;
        let target = daily_sales * 3_u32;
        if target > stock {
            (target - stock,)
        } else {
            (0_u32,)
        }
    }
}

struct Output<B: cubecl::prelude::Runtime> {
    sku: DeviceVec<B, u32>,
    urgency: DeviceVec<B, u32>,
}

fn solve<B>(
    exec: &Executor<B>,
    sku: DeviceVec<B, u32>,
    stock: DeviceVec<B, u32>,
    daily_sales: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let urgency = exec.constant(stock.len(), 0_u32)?;
    transform(
        exec,
        SoA2(stock.slice(..), daily_sales.slice(..)),
        InventoryUrgency,
        (),
        SoA1(urgency.slice_mut(..)),
    )?;
    let filtered_sku = exec.constant(sku.len(), 0_u32)?;
    let filtered_urgency = exec.constant(urgency.len(), 0_u32)?;
    let len = copy_where(
        exec,
        SoA2(sku.slice(..), urgency.slice(..)),
        urgency.slice(..),
        SoA2(filtered_sku.slice_mut(..), filtered_urgency.slice_mut(..)),
    )?;
    let sorted_urgency = exec.constant(len, 0_u32)?;
    let sorted_sku = exec.constant(len, 0_u32)?;
    sort_by_key(
        exec,
        SoA1(filtered_urgency.slice(..len)),
        SoA1(filtered_sku.slice(..len)),
        common::LessU32,
        SoA1(sorted_urgency.slice_mut(..)),
        SoA1(sorted_sku.slice_mut(..)),
    )?;
    let urgency = exec.constant(len, 0_u32)?;
    let sku = exec.constant(len, 0_u32)?;
    reverse(
        exec,
        SoA1(sorted_urgency.slice(..)),
        SoA1(urgency.slice_mut(..)),
    )?;
    reverse(exec, SoA1(sorted_sku.slice(..)), SoA1(sku.slice_mut(..)))?;
    Ok(Output { sku, urgency })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[100, 200, 300, 400])?,
        exec.to_device(&[10, 2, 50, 1])?,
        exec.to_device(&[3, 2, 10, 4])?,
    )?;
    assert_eq!(exec.to_host(&output.sku)?, vec![400, 200]);
    assert_eq!(exec.to_host(&output.urgency)?, vec![11, 4]);
    Ok(())
}
