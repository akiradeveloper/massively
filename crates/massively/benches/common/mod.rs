#![allow(dead_code)]
use std::time::Duration;

use criterion::{Bencher, Criterion};
use cubecl::frontend::PartialEqExpand;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use massively::Executor;
use massively::op::UnaryOp;

pub const SIZES: &[usize] = &[1024, 16 * 1024, 256 * 1024, 1024 * 1024];
pub const SORT_SIZES: &[usize] = &[1024, 16 * 1024, 256 * 1024];
pub const SAMPLE_COUNT: usize = 10;

pub struct U32Flag;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for U32Flag {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input != 0
    }
}

#[derive(Clone, Copy)]
pub enum Runtime {
    Gpu,
}

impl Runtime {
    pub fn available() -> Vec<Self> {
        vec![Self::Gpu]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
        }
    }

    pub fn exec(self) -> Executor<WgpuRuntime> {
        match self {
            Self::Gpu => Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice),
        }
    }
}

pub fn dense_f32(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 251) as f32).collect()
}

pub fn descending_f32(len: usize) -> Vec<f32> {
    (0..len).rev().map(|index| index as f32).collect()
}

pub fn shuffled_u32(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 1_103_515_245 + 12_345) % len.max(1)) as u32)
        .collect()
}

pub fn reverse_indices(len: usize) -> Vec<u32> {
    (0..len).rev().map(|index| index as u32).collect()
}

pub fn half_select_flags(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index % 2 == 0) as u32).collect()
}

pub fn select_flags(len: usize, selected_per_100: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index % 100) < selected_per_100) as u32)
        .collect()
}

pub fn run_keys(len: usize, run_len: usize) -> Vec<u32> {
    if run_len >= len {
        return vec![0; len];
    }
    (0..len)
        .map(|index| (index / run_len.max(1)) as u32)
        .collect()
}

pub fn ascending_u32(len: usize) -> Vec<u32> {
    (0..len).map(|index| index as u32).collect()
}

pub fn descending_u32(len: usize) -> Vec<u32> {
    (0..len).rev().map(|index| index as u32).collect()
}

pub fn even_u32(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index * 2) as u32).collect()
}

pub fn odd_u32(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index * 2 + 1) as u32).collect()
}

pub fn sync(exec: &Executor<WgpuRuntime>) {
    exec.sync().unwrap();
}

pub fn criterion() -> Criterion {
    Criterion::default()
        .sample_size(SAMPLE_COUNT)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250))
}

pub fn iter_gpu<F, O>(b: &mut Bencher<'_>, mut routine: F)
where
    F: FnMut() -> O,
{
    b.iter(|| routine());
}
