#![allow(dead_code)]

use massively::{Executor, Wgpu};

pub const SIZES: &[usize] = &[1024, 16 * 1024, 256 * 1024, 1024 * 1024];
pub const SORT_SIZES: &[usize] = &[1024, 16 * 1024, 256 * 1024];

#[derive(Clone, Copy)]
pub enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    pub fn available() -> Vec<Self> {
        let mut backends = vec![Self::Cpu];
        if Self::gpu_available() {
            backends.push(Self::Gpu);
        }
        backends
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }

    pub fn exec(self) -> Executor<Wgpu> {
        match self {
            Self::Cpu => Executor::<Wgpu>::cpu(),
            Self::Gpu => Executor::<Wgpu>::new(),
        }
    }

    fn gpu_available() -> bool {
        false
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

pub fn sync(exec: &Executor<Wgpu>) {
    exec.sync().unwrap();
}
