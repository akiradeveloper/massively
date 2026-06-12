#![allow(dead_code)]

use cubecl::device::Device;
use massively::CubeWgpu;

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

    pub fn policy(self) -> CubeWgpu {
        match self {
            Self::Cpu => CubeWgpu::cpu(),
            Self::Gpu => {
                if cubecl::wgpu::WgpuDevice::device_count(0) > 0 {
                    CubeWgpu::discrete_gpu(0)
                } else if cubecl::wgpu::WgpuDevice::device_count(1) > 0 {
                    CubeWgpu::integrated_gpu(0)
                } else {
                    panic!("No WGPU GPU adapter found; only CPU adapters are available")
                }
            }
        }
    }

    fn gpu_available() -> bool {
        cubecl::wgpu::WgpuDevice::device_count(0) > 0
            || cubecl::wgpu::WgpuDevice::device_count(1) > 0
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

pub fn sync(policy: &CubeWgpu) {
    futures_lite::future::block_on(policy.client().sync()).unwrap();
}
