use super::CubePolicy;

/// CubeCL CUDA execution policy.
#[cfg(feature = "cuda")]
pub type CubeCuda = CubePolicy<cubecl::cuda::CudaRuntime>;

#[cfg(feature = "cuda")]
impl Default for CubeCuda {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
impl CubeCuda {
    /// Creates a policy backed by CubeCL's CUDA runtime for device 0.
    pub fn new() -> Self {
        Self::new_with_index(0)
    }

    /// Creates a policy backed by CubeCL's CUDA runtime for a specific device index.
    pub fn new_with_index(index: usize) -> Self {
        let device = cubecl::cuda::CudaDevice { index };
        CubePolicy::from_device(&device)
    }
}

/// CubeCL HIP execution policy.
#[cfg(feature = "hip")]
pub type CubeHip = CubePolicy<cubecl::hip::HipRuntime>;

#[cfg(feature = "hip")]
impl Default for CubeHip {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "hip")]
impl CubeHip {
    /// Creates a policy backed by CubeCL's HIP runtime for device 0.
    pub fn new() -> Self {
        Self::new_with_index(0)
    }

    /// Creates a policy backed by CubeCL's HIP runtime for a specific AMD device index.
    pub fn new_with_index(index: usize) -> Self {
        let device = cubecl::hip::AmdDevice { index };
        CubePolicy::from_device(&device)
    }
}

/// CubeCL WGPU execution policy.
#[cfg(feature = "wgpu")]
pub type CubeWgpu = CubePolicy<cubecl::wgpu::WgpuRuntime>;

#[cfg(feature = "wgpu")]
impl Default for CubeWgpu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "wgpu")]
impl CubeWgpu {
    /// Creates a policy backed by CubeCL's WGPU runtime using the default device.
    pub fn new() -> Self {
        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
        CubePolicy::from_device(&device)
    }

    /// Creates a policy backed by a WGPU discrete GPU selected by index.
    pub fn discrete_gpu(index: usize) -> Self {
        let device = cubecl::wgpu::WgpuDevice::DiscreteGpu(index);
        CubePolicy::from_device(&device)
    }

    /// Creates a policy backed by a WGPU integrated GPU selected by index.
    pub fn integrated_gpu(index: usize) -> Self {
        let device = cubecl::wgpu::WgpuDevice::IntegratedGpu(index);
        CubePolicy::from_device(&device)
    }

    /// Creates a policy backed by WGPU's CPU adapter.
    pub fn cpu() -> Self {
        let device = cubecl::wgpu::WgpuDevice::Cpu;
        CubePolicy::from_device(&device)
    }
}
