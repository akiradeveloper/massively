use crate::error::Error;
use cubecl::prelude::*;

#[allow(dead_code)]
pub(crate) const MAX_1D_WORKGROUPS: u32 = 65_535;

pub(crate) struct Launch1D {
    pub(crate) logical_blocks: usize,
    pub(crate) logical_blocks_u32: u32,
    #[allow(dead_code)]
    pub(crate) launch_blocks: u32,
    x: u32,
    y: u32,
    z: u32,
}

impl Launch1D {
    pub(crate) fn cube_count(&self) -> CubeCount {
        CubeCount::Static(self.x, self.y, self.z)
    }
}

pub(crate) fn cube_count_1d(logical_blocks: u32) -> CubeCount {
    let (x, y, z) = split_1d_blocks(logical_blocks);
    CubeCount::Static(x, y, z)
}

pub(crate) fn block_count(len: usize, block_size: u32) -> Result<usize, Error> {
    let block_size = block_size as usize;
    debug_assert_ne!(block_size, 0);
    Ok(len.div_ceil(block_size))
}

pub(crate) fn launch_1d<R: Runtime>(
    client: &ComputeClient<R>,
    len: usize,
    block_size: u32,
) -> Result<Launch1D, Error> {
    let logical_blocks = block_count(len, block_size)?;
    launch_blocks_1d(client, logical_blocks)
}

pub(crate) fn launch_blocks_1d<R: Runtime>(
    _client: &ComputeClient<R>,
    logical_blocks: usize,
) -> Result<Launch1D, Error> {
    let logical_blocks_u32 = u32::try_from(logical_blocks).map_err(|_| Error::LengthTooLarge {
        len: logical_blocks,
    })?;
    let (x, y, z) = split_1d_blocks(logical_blocks_u32);
    let launch_blocks = x.saturating_mul(y).saturating_mul(z);
    Ok(Launch1D {
        logical_blocks,
        logical_blocks_u32,
        launch_blocks,
        x,
        y,
        z,
    })
}

fn split_1d_blocks(logical_blocks: u32) -> (u32, u32, u32) {
    if logical_blocks <= MAX_1D_WORKGROUPS {
        return (logical_blocks, 1, 1);
    }

    if let Some((x, y)) = split_exact_2d(logical_blocks) {
        return (x, y, 1);
    }

    if let Some((x, y, z)) = split_exact_3d(logical_blocks) {
        return (x, y, z);
    }

    let x = MAX_1D_WORKGROUPS;
    let y = logical_blocks.div_ceil(x);
    if y <= MAX_1D_WORKGROUPS {
        return (x, y, 1);
    }

    let z = y.div_ceil(MAX_1D_WORKGROUPS);
    (x, MAX_1D_WORKGROUPS, z)
}

fn split_exact_2d(logical_blocks: u32) -> Option<(u32, u32)> {
    let mut y = logical_blocks.div_ceil(MAX_1D_WORKGROUPS);
    while y <= MAX_1D_WORKGROUPS {
        if logical_blocks.is_multiple_of(y) {
            let x = logical_blocks / y;
            if x <= MAX_1D_WORKGROUPS {
                return Some((x, y));
            }
        }
        y += 1;
    }
    None
}

fn split_exact_3d(logical_blocks: u32) -> Option<(u32, u32, u32)> {
    let min_z = logical_blocks.div_ceil(MAX_1D_WORKGROUPS * MAX_1D_WORKGROUPS);
    let mut z = min_z.max(2);
    while z <= MAX_1D_WORKGROUPS {
        if logical_blocks.is_multiple_of(z) {
            let rest = logical_blocks / z;
            if let Some((x, y)) = split_exact_2d(rest) {
                return Some((x, y, z));
            }
        }
        z += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_1d_blocks_keeps_x_within_wgpu_limit() {
        assert_eq!(split_1d_blocks(65_535), (65_535, 1, 1));
        assert_eq!(split_1d_blocks(65_536), (32_768, 2, 1));
        assert_eq!(split_1d_blocks(78_125), (15_625, 5, 1));
        assert_eq!(split_1d_blocks(400_000), (50_000, 8, 1));
    }

    #[test]
    fn split_1d_blocks_covers_large_logical_block_counts() {
        assert_eq!(split_1d_blocks(65_535 * 65_535), (65_535, 65_535, 1));
        let (x, y, z) = split_1d_blocks(65_535 * 65_535 + 1);
        assert!(x <= MAX_1D_WORKGROUPS);
        assert!(y <= MAX_1D_WORKGROUPS);
        assert!(z <= MAX_1D_WORKGROUPS);
        assert!((x as u64) * (y as u64) * (z as u64) >= (65_535_u64 * 65_535_u64 + 1));

        let (x, y, z) = split_1d_blocks(u32::MAX);
        assert!(x <= MAX_1D_WORKGROUPS);
        assert!(y <= MAX_1D_WORKGROUPS);
        assert!(z <= MAX_1D_WORKGROUPS);
        assert!((x as u64) * (y as u64) * (z as u64) >= u32::MAX as u64);
    }
}
