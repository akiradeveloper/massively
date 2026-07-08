use crate::error::Error;
use cubecl::prelude::*;
use cubecl::server::CubeCountSelection;

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
    client: &ComputeClient<R>,
    logical_blocks: usize,
) -> Result<Launch1D, Error> {
    let logical_blocks_u32 =
        u32::try_from(logical_blocks).map_err(|_| Error::LengthTooLarge { len: logical_blocks })?;
    let selection = CubeCountSelection::new(client, logical_blocks_u32);
    let cube_count = selection.cube_count();
    let (x, y, z) = cube_count_xyz(cube_count);
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

fn cube_count_xyz(cube_count: CubeCount) -> (u32, u32, u32) {
    match cube_count {
        CubeCount::Static(x, y, z) => (x, y, z),
        CubeCount::Dynamic(_) => unreachable!("CubeCountSelection::new returns a static count"),
    }
}
