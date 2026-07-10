use cubecl::prelude::CubeCount;

const MAX_AXIS: u32 = 65_535;

pub(crate) fn cube_count_1d(blocks: usize) -> Result<CubeCount, crate::Error> {
    let blocks = u32::try_from(blocks).map_err(|_| crate::Error::LengthTooLarge { len: blocks })?;
    if blocks <= MAX_AXIS {
        return Ok(CubeCount::Static(blocks, 1, 1));
    }
    let mut y = blocks.div_ceil(MAX_AXIS);
    while y <= MAX_AXIS {
        if blocks.is_multiple_of(y) {
            return Ok(CubeCount::Static(blocks / y, y, 1));
        }
        y += 1;
    }
    let x = MAX_AXIS;
    let y = blocks.div_ceil(x);
    let z = y.div_ceil(MAX_AXIS);
    Ok(CubeCount::Static(x, y.min(MAX_AXIS), z))
}
