#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RangeMapping {
    Reverse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RangeControl {
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
    pub(crate) mapping: RangeMapping,
}

impl RangeControl {
    pub(crate) fn reverse(len: usize) -> Result<Self, crate::Error> {
        let len_u32 = u32::try_from(len).map_err(|_| crate::Error::LengthTooLarge { len })?;
        Ok(Self {
            len,
            len_u32,
            mapping: RangeMapping::Reverse,
        })
    }
}
