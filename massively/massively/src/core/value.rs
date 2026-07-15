//! Semantic and physical value traits.

use cubecl::prelude::{CubeElement, CubePrimitive};

/// A scalar that can occupy one physical storage column.
pub trait MStorageElement:
    CubePrimitive
    + CubeElement
    + crate::StorageLayout<StorageArity = crate::S1, StorageLeaves = crate::storage::Last<Self>>
    + Copy
    + Send
    + Sync
    + 'static
{
}

macro_rules! impl_storage_element {
    ($($ty:ty),+ $(,)?) => {
        $(impl MStorageElement for $ty {})+
    };
}

impl_storage_element!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
