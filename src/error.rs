use std::fmt;

/// Error returned by parallel algorithms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    /// The input and output slices do not have the same length.
    LengthMismatch { input: usize, output: usize },
    /// The requested output length cannot be represented by CubeCL launch dimensions.
    LengthTooLarge { len: usize },
    /// CubeCL rejected the kernel launch.
    Launch { message: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { input, output } => {
                write!(
                    f,
                    "input length ({input}) does not match output length ({output})"
                )
            }
            Self::LengthTooLarge { len } => {
                write!(f, "input length ({len}) is too large for a CubeCL launch")
            }
            Self::Launch { message } => write!(f, "CubeCL launch failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn ensure_same_len(actual: usize, expected: usize) -> Result<(), Error> {
    if actual != expected {
        return Err(Error::LengthMismatch {
            input: actual,
            output: expected,
        });
    }
    Ok(())
}
