use std::io;

#[derive(thiserror::Error, Debug)]
pub enum ZkGpuError {
    /// A simple error that is described by a string.
    #[error("ZkGpuError: {0}")]
    Simple(&'static str),
}