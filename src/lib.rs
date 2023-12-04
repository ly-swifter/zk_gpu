pub mod api;
use std::os::raw::c_int;

pub use api::*;

pub mod config;
pub use config::*;

pub mod threadpool;
pub mod error;

pub mod gpulock;

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FFITraitObject {
    pub data: c_int,
    pub vtable: c_int,
    pub infinity: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAffine {
    pub x: [u64; 4],
    pub y: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProjective {
    pub x: [u64; 4],
    pub y: [u64; 4],
    pub z: [u64; 4],
}
