use crate::*;

extern "C" {
    pub fn do_fft_ffi(device_id: usize) -> u64;

    pub fn parallel_fft_api(device_id: usize, req: *mut u64, omega: *const u64, twiddle: *const u64, n: u64, log_n: u64, twiddle_size: u64) -> u64;

    pub fn parallel_ifft_api(device_id: usize, req: *mut u64, omega: *const u64, twiddle: *const u64, n: u64, log_n: u64, twiddle_size: u64, divisor: *const u64) -> u64;

    pub fn parallel_coeff_to_extended_part_api(device_id: usize, req: *mut u64, omega: *const u64, twiddle: *const u64, n: u64, log_n: u64, twiddle_size: u64, c: *const u64) -> u64;

    pub fn parallel_msm_api(device_id: usize, data: *const u64, bases: *const GpuAffine, lenght: u64, acc: *mut GpuProjective) -> u64;

}