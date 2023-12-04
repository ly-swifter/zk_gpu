use std::path::Path;

extern crate cc;

fn main() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    build();
}

fn build() {
    // println!("The feature is set to support gpu, rust build start to build cuda_ffi");
    let cuda_files = vec!["cuda/fft.cu"];
    let library_path = Path::new("/usr/local/cuda");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=static")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")
        .flag("-gencode")
        .flag("arch=compute_75,code=compute_75")
        .flag("-gencode")
        .flag("arch=compute_61,code=compute_61")
        .flag("-std=c++11")
        .files(cuda_files)
        .include(library_path)
        .compile("fft");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}

