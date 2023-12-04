// use std::time::Instant;

// use halo2curves::bn256::Fr;
// use halo2_proofs::*;
//use cuda_fft_ffi::*;
// use halo2_proofs::arithmetic::generate_twiddle_lookup_table;
// pub use ff::{Field, PrimeField};

fn main() 
{
    let n = 10; 
    for i in (0..n).rev() {
        println!("{}", i);
    }
    // unsafe {
    //     parallel_msm_api(0)
    // };
    // let data = FFITraitObject {
    //     data: 6,
    //     vtable: 6,
    //     infinity: 0,
    // };
    // unsafe {
    //     println!("request: {:?}", data);
    //     let demo = ArrayDemo {
    //         data: [1u64, 2u64, 3u64, 4u64]
    //     };
    //     let demo1 = ArrayDemo {
    //         data: [5u64, 6u64, 7u64, 8u64]
    //     };


    //     let mut demo_array = Vec::new();

    //     demo_array.push(demo.clone());
    //     demo_array.push(demo1);

    //     let code = print_struct(0, data, demo, demo_array.as_ptr());
    //     println!("resp code: {}", code);
    // }
    
}

// fn main() {
//     println!("Hello, world!");
//     parallel_fft();
//     // serial_fft();
// }

// fn parallel_fft() {
//     let mut data: Vec<u64> = Vec::new();
//     let base = 2u64;
//     let log_n = 8u64;
//     let n = base.pow(log_n as u32);
//     println!("n: {}", n);
//     let a = (0..(1 << log_n)).map(|i| Fr::from_raw(
//         [i, 9720557559516996230, 14346186778711300467, 2065298866258849470]
//     )).collect::<Vec<_>>();
//     println!("a.len: {}", a.len());
//     for (idx, val) in a.iter().enumerate() {
//         // println!("a idx: {}: {:?}", idx, val.0);
//         for xx in val.0 {
//             data.push(xx);
//         }
//     }
//     let get_omega = get_omega(log_n);
//     let omega = get_omega.0.to_vec();
//     println!("omega: {:?}", omega);
//     let start = Instant::now();
//     println!("start parallel_fft_api");
//     println!("data length: {}", data.len());
//     println!("start: {:?}, end: {:?} ", data[0], data[(n - 1) as usize]);

//     // let exp = Fr::from(2u64);
//     // println!("exp: {:?}", exp.0);
//     // let new_omega = get_omega.pow_vartime([2, 0, 0, 0]);
//     // println!("new_omega: {:?}", new_omega.0);
//     let twiddle = get_twiddle(get_omega, log_n);
//     unsafe{
//         parallel_fft_api(0, data.as_mut_ptr(), omega.as_ptr(), twiddle.as_ptr(), n, log_n, (twiddle.len() / 4) as u64);
//     }
//     println!("cost time: {:?}", start.elapsed());
//     for i in 0..n {
//         print!("idx: {}, ", i);
//         for j in 0..4 {
//             print!("{:?}, ", data[(i * 4 + j) as usize]);
//         }
//         println!();
//     }
//     println!();
// }

// // 8765445983881484906,
// // 3daec14d565241d9,
// // 4444702420166132185,

// fn serial_fft(){
//     let mut data: Vec<u64> = Vec::new();
//     let base = 2u64;
//     let log_n = 3u64;
//     let n = base.pow(log_n as u32);
//     for i in 0..n {
//         data.push(0x1 + i);
//         data.push(0x0b7af45b6073944b);
//         data.push(0xea5b8bd611a5bd4c);
//         data.push(0x150160330625db3d);
//     }

//     let omega = vec![0x3daec14d565241d9, 0x0b7af45b6073944b, 0xea5b8bd611a5bd4c, 0x150160330625db3d];
//     let start = Instant::now();
//     println!("start do_fft_ffi");
//     unsafe{
//         do_fft_ffi(0, data.as_mut_ptr(), omega.as_ptr(), 0, log_n);
//     }
//     println!("cost time: {:?}", start.elapsed());
//     for i in 0..n * 4 {
//         if i % 4 == 0 {
//             println!();
//         }
//         print!("{:?}, ", data[i as usize]);
//     }
//     println!();

// }



// fn get_twiddle(omega: Fr, k: u64) -> Vec<u64>{
//     let twiddle = generate_twiddle_lookup_table(omega, k as u32, 10, true);
//     let mut data = Vec::new();
//     for val in twiddle {
//         for v in val.0 {
//             data.push(v);
//         }
//     }
//     println!("get_twiddle: {}", data.len() / 4);
//     data
// }

// fn get_omega(k: u64) -> Fr {
//     let j = 3;
//     let quotient_poly_degree = (j - 1) as u64;
//     // n = 2^k
//     let n = 1u64 << k;
//     let mut extended_k = k;
//     while (1 << extended_k) < (n * quotient_poly_degree) {
//         extended_k += 1;
//     }

//     let mut extended_omega = Fr::root_of_unity();
//     for _ in extended_k..28 {
//         extended_omega = extended_omega.square();
//     }
//     let extended_omega = extended_omega;
//     let mut extended_omega_inv = extended_omega;
//     let mut omega = extended_omega;
//     for _ in k..extended_k {
//         omega = omega.square();
//     }
//     let omega = omega;
//     // let mut omega_inv = omega; // Inversion computed later
//     omega
// }