#include <stdio.h>
#include <stdint.h>

#include <iostream>
#include <cuda_runtime.h>
#include "gpu_math.h"
#include "bn256.h"
#include "curve.cuh"
#include <sys/time.h>

__device__ void serial_fft(BigNum *a, BigNum *omega, u_int64_t n, u_int64_t log_n, u_int64_t start_idx)
{
    for (u_int64_t i = 0; i < n; i++) {
        u_int64_t rk = bit_reverse(i, log_n);
        if (i < rk) {
            BigNum tmp = ZERO;
            copy_value(&a[i + start_idx], &tmp);
            copy_value(&a[rk + start_idx], &a[i + start_idx]);
            copy_value(&tmp, &a[rk + start_idx]);
        }
    }

    int m = 1;
    for (u_int64_t i = 0; i < log_n; i++) {
        BigNum exp = from_int(n / (2 * m));
        BigNum w_m = pow_vartime(omega, &exp);
        int k = 0;
        while (k < n) {
            BigNum w = FR_R;
            for (int j = 0; j < m; j++) {
                BigNum t = a[k + j + m  + start_idx];
                t = group_scale(&t, &w);
                a[k + j + m + start_idx] = a[k + j + start_idx];
                a[k + j + m + start_idx] = group_sub(&a[k + j + m + start_idx], &t);
                a[k + j + start_idx] = group_add(&a[k + j + start_idx], &t);
                w = group_scale(&w, &w_m);
            }
            k += 2 * m;
        }
        m *= 2;
    }
}

__global__ void do_fft_gpu()
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("idx: %d\n", idx);
    // serial_fft(a, omega, n, log_n, idx * n);
    BigNum d = {9191727205810674688, 8873516619432482690, 6234034562745979772, 1723971208416369580};
    BigNum b = {9191727205810674688, 8873516619432482690, 6234034562745979772, 1723971208416369580};
    BigNum c = group_scale(&d, &b);
    BigNum e = fq_mul(&d, &b);
    BigNum exp = {2, 0, 0, 0};
    pow_vartime(&d, &exp);
    // printf("do_fft_gpu c: %lu, %lu, %lu, %lu\n", c.data[0], c.data[1], c.data[2], c.data[3]);
    // printf("do_fft_gpu e: %lu, %lu, %lu, %lu\n", e.data[0], e.data[1], e.data[2], e.data[3]);
    // printf("do_fft_gpu f: %lu, %lu, %lu, %lu\n", f.data[0], f.data[1], f.data[2], f.data[3]);
    // printf("finish!!\n");
}

__device__ void split_radix_fft(BigNum *tmp, BigNum *a, BigNum *twiddle, const u_int64_t n, const u_int64_t sub_fft_offset, const u_int64_t log_split, const int split_m, const int idx) 
{
    const u_int64_t sub_n = n / split_m;
    for (u_int64_t i = 0; i < split_m; i++) 
    {   
        tmp[bit_reverse(i, log_split) + idx * split_m] = a[i * sub_n + sub_fft_offset];
    }
    int m = 1;
    int new_n = split_m;
    for (int i = 0; i < log_split; i++)
    {
        int omega_idx = sub_n * new_n / (2 * m);
        int low_idx = omega_idx % (1 << SPARSE_TWIDDLE_DEGREE);
        int high_idx = omega_idx >> SPARSE_TWIDDLE_DEGREE;
        BigNum w_m = twiddle[low_idx];
        if (high_idx > 0) 
        {
            w_m = group_scale(&w_m, &twiddle[(1 << SPARSE_TWIDDLE_DEGREE) + high_idx]);
        }
        int k = 0;
        while (k < split_m)
        {
            BigNum w = FR_R;
            for (int j = 0; j < m; j++) {
                BigNum t = tmp[k + j + m + idx * split_m];
                t = group_scale(&t, &w);
                tmp[k + j + m + idx * split_m] = tmp[k + j + idx * split_m];
                tmp[k + j + m + idx * split_m] = group_sub(&tmp[k + j + m + idx * split_m], &t);
                tmp[k + j + idx * split_m] = group_add(&tmp[k + j + idx * split_m], &t);
                w = group_scale(&w, &w_m);
            }
            k += 2 * m;
        }
        m *= 2;
    }

    int omega_idx = sub_fft_offset;
    int low_idx = omega_idx % (1 << SPARSE_TWIDDLE_DEGREE);
    int high_idx = omega_idx >> SPARSE_TWIDDLE_DEGREE;

    BigNum omega = twiddle[low_idx];

    if (high_idx > 0) 
    {
        omega = group_scale(&omega, &twiddle[(1 << SPARSE_TWIDDLE_DEGREE) + high_idx]);
    }
    BigNum w_m = FR_R;
    for (int i = 0; i < split_m; i++) {
        BigNum c = group_scale(&tmp[i + idx * split_m], &w_m);
        copy_value(&c, &tmp[idx * split_m + i]);
        w_m = group_scale(&w_m, &omega);
    }
}

__global__ void sub_fft(BigNum *a, BigNum *omega, u_int64_t n, u_int64_t log_split, const int split_m)
{
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    BigNum exp = from_int(split_m);
    BigNum new_omega = pow_vartime(omega, &exp);
    serial_fft(a, &new_omega, n, log_split, idx * n);
}

__global__ void parallel_fft(BigNum *a, BigNum *omega, BigNum *twiddle, u_int64_t n, u_int64_t log_n, BigNum *tmp, int split_m, int log_split)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    split_radix_fft(tmp, a, twiddle, n, idx, log_split, split_m, idx);
}

__global__ void shuffle(BigNum *a, BigNum *tmp, int sub_n, int split_m)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / sub_n;
    int j = idx % sub_n;
    a[idx] = tmp[j * split_m + i];
}

__global__ void print_tmp(u_int64_t round, int n, int index, BigNum *tmp)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tmp round: %lu, n: %d, %d: %lu, %lu, %lu, %lu\n", round, n, index, tmp[idx].data[0], tmp[idx].data[1], tmp[idx].data[2], tmp[idx].data[3]);
}

__global__ void copy_req_data(BigNum *a, BigNum *tmp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tmp[idx] = a[idx];
}

__global__ void unshuffle(BigNum *a, BigNum *tmp, int sub_n, int mask, int log_split, int req_type, BigNum *divisor)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = tmp[sub_n * (idx & mask) + (idx >> log_split)];
    if (req_type == 1){
        a[idx] = group_scale(&a[idx], divisor);
    }
}

__global__ void do_msm_gpu(BigNum *data, AffinePoint *bases, u_int64_t length, ProjectivePoint *resp)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = idx * length;

    resp[idx] = projective_zero_point();
    
    int c = 0;
    if (length < 4) {
        c = 1;
    } else if (length < 32) {
        c = 3;
    } else {
        c = ceilf(__logf(length));
    }   
    int segments = (256 / c) + 1;
    for (int current_segment = segments - 1; current_segment >= 0; current_segment--) {
        for (int j = 0; j < c; j++) {
            resp[idx] = fq_double(resp[idx]);
        }
        BucketPoint buckets[127] = {bucket_zero_point()};
        for(int m = 0; m < length; m++) {
            int coeff = get_at(current_segment, c, &data[m + start_idx]);
            if (coeff != 0) {
                ProjectivePoint base_point = fromAffinePoint(bases[m + start_idx]);
                if (projective_compare(buckets[coeff - 1].p, projective_zero_point())) {
                    buckets[coeff - 1].p = base_point;
                }else {
                    if (buckets[coeff - 1].num == 0) {
                        buckets[coeff - 1].p = affine_add(buckets[coeff - 1].p, base_point);
                    }else {
                        buckets[coeff - 1].p = projective_add(buckets[coeff - 1].p, base_point);
                    }
                    buckets[coeff - 1].num = 1;
                }
            }
        }

        int buckets_length = (1 << c) - 1;
        ProjectivePoint running_sum = projective_zero_point();
        for (int n = buckets_length - 1; n >= 0; n--) {
            if (projective_compare(buckets[n].p, projective_zero_point())) {
            }else {
                if (buckets[n].num == 0) {
                    running_sum = projective_add_affine(running_sum, buckets[n].p);
                }else {
                    running_sum = projective_add(buckets[n].p, running_sum);
                }
            }
            resp[idx] = projective_add(resp[idx], running_sum);
        }
    }
}

    __global__ void do_msm_collect(ProjectivePoint *req, ProjectivePoint *resp, int num, int start, ProjectivePoint *acc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    resp[num + idx] = projective_add(req[start + idx * 2], req[1 + start + idx * 2]);
    if (num == 1) {
        acc->x = resp[1].x;
        acc->y = resp[1].y;
        acc->z = resp[1].z;
    }
}

__global__ void distribute_powers(BigNum *a, int split_m, BigNum *c)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < split_m; i++){
        int index = idx * split_m + i;
        BigNum t = from_int(index);
        BigNum c_power = pow_vartime(c, &t);
        a[index] = group_scale(&a[index], &c_power);
    }
}

int do_fft(int device_id) 
{
    cudaSetDevice(device_id);
    cudaError_t err = cudaSuccess;

    printf("before do_fft_gpu");
    do_fft_gpu<<<512, 512>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "compute function failed (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }else{
        // printf("compute success!!\n");
    } 

    // printf("Copy output data from the CUDA device to the host memory\n");
    // err = cudaMemcpy(a, req_a, size, cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr,
    //             "Failed to copy res from device to host (error code %s)!\n",
    //             cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }else {
    //     // printf("copy output data from CUDA device success!!\n");
    // }

    // printf("resData: \n");
    // // for (int x = 0; x < n; ++x) {
    // //     printf("a[%d]: %lu, %lu, %lu, %lu\n", x, a[x].data[0], a[x].data[1], a[x].data[2], a[x].data[3]);
    // // }

    // cudaFree(req_a);
    // cudaFree(req_omega);
    return 0;
}

int do_parallel_fft(int device_id, BigNum *a, BigNum *omega, BigNum *twiddle, u_int64_t n, u_int64_t log_n, int twiddle_len, int req_type, BigNum *divisor, BigNum *c) 
{
    struct timeval start, end;
    gettimeofday( &start, NULL );
    cudaSetDevice(device_id);
    cudaError_t err = cudaSuccess;

    BigNum *req_a;
    u_int64_t size = sizeof(BigNum) * n;
    cudaMalloc((void**)&req_a, size);
    cudaMemcpy(req_a, a, size, cudaMemcpyHostToDevice);
    
    BigNum *req_c;
    u_int64_t c_size = sizeof(BigNum);
    cudaMalloc((void**)&req_c, c_size);
    cudaMemcpy(req_c, c, c_size, cudaMemcpyHostToDevice);
    const int base_size = 512;
    if (req_type == 2) {
        BigNum *req_c;
        u_int64_t c_size = sizeof(BigNum);
        cudaMalloc((void**)&req_c, c_size);
        cudaMemcpy(req_c, c, c_size, cudaMemcpyHostToDevice);
        
        int block_size = base_size;
        int grid_size = 1;
        int split_m = 64;
        int split_count = n / split_m;

        if (split_count <= block_size) {
            block_size = split_count;
        } else {
            grid_size = split_count / base_size;
        }
        distribute_powers<<<grid_size, block_size>>>(req_a, split_m, req_c);
        cudaDeviceSynchronize();
        cudaFree(req_c);
    }
    
    BigNum *req_tmp;
    cudaMalloc((void**)&req_tmp, size);

    BigNum *req_omega;
    u_int64_t omega_size = sizeof(BigNum);
    cudaMalloc((void**)&req_omega, omega_size);
    cudaMemcpy(req_omega, omega, omega_size, cudaMemcpyHostToDevice);

    BigNum *req_divisor;
    cudaMalloc((void**)&req_divisor, omega_size);
    cudaMemcpy(req_divisor, divisor, omega_size, cudaMemcpyHostToDevice);

    BigNum *req_twiddle;
    int twiddle_size = sizeof(BigNum) * twiddle_len;
    cudaMalloc((void**)&req_twiddle, twiddle_size);
    cudaMemcpy(req_twiddle, twiddle, twiddle_size, cudaMemcpyHostToDevice);

    int block_size = base_size;
    int grid_size = 1;
    int split_m = TMP_LEN;
    int log_split = TMP_LEN_DEGREE;
    int split_count = n / split_m;
    if (split_count <= block_size) {
        block_size = split_count;
    } else {
        grid_size = split_count / base_size;
    }
    parallel_fft<<<grid_size, block_size>>>(req_a, req_omega, req_twiddle, n, log_n, req_tmp, split_m, log_split);
    cudaDeviceSynchronize();
    cudaFree(req_twiddle);
    cudaFree(req_c);
    block_size = base_size;
    grid_size = 1;
    if (n <= block_size) {
        block_size = n;
    } else {
        grid_size = n / base_size;
    }
    shuffle<<<grid_size, block_size>>>(req_a, req_tmp, split_count, split_m);
    cudaDeviceSynchronize();
    
    block_size = base_size;
    grid_size = 1;
    if (split_m <= block_size) {
        block_size = split_m;
    } else {
        grid_size = split_m / base_size;
    }
    sub_fft<<<grid_size, block_size>>>(req_a, req_omega, split_count, log_n - log_split, split_m);
    cudaDeviceSynchronize();
    cudaFree(req_omega);

    int mask = ( 1 << log_split) - 1;
    block_size = base_size;
    grid_size = 1;
    if (n <= block_size) {
        block_size = n;
    }else {
        grid_size = n / base_size;
    }
    copy_req_data<<<grid_size, block_size>>>(req_a, req_tmp);
    cudaDeviceSynchronize();
    
    unshuffle<<<grid_size, block_size>>>(req_a, req_tmp, split_count, mask, log_split, req_type, req_divisor);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "compute function failed (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

    err = cudaMemcpy(a, req_a, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy res from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);

    cudaFree(req_a);
    cudaFree(req_tmp);
    cudaFree(req_divisor);
    return 0;
}

// int do_parallel_coefficient(int device_id, BigNum *a, u_int64_t n, BigNum *c) 
// {
//     struct timeval start, end;
//     gettimeofday( &start, NULL );
//     cudaSetDevice(device_id);
//     cudaError_t err = cudaSuccess;

//     BigNum *req_a;
//     u_int64_t size = sizeof(BigNum) * n;
//     cudaMalloc((void**)&req_a, size);
//     cudaMemcpy(req_a, a, size, cudaMemcpyHostToDevice);

    

//     err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         fprintf(stderr,
//                 "compute function failed (error code %s)!\n",
//                 cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     } 

//     err = cudaMemcpy(a, req_a, size, cudaMemcpyDeviceToHost);
//     if (err != cudaSuccess)
//     {
//         fprintf(stderr,
//                 "Failed to copy res from device to host (error code %s)!\n",
//                 cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     gettimeofday( &end, NULL );
//     int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
//     printf("total time is %d ms\n", timeuse/1000);
//     cudaFree(req_a);
//     cudaFree(req_c);
//     return 0;
// }

// [2023-11-22T02:51:46Z INFO  msm] commit_lagrange gpu takes 1031.868653704s   k=24  base_split=128
// [2023-11-22T03:23:24Z INFO  msm] commit_lagrange gpu takes 951.633471051s    k=24  base_split=256

int do_msm(int device_id, BigNum *data, AffinePoint *bases, u_int64_t length, ProjectivePoint *acc, u_int64_t round) 
{
    cudaSetDevice(device_id);
    cudaError_t err = cudaSuccess;

    BigNum *req_a;
    u_int64_t data_size = sizeof(BigNum) * length;
    cudaMalloc((void**)&req_a, data_size);
    cudaMemcpy(req_a, data, data_size, cudaMemcpyHostToDevice);
    
    const int base_size = 256;
    int base_split = 256;
    if (length <= 1048576) {
        base_split = 64;
    }
    int block_size = base_size;
    int grid_size = 1;
    int split_size = 1;
    int split_length = 1;
    if (length <= block_size) {
        block_size = length;
    } else {
        split_length = base_split;
        split_size = length / base_split;
        if (split_size >= 1024) {
            grid_size = length / (block_size * base_split);
        } else {
            block_size = split_size;
        }
    }

    u_int64_t base_projective_size = sizeof(ProjectivePoint);
    ProjectivePoint *resp;
    u_int64_t pp_size = base_projective_size * split_size;
    cudaMalloc((void**)&resp, pp_size);
    
    AffinePoint *req_bases;
    u_int64_t bases_size = sizeof(AffinePoint) * length;
    cudaMalloc((void**)&req_bases, bases_size);
    cudaMemcpy(req_bases, bases, bases_size, cudaMemcpyHostToDevice);

    // print_tmp<<<1, 1>>>(round, 0, split_size / 2, &req_a[split_size / 2]);
    // print_tmp<<<1, 1>>>(round, 1, split_size / 2, &req_bases[split_size / 2].x);
    // print_tmp<<<1, 1>>>(round, 2, split_size / 2, &req_bases[split_size / 2].y);

    printf("before do_msm_gpu, grid_size: %d, block_size: %d, split_length: %d, split_size: %d \n", grid_size, block_size, split_length, split_size);
    do_msm_gpu<<<grid_size, block_size>>>(req_a, req_bases, split_length, resp);
    cudaDeviceSynchronize();
    cudaFree(req_a);
    cudaFree(req_bases);

    ProjectivePoint *req_acc;
    cudaMalloc((void**)&req_acc, base_projective_size);

    ProjectivePoint *value;
    u_int64_t value_size = base_projective_size * split_size;
    cudaMalloc((void**)&value, value_size);

    // print_tmp<<<1, 1>>>(round, 3, split_size / 2, &resp[split_size / 2].x);
    // print_tmp<<<1, 1>>>(round, 4, split_size / 2, &resp[split_size / 2].y);
    // print_tmp<<<1, 1>>>(round, 5, split_size / 2, &resp[split_size / 2].z);

    int height = ceil(log2f(split_size));
    int nodes = split_size;
    for (int i = 0; i < height; i++) {
        int num = nodes / 2;
        int grid_size = (num + base_size - 1) / base_size;
        if (num <= base_size) {
            grid_size = 1;
            block_size = num;
        }
        if (i == 0) {
            do_msm_collect<<<grid_size, block_size>>>(resp, value, num, 0, req_acc);
            cudaFree(resp);
        }else {
            do_msm_collect<<<grid_size, block_size>>>(value, value, num, num * 2, req_acc);
        }
        nodes /= 2;
        cudaDeviceSynchronize();
    }
    cudaFree(value);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "compute function failed (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }else{
    } 

    err = cudaMemcpy(acc, req_acc, base_projective_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy res from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(req_acc);
    return 0;
}

extern "C" {
    int do_fft_ffi(int device_id) 
    {
        printf("before do_fft \n");
        return do_fft(device_id);
    }

    int parallel_fft_api(int device_id, BigNum *data, BigNum *omega, BigNum *twiddle, u_int64_t n, int log_n, int twiddle_len) 
    {
        int code = do_parallel_fft(device_id, data, omega, twiddle, n, log_n, twiddle_len, 0, &ZERO, &ZERO);
        return code;
    }

    int parallel_ifft_api(int device_id, BigNum *data, BigNum *omega, BigNum *twiddle, u_int64_t n, int log_n, int twiddle_len, BigNum *divisor) 
    {
        int code = do_parallel_fft(device_id, data, omega, twiddle, n, log_n, twiddle_len, 1, divisor, &ZERO);
        return code;
    }

    int parallel_coeff_to_extended_part_api(int device_id, BigNum *data, BigNum *omega, BigNum *twiddle, u_int64_t n, int log_n, int twiddle_len, BigNum *c) 
    {
        int code = do_parallel_fft(device_id, data, omega, twiddle, n, log_n, twiddle_len, 2, &ZERO, c);
        return code;
    }

    int parallel_msm_api(int device_id, BigNum *data, AffinePoint *point, u_int64_t length, ProjectivePoint *acc, u_int64_t round)
    {  
        int code = do_msm(device_id, data, point, length, acc, round);
        return code;
    }

}