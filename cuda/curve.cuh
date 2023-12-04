#ifndef _CURVE_H
#define _CURVE_H

#include <stdio.h>
#include <stdint.h>

#include <iostream>
#include <cuda_runtime.h>
#include "gpu_math.h"
#include "bn256.h"


__device__ ProjectivePoint identity() {
    return ProjectivePoint {
        x: ZERO,
        y: ZERO,
        z: FQ_R,
        // infinity: false
    };
}

__device__ ProjectivePoint projective_zero_point() {
    return ProjectivePoint {
        x: ZERO,
        y: ZERO,
        z: ZERO,
        // infinity: false
    };
}

__device__ AffinePoint affine_zero_point() {
    return AffinePoint {
        x: ZERO,
        y: ZERO,
        // infinity: false
    };
}

__device__ BucketPoint bucket_zero_point() {
    return BucketPoint {
        num: 0,
        p: projective_zero_point(),
    };
};

__device__ ProjectivePoint fromAffinePoint(AffinePoint point) {
    return ProjectivePoint {
        x: point.x,
        y: point.y,
        z: FQ_R,
        // infinity: false
    };
}

__device__ bool projective_compare(ProjectivePoint a, ProjectivePoint b) {
    if (is_equal(a.x, b.x) && is_equal(a.y, b.y) && is_equal(a.z, b.z))
    {
        return true;
    }else {
        return false;
    }
}

__device__ bool affine_compare(AffinePoint a, AffinePoint b) {
    if (compare(&a.x, &b.x) && compare(&a.y, &b.y))
    {
        return true;
    }else {
        return false;
    }
}

__device__ bool x_y_is_zero(ProjectivePoint a) 
{
    if (is_equal(a.x, ZERO) && is_equal(a.y, ZERO)) {
        return true;
    }else {
        return false;
    }
}

__device__ bool z_is_zero(ProjectivePoint a) 
{
    if (is_equal(a.z, ZERO)) {
        return true;
    }else {
        return false;
    }
}

__device__ BigNum fq_pow(BigNum value, BigNum pow) 
{
    BigNum res = FQ_R;
    for (int i = 3; i >=0; i--)
    {
        for (int num = 63; num >= 0; num--)
        {
            res = fq_square(&res);
            if (((pow.data[i] >> num) & 0x1)) {
                BigNum tmp = fq_mul(&res, &value);
                copy_value(&tmp, &res);
            }
        }
    }
    return res;
}

__device__ int get_at(int segment, int c, BigNum *coeff)
{   
    u_int8_t coeff_bytes[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    BigNum tmp_c = fr_mul(coeff, &ONE);
    bignum_to_bytes(tmp_c, coeff_bytes);
    int skip_bits = segment * c;
    int skip_bytes = skip_bits / 8;
    if (skip_bytes >= 32) {
        return 0;
    }
    
    u_int8_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 8; i++) {
        if (skip_bytes + i >= 32) {
        }else {
            v[i] = coeff_bytes[skip_bytes + i];
        }
    }
    u_int64_t tmp = from_le_bytes(v);
    tmp >>= skip_bits - (skip_bytes * 8);
    tmp = tmp % (1 << c);
    return tmp;
}

__device__ AffinePoint to_affine_point(ProjectivePoint point)
{
    BigNum zinv = fq_pow(point.z, FQ_INVERT_POW);
    BigNum zinv2 = fq_square(&zinv);
    BigNum x = fq_mul(&point.x, &zinv2);
    BigNum zinv3 = fq_mul(&zinv2, &zinv);
    BigNum y = fq_mul(&point.y, &zinv3);
    return AffinePoint{x, y};
}

__device__ ProjectivePoint fq_double(ProjectivePoint point)
{
    BigNum a = fq_square(&point.x);
    BigNum b = fq_square(&point.y);
    BigNum c = fq_square(&b);
    BigNum d1 = fq_add(&point.x, &b);
    BigNum d2 = fq_square(&d1);
    BigNum d3 = fq_sub(&d2, &a);
    BigNum d4 = fq_sub(&d3, &c);
    BigNum d = fq_add(&d4, &d4);
    BigNum aa = fq_add(&a, &a);
    BigNum e = fq_add(&aa, &a);
    BigNum f = fq_square(&e);
    BigNum z = fq_mul(&point.z, &point.y);
    BigNum z3 = fq_add(&z, &z);
    BigNum dd = fq_add(&d, &d);
    BigNum x3 = fq_sub(&f, &dd);
    BigNum c1 = fq_add(&c, &c);
    BigNum c2 = fq_add(&c1, &c1);
    BigNum c3 = fq_add(&c2, &c2);
    BigNum y1 = fq_sub(&d, &x3);
    BigNum y2 = fq_mul(&e, &y1);
    BigNum y3 = fq_sub(&y2, &c3);
    return ProjectivePoint{x3, y3, z3};   
}

__device__ ProjectivePoint affine_add(ProjectivePoint a, ProjectivePoint b)
{
    if (z_is_zero(a)) {
        return b;
    }else if (x_y_is_zero(b)) {
        return a;
    }else {
        if (is_equal(a.x, b.x)) {
            if (is_equal(a.y, b.y)){
                return fq_double(a);  //double
            }else {
                return identity();
            }
        }else {
            BigNum h = fq_sub(&b.x, &a.x);
            BigNum hh = fq_square(&h);
            BigNum i = fq_add(&hh, &hh);
            BigNum ii = fq_add(&i, &i);
            BigNum j = fq_mul(&h, &ii);
            BigNum r = fq_sub(&b.y, &a.y);
            BigNum rr = fq_add(&r, &r);
            BigNum v = fq_mul(&a.x, &ii);
            BigNum rr_sq = fq_square(&rr);
            BigNum r_square_sub_j = fq_sub(&rr_sq, &j);
            BigNum r_square_sub_j_sv = fq_sub(&r_square_sub_j, &v);
            BigNum x3 = fq_sub(&r_square_sub_j_sv, &v);
            BigNum j2 = fq_mul(&a.y, &j);
            BigNum j3 = fq_add(&j2, &j2);
            BigNum y1 = fq_sub(&v, &x3);
            BigNum y2 = fq_mul(&rr, &y1);
            BigNum y3 = fq_sub(&y2, &j3);
            BigNum z3 = fq_add(&h, &h);
            return ProjectivePoint { x3, y3, z3};
        }
    }
    
}

__device__ ProjectivePoint projective_add(ProjectivePoint a, ProjectivePoint b)
{
    if (z_is_zero(a)) {
        return b;
    }else if (z_is_zero(b)) {
        return a;
    }else {
        BigNum z1z1 = fq_square(&a.z);
        BigNum z2z2 = fq_square(&b.z);
        BigNum u1 = fq_mul(&a.x, &z2z2);
        BigNum u2 = fq_mul(&b.x, &z1z1);
        BigNum tmp1 = fq_mul(&a.y, &z2z2);
        BigNum s1 = fq_mul(&tmp1, &b.z);
        BigNum tmp2 = fq_mul(&b.y, &z1z1);
        BigNum s2 = fq_mul(&tmp2, &a.z);
        if (is_equal(u1, u2)) {
            if (is_equal(s1, s2)) {
                return fq_double(a);
            }else {
                return identity();
            }
        }else {
            BigNum h = fq_sub(&u2, &u1);
            BigNum h2 = fq_add(&h, &h);
            BigNum i = fq_square(&h2);
            BigNum j = fq_mul(&h, &i);
            BigNum r = fq_sub(&s2, &s1);
            BigNum r2 = fq_add(&r, &r);
            BigNum v = fq_mul(&u1, &i);
            BigNum r_s = fq_square(&r2);
            BigNum r_s_j = fq_sub(&r_s, &j);
            BigNum r_s_j_v = fq_sub(&r_s_j, &v);
            BigNum x3 = fq_sub(&r_s_j_v, &v);
            BigNum tmp3 = fq_mul(&s1, &j);
            BigNum tmp4 = fq_add(&tmp3, &tmp3);
            BigNum tmp5 = fq_sub(&v, &x3);
            BigNum tmp6 = fq_mul(&tmp5, &r2);
            BigNum y3 = fq_sub(&tmp6, &tmp4);
            BigNum zz = fq_add(&a.z, &b.z);
            BigNum zz_s = fq_square(&zz);
            BigNum tmp7 = fq_sub(&zz_s, &z1z1);
            BigNum tmp8 = fq_sub(&tmp7, &z2z2);
            BigNum z3 = fq_mul(&tmp8, &h);
            return ProjectivePoint { x3, y3, z3 };
        }
    }
    
}

__device__ ProjectivePoint projective_add_affine(ProjectivePoint a, ProjectivePoint b)
{
    if (z_is_zero(a)) {
        return b;
    }else if (z_is_zero(b)) {
        return a;
    }else {
        BigNum z1z1 = fq_square(&a.z);
        BigNum u2 = fq_mul(&b.x, &z1z1);
        BigNum s1 = fq_mul(&b.y, &z1z1);
        BigNum s2 = fq_mul(&s1, &a.z);
        if (is_equal(a.x, u2)) {
            if (is_equal(a.y, s2)) {
                return fq_double(a);
            }else {
                return identity();
            }
        }else {
            BigNum h = fq_sub(&u2, &a.x);
            BigNum hh = fq_square(&h);
            BigNum hh_2 = fq_add(&hh, &hh);
            BigNum i = fq_add(&hh_2, &hh_2);
            BigNum j = fq_mul(&h, &i);
            BigNum r = fq_sub(&s2, &a.y);
            BigNum rr = fq_add(&r, &r);
            BigNum v = fq_mul(&a.x, &i);
            BigNum rr_sq = fq_square(&rr);
            BigNum tmp1 = fq_sub(&rr_sq, &j);
            BigNum tmp2 = fq_sub(&tmp1, &v);
            BigNum x3 = fq_sub(&tmp2, &v);
            BigNum tmp4 = fq_mul(&a.y, &j);
            BigNum tmp5 = fq_add(&tmp4, &tmp4);
            BigNum tmp6 = fq_sub(&v, &x3);
            BigNum tmp7 = fq_mul(&rr, &tmp6);
            BigNum y3 = fq_sub(&tmp7, &tmp5);
            BigNum tmp8 = fq_add(&a.z, &h);
            BigNum tmp9 = fq_square(&tmp8);
            BigNum tmp10 = fq_sub(&tmp9, &z1z1);
            BigNum z3 = fq_sub(&tmp10, &hh);
            return ProjectivePoint { x3, y3, z3 };
        }
    }
}

#endif