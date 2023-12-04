/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------

#ifndef _GPUMATH_H
#define _GPUMATH_H

#include <limits>
#include "bn256.h"

#define NBBLOCK 5

// Assembly directives
#define UADDO(c, a, b) asm volatile("add.cc.u64 %0, %1, %2;" \
                                    : "=l"(c)                \
                                    : "l"(a), "l"(b)         \
                                    : "memory");
#define UADDC(c, a, b) asm volatile("addc.cc.u64 %0, %1, %2;" \
                                    : "=l"(c)                 \
                                    : "l"(a), "l"(b)          \
                                    : "memory");
#define UADD(c, a, b) asm volatile("addc.u64 %0, %1, %2;" \
                                   : "=l"(c)              \
                                   : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile("add.cc.u64 %0, %0, %1;" \
                                  : "+l"(c)                \
                                  : "l"(a)                 \
                                  : "memory");
#define UADDC1(c, a) asm volatile("addc.cc.u64 %0, %0, %1;" \
                                  : "+l"(c)                 \
                                  : "l"(a)                  \
                                  : "memory");
#define UADD1(c, a) asm volatile("addc.u64 %0, %0, %1;" \
                                 : "+l"(c)              \
                                 : "l"(a));

#define USUBO(c, a, b) asm volatile("sub.cc.u64 %0, %1, %2;" \
                                    : "=l"(c)                \
                                    : "l"(a), "l"(b)         \
                                    : "memory");
#define USUBC(c, a, b) asm volatile("subc.cc.u64 %0, %1, %2;" \
                                    : "=l"(c)                 \
                                    : "l"(a), "l"(b)          \
                                    : "memory");
#define USUB(c, a, b) asm volatile("subc.u64 %0, %1, %2;" \
                                   : "=l"(c)              \
                                   : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile("sub.cc.u64 %0, %0, %1;" \
                                  : "+l"(c)                \
                                  : "l"(a)                 \
                                  : "memory");
#define USUBC1(c, a) asm volatile("subc.cc.u64 %0, %0, %1;" \
                                  : "+l"(c)                 \
                                  : "l"(a)                  \
                                  : "memory");
#define USUB1(c, a) asm volatile("subc.u64 %0, %0, %1;" \
                                 : "+l"(c)              \
                                 : "l"(a));

#define UMULLO(lo, a, b) asm volatile("mul.lo.u64 %0, %1, %2;" \
                                      : "=l"(lo)               \
                                      : "l"(a), "l"(b));
#define UMULHI(hi, a, b) asm volatile("mul.hi.u64 %0, %1, %2;" \
                                      : "=l"(hi)               \
                                      : "l"(a), "l"(b));
#define MADDO(r, a, b, c) asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;" \
                                       : "=l"(r)                       \
                                       : "l"(a), "l"(b), "l"(c)        \
                                       : "memory");
#define MADDC(r, a, b, c) asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;" \
                                       : "=l"(r)                        \
                                       : "l"(a), "l"(b), "l"(c)         \
                                       : "memory");
#define MADD(r, a, b, c) asm volatile("madc.hi.u64 %0, %1, %2, %3;" \
                                      : "=l"(r)                     \
                                      : "l"(a), "l"(b), "l"(c));
#define MADDS(r, a, b, c) asm volatile("madc.hi.s64 %0, %1, %2, %3;" \
                                       : "=l"(r)                     \
                                       : "l"(a), "l"(b), "l"(c));

#define Add2(r, a, b)            \
{                            \
    UADDO(r[0], a[0], b[0]); \
    UADDC(r[1], a[1], b[1]); \
    UADDC(r[2], a[2], b[2]); \
    UADD(r[3], a[3], b[3]); \
}

#define fq_add3(r, a)            \
{                            \
    UADDO(r[0], a[0], MODULUS.data[0]); \
    UADDC(r[1], a[1], MODULUS.data[1]); \
    UADDC(r[2], a[2], MODULUS.data[2]); \
    UADDC(r[3], a[3], MODULUS.data[3]); \
    UADD(r[4], 0ULL, 0ULL);     \
}

#define fr_add3(r, a)            \
{                            \
    UADDO(r[0], a[0], FR_MODULUS.data[0]); \
    UADDC(r[1], a[1], FR_MODULUS.data[1]); \
    UADDC(r[2], a[2], FR_MODULUS.data[2]); \
    UADDC(r[3], a[3], FR_MODULUS.data[3]); \
    UADD(r[4], 0ULL, 0ULL);     \
}


#define sub3(r, a, b)            \
{                            \
    USUBO(r[0], a[0], b[0]); \
    USUBC(r[1], a[1], b[1]); \
    USUBC(r[2], a[2], b[2]); \
    USUBC(r[3], a[3], b[3]); \
}

__device__ bool is_equal(BigNum x, BigNum y)
{
    if (x.data[0] == y.data[0] && x.data[1] == y.data[1] && x.data[2] == y.data[2] && x.data[3] == y.data[3]) {
        return true;
    }else {
        return false;
    }
}

__device__ int compare(const BigNum* x, const BigNum* y)
{
    if (x->data[3] > y->data[3]) {
        return 1;
    }else if (x->data[3] < y->data[3]) {
        return -1;
    }else {
        if (x->data[2] > y->data[2]) {
            return 1;
        }else if (x->data[2] < y->data[2]) {
            return -1;
        }else {
            if (x->data[1] > y->data[1]) {
                return 1;
            }else if(x->data[1] < y->data[1]) {
                return -1;
            }else {
                if (x->data[0] > y->data[0]) {
                    return 1;
                }else if (x->data[0] < y->data[0]) {
                    return -1;
                }else {
                    return 0;
                }
            }
        }
    }
}

__device__ int compare_modulus(const BigNum* x, BigNum modulus)
{
    if (x->data[3] > modulus.data[3]) {
        return 1;
    }else if (x->data[3] < modulus.data[3]) {
        return -1;
    }else {
        if (x->data[2] > modulus.data[2]) {
            return 1;
        }else if (x->data[2] < modulus.data[2]) {
            return -1;
        }else {
            if (x->data[1] > modulus.data[1]) {
                return 1;
            }else if(x->data[1] < modulus.data[1]) {
                return -1;
            }else {
                if (x->data[0] > modulus.data[0]) {
                    return 1;
                }else if (x->data[0] < modulus.data[0]) {
                    return -1;
                }else {
                    return 0;
                }
            }
        }
    }
}

__device__ static void to_bits_array(const BigNum *a, unsigned int *b)
{
    for (int n = 0; n < DATA_SIZE; n++)
    {
        u_int64_t value = a->data[n];
        for (int i = 0; i < 64; i++)
        {
            b[n * 64 + i] = ((value & 1) != 0);
            value >>= 1;
        }
    }
}

__device__ void subtract_modulus(BigNum *a, BigNum modulus)
{
    USUBO1(a->data[0], modulus.data[0]);
    USUBC1(a->data[1], modulus.data[1]);
    USUBC1(a->data[2], modulus.data[2]);
    USUB1(a->data[3], modulus.data[3]);
}

__device__ BigNum fq_subtract(BigNum *a, BigNum *b)
{
    BigNum res = {0};
    int c = compare(a, b);
    if (c == 1){
        sub3(res.data, a->data, b->data);
    }else if (c == 0) {
        res = ZERO;
    }else {
        u_int64_t r[NBBLOCK] = {0};
        fq_add3(r, a->data);
        sub3(res.data, r, b->data);
    }
    return res;
}

__device__ BigNum fr_subtract(BigNum *a, BigNum *b)
{
    BigNum res = {0};
    int c = compare(a, b);
    if (c == 1){
        sub3(res.data, a->data, b->data);
    }else if (c == 0) {
        res = ZERO;
    }else {
        u_int64_t r[NBBLOCK] = {0};
        fr_add3(r, a->data);
        sub3(res.data, r, b->data);
    }
    return res;
}

__device__ BigNum add(BigNum *a, BigNum *b, BigNum modulus)
{
    BigNum res = {0};
    Add2(res.data, a->data, b->data);
    if (res.data[3] >= modulus.data[3])
    {
        subtract_modulus(&res, modulus);
    }
    return res;
}

// ---------------------------------------------------------------------------------------
// Compute a*b*(mod n)
// a and b must be lower than n
// ---------------------------------------------------------------------------------------
__device__ BigNum montgomeryMul(BigNum *a, BigNum *b, u_int64_t inv, BigNum modulus)
{
    BigNum r = {0};
    for (int i = 0; i < DATA_SIZE; i++)
    {
        u_int64_t t1 = 0, t2 = 0;
        UMULLO(t1, a->data[i], b->data[0]);
        UMULHI(t2, a->data[i], b->data[0]);
        UADDO(t1, t1, r.data[0]);
        UADDC(t2, t2, 0ULL);

        u_int64_t u_i = 0;
        UMULLO(u_i, t1, inv);

        u_int64_t carry1 = 0, carry2 = 0;

        for (int j = 0; j < DATA_SIZE; j++)
        {
            u_int64_t low = 0, t3 = 0, t4 = 0;
            if (j != 0)
            {
                UMULLO(t1, a->data[i], b->data[j]);
                UMULHI(t2, a->data[i], b->data[j]);
                UADDO(t1, t1, r.data[j]);
                UADDC(t2, t2, 0ULL);
            }
            UADD(low, carry1, t1);
            if (low < carry1)
            {
                UADD(carry1, t2, 1ULL);
            }
            else
            {
                UADD(carry1, t2, 0ULL);
            }
            UMULLO(t3, u_i, modulus.data[j]);
            UMULHI(t4, u_i, modulus.data[j]);
            UADDO(t3, t3, carry2);
            UADDC(t4, t4, 0ULL);
            UADD(r.data[j], low, t3);
            if (r.data[j] < low)
            {
                UADD(carry2, t4, 1ULL);
            }
            else
            {
                UADD(carry2, t4, 0ULL);
            }
        }
        for (int j = 0; j < DATA_SIZE - 1; j++)
        {
            r.data[j] = r.data[j + 1];
        }
        r.data[DATA_SIZE - 1] = carry1 + carry2;
    }

    if (compare_modulus(&r, modulus) == 1)
    {
        subtract_modulus(&r, modulus);
    }

    return r;
}

__device__ void copy_value(BigNum *a, BigNum *r)
{
    for(int i = 0; i < DATA_SIZE; i++) 
    {
        r->data[i] = a->data[i];
    }
}


__device__ BigNum fq_mul(BigNum *a, BigNum *b) 
{
    return montgomeryMul(a, b, FQ_INV, MODULUS);
}

__device__ BigNum fq_add(BigNum *a, BigNum *b) 
{
    return add(a, b, MODULUS);
}

__device__ BigNum fq_sub(BigNum *a, BigNum *b) 
{
    return fq_subtract(a, b);
}

__device__ BigNum fr_mul(BigNum *a, BigNum *b) 
{
    return montgomeryMul(a, b, FR_INV, FR_MODULUS);
}

__device__ BigNum fr_add(BigNum *a, BigNum *b) 
{
    return add(a, b, FR_MODULUS);
}

__device__ BigNum fr_sub(BigNum *a, BigNum *b) 
{
    return fr_subtract(a, b);
}

__device__ BigNum group_add(BigNum *a, BigNum *b) 
{
    return fr_add(a, b);
}

__device__ BigNum group_sub(BigNum *a, BigNum *b) 
{
    return fr_sub(a, b);
}

__device__ BigNum group_scale(BigNum *a, BigNum *b) 
{
    BigNum c = fr_mul(a, b);
    return c;
}

__device__ BigNum fq_square(BigNum *a) 
{
    BigNum c = fq_mul(a, a);
    return c;
}

__device__ int bit_reverse(u_int64_t n, u_int64_t l) 
{
    u_int64_t r = 0;
    for (u_int64_t i = 0; i < l; i++) {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

__device__ BigNum pow_vartime(BigNum *omega, BigNum *exp) 
{
    BigNum r = FR_R;
    for (int i = DATA_SIZE - 1; i >= 0; i--) {
        for (int j = 64 - 1; j >= 0; j--) {
            r = group_scale(&r, &r);
            if (((exp->data[i]) >> j) & 1 ) {
                r = group_scale(&r, omega);
            }
        }
    }
    return r;
}

__device__ BigNum from_int(int length) 
{
    BigNum b = {0};
    b.data[0] = length;
    b.data[1] = 0;
    b.data[2] = 0;
    b.data[3] = 0;
    return b;
}

__device__ u_int64_t from_le_bytes(u_int8_t *bytes)
{
    u_int64_t value = 0;
    for (int i = 0; i < 8; i++) {
        value |= (uint64_t)bytes[i] << (i * 8);
    }
    return value;
}

__device__ void to_le_bytes(u_int64_t value, u_int8_t *bytes)
{
    for (int i = 0; i < 8; i++) {
        bytes[i] = (uint8_t)((value >> (i * 8)) & 0xff);
    }
}

__device__ void bignum_to_bytes(BigNum value, u_int8_t *repr)
{
    for(int i = 0; i < 4; i++) {
        u_int8_t c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        to_le_bytes(value.data[i], c);
        for (int j = 0; j < 8; j++) {
            repr[i * 8 + j] = c[j];
        }
    }
}

#endif