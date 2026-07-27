// df32: 双单精度 (double-single), 两个 float32 表示 ~45 位有效精度。
// 值 ≈ h + l, |l| <= 0.5 ulp(h)。基础运算 + exp/log/sin/cos。
#pragma once
#include <cuda_runtime.h>
#include <cmath>

struct df { float h, l; };

__device__ __forceinline__ df mkdf(float x) { return {x, 0.f}; }
__device__ __forceinline__ float to_float(df a) { return a.h + a.l; }

__device__ __forceinline__ df quick_two_sum(float a, float b) {
    float s = a + b; float e = b - (s - a); return {s, e};
}
__device__ __forceinline__ df two_sum(float a, float b) {
    float s = a + b; float bb = s - a; float e = (a - (s - bb)) + (b - bb); return {s, e};
}
__device__ __forceinline__ df two_prod(float a, float b) {
    float p = a * b; float e = fmaf(a, b, -p); return {p, e};
}
__device__ __forceinline__ df df_add(df a, df b) {
    df s = two_sum(a.h, b.h); s.l += a.l + b.l; return quick_two_sum(s.h, s.l);
}
__device__ __forceinline__ df df_sub(df a, df b) { return df_add(a, {-b.h, -b.l}); }
__device__ __forceinline__ df df_mul(df a, df b) {
    df p = two_prod(a.h, b.h); p.l += a.h * b.l + a.l * b.h; return quick_two_sum(p.h, p.l);
}
__device__ __forceinline__ df df_div(df a, df b) {
    float q1 = a.h / b.h;
    df r = df_sub(a, df_mul(b, mkdf(q1)));
    float q2 = r.h / b.h;
    r = df_sub(r, df_mul(b, mkdf(q2)));
    float q3 = r.h / b.h;
    df q = quick_two_sum(q1, q2); q = df_add(q, mkdf(q3)); return q;
}
__device__ __forceinline__ df df_scale2(df a, int k) {   // a * 2^k (精确)
    return {ldexpf(a.h, k), ldexpf(a.l, k)};
}

// 常数 (高低两段)
__device__ __constant__ float LN2_H = 0.6931472f, LN2_L = -1.904654e-09f;
__device__ __constant__ float PIO2_H = 1.5707964f, PIO2_L = -4.371139e-08f;

// exp: x = k*ln2 + r, exp = 2^k * exp(r), exp(r) Taylor (|r|<=ln2/2)
__device__ __forceinline__ df df_exp(df x) {
    float kf = floorf(x.h / LN2_H + 0.5f);
    int k = (int)kf;
    df r = df_sub(x, df_mul(mkdf(kf), (df){LN2_H, LN2_L}));
    // exp(r) = sum r^n/n!, 12 项
    df term = mkdf(1.f), sum = mkdf(1.f);
    #pragma unroll
    for (int n = 1; n <= 12; n++) {
        term = df_mul(term, df_div(r, mkdf((float)n)));
        sum = df_add(sum, term);
    }
    return df_scale2(sum, k);
}
// log: 以 fp32 为种子, 用 df_exp 做 2 次 Newton: y += x*exp(-y) - 1
__device__ __forceinline__ df df_log(df x) {
    df y = mkdf(__logf(x.h));
    #pragma unroll
    for (int it = 0; it < 3; it++) {
        df em = df_exp((df){-y.h, -y.l});          // exp(-y) ~ 1/x
        df corr = df_sub(df_mul(x, em), mkdf(1.f)); // x*exp(-y) - 1
        y = df_add(y, corr);
    }
    return y;
}
// sin/cos: 约化 x = k*(pi/2) + r, |r|<=pi/4, 按 k mod 4 选多项式
__device__ __forceinline__ void df_sincos(df x, df& s, df& c) {
    float kf = floorf(x.h / PIO2_H + 0.5f);
    int k = ((int)kf) & 3;
    df r = df_sub(x, df_mul(mkdf(kf), (df){PIO2_H, PIO2_L}));
    df r2 = df_mul(r, r);
    // sin(r), cos(r) Taylor, 各 ~8 项
    df sn = r, tn = r;
    #pragma unroll
    for (int n = 1; n <= 7; n++) {
        tn = df_mul(tn, df_div((df){-r2.h, -r2.l}, mkdf((float)((2*n)*(2*n+1)))));
        sn = df_add(sn, tn);
    }
    df cs = mkdf(1.f), tc = mkdf(1.f);
    #pragma unroll
    for (int n = 1; n <= 7; n++) {
        tc = df_mul(tc, df_div((df){-r2.h, -r2.l}, mkdf((float)((2*n-1)*(2*n)))));
        cs = df_add(cs, tc);
    }
    df ms = {-sn.h, -sn.l}, mc = {-cs.h, -cs.l};
    if (k == 0) { s = sn; c = cs; }
    else if (k == 1) { s = cs; c = ms; }
    else if (k == 2) { s = ms; c = mc; }
    else { s = mc; c = sn; }
}
