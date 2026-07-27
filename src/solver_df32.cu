// 实验一 · G4 主配置: df32 (双单精度) 单融合核 Householder-3。
// 编译: nvcc -O3 -arch=sm_103 solver_df32.cu -o solver_df32
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "df32.cuh"

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ df dff(double v) {          // fp64 -> df32
    float h = (float)v; return {h, (float)(v - (double)h)};
}

__device__ void derivs(df x, df a, df b, df c, df dd, df w, df s0, df s1,
                        df& f, df& f1, df& f2, df& f3) {
    df ax = df_mul(a, x);
    df ex = df_exp(ax);
    df E  = df_mul(x, ex);
    df E1 = df_mul(ex, df_add(ax, mkdf(1.f)));
    df E2 = df_mul(df_mul(a, ex), df_add(ax, mkdf(2.f)));
    df E3 = df_mul(df_mul(df_mul(a, a), ex), df_add(ax, mkdf(3.f)));
    df u  = df_add(mkdf(1.f), df_mul(c, x));
    df bc = df_mul(b, c);
    df L1 = df_div(bc, u);
    df L2 = df_div((df){-1.f,0.f}, mkdf(1.f)), L3;   // placeholder
    L2 = df_sub(mkdf(0.f), df_div(df_mul(bc, c), df_mul(u, u)));
    L3 = df_div(df_mul(df_mul(mkdf(2.f), bc), df_mul(c, c)), df_mul(df_mul(u, u), u));
    df L  = df_mul(b, df_log(u));
    df wx = df_mul(w, x);
    df sn, cs; df_sincos(wx, sn, cs);
    df S  = df_mul(dd, sn);
    df dw = df_mul(dd, w);
    df S1 = df_mul(dw, cs);
    df S2 = df_sub(mkdf(0.f), df_mul(df_mul(dw, w), sn));
    df S3 = df_sub(mkdf(0.f), df_mul(df_mul(df_mul(dw, w), w), cs));
    f  = df_add(df_add(df_add(E, L), S), df_add(df_mul(s1, x), s0));
    f1 = df_add(df_add(df_add(E1, L1), S1), s1);
    f2 = df_add(df_add(E2, L2), S2);
    f3 = df_add(df_add(E3, L3), S3);
}

__device__ __forceinline__ df hh3(df f, df f1, df f2, df f3) {
    df two_f1sq = df_mul(mkdf(2.f), df_mul(f1, f1));
    df num = df_mul(df_mul(mkdf(3.f), f), df_sub(two_f1sq, df_mul(f, f2)));
    df t1  = df_mul(df_mul(mkdf(6.f), f), df_mul(f1, f2));
    df t2  = df_mul(df_mul(f, f), f3);
    df t3  = df_mul(df_mul(mkdf(6.f), f1), df_mul(f1, f1));
    df den = df_sub(df_sub(t1, t2), t3);
    df s = df_div(num, den);
    return isfinite(to_float(s)) ? s : mkdf(0.f);
}

__global__ void solve_g4(int N, const double* a, const double* b, const double* c, const double* dd,
                         const double* w, const double* s0, const double* s1, const double* x0,
                         double* xhat, int nstep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    df A = dff(a[i]), B = dff(b[i]), C = dff(c[i]), D = dff(dd[i]), W = dff(w[i]);
    df S0 = dff(s0[i]), S1 = dff(s1[i]);
    df x = dff(x0[i]), f, f1, f2, f3;
    for (int k = 0; k < nstep; k++) {
        derivs(x, A, B, C, D, W, S0, S1, f, f1, f2, f3);
        x = df_add(x, hh3(f, f1, f2, f3));
    }
    xhat[i] = (double)x.h + (double)x.l;
}

int main(int argc, char** argv) {
    const char* binpath = (argc > 1) ? argv[1] : "../data/inst.bin";
    int nstep = (argc > 2) ? atoi(argv[2]) : 8;
    FILE* fp = fopen(binpath, "rb"); if (!fp) { fprintf(stderr, "open fail\n"); return 1; }
    int N; if (fread(&N, 4, 1, fp) != 1) return 1;
    const int NCOL = 10;
    double* host = (double*)malloc((size_t)NCOL * N * sizeof(double));
    if (fread(host, sizeof(double), (size_t)NCOL * N, fp) != (size_t)NCOL * N) return 1;
    int* cert = (int*)malloc((size_t)N * sizeof(int));
    if (fread(cert, sizeof(int), N, fp) != (size_t)N) return 1;
    fclose(fp);
    double *pxmach = host + 9 * N;
    double *dev[8], *dxhat; size_t sz = (size_t)N * sizeof(double);
    for (int j = 0; j < 8; j++) { CK(cudaMalloc(&dev[j], sz)); CK(cudaMemcpy(dev[j], host + (long)j * N, sz, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&dxhat, sz));
    int th = 256, bl = (N + th - 1) / th;
    solve_g4<<<bl, th>>>(N, dev[0], dev[1], dev[2], dev[3], dev[4], dev[5], dev[6], dev[7], dxhat, nstep);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    double* xhat = (double*)malloc(sz); CK(cudaMemcpy(xhat, dxhat, sz, cudaMemcpyDeviceToHost));
    double* erc = (double*)malloc(sz); int nc = 0; double maxc = 0;
    for (int i = 0; i < N; i++) if (cert[i]) { double e = fabs(xhat[i] - pxmach[i]) / fabs(pxmach[i]); erc[nc++] = e; if (e > maxc) maxc = e; }
    std::sort(erc, erc + nc);
    auto q = [&](double t){ return nc ? erc[std::min(nc-1,(int)(t*nc))] : 0.0; };
    printf("[G4 df32] N=%d nstep=%d | 认证 %d: e_repr p50 %.3e p99 %.3e max %.3e\n", N, nstep, nc, q(0.5), q(0.99), maxc);
    return 0;
}
