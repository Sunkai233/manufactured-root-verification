// 实验一 · 步骤3 CUDA 求解器 (G3 单融合核, fp64/fp32)
// 制造残差 f(x)=x e^{ax}+b ln(1+cx)+d sin(wx)+s1 x+s0
// Householder-3 直接有理式步长(无除零)。编译:
//   nvcc -O3 -arch=sm_103 solver.cu -o solver_fp64
//   nvcc -O3 -arch=sm_103 -DUSE_FP32 solver.cu -o solver_fp32
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#ifdef USE_FP32
typedef float real;
#define REXP __expf
#define RLOG __logf
#define RSIN __sinf
#define RCOS __cosf
const char* PREC = "fp32";
#else
typedef double real;
#define REXP exp
#define RLOG log
#define RSIN sin
#define RCOS cos
const char* PREC = "fp64";
#endif

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ void derivs(real x, real a, real b, real c, real dd, real w,
                                        real s0, real s1, real& f, real& f1, real& f2, real& f3) {
    real ex = REXP(a * x);
    real E = x * ex, E1 = ex * (a * x + (real)1), E2 = a * ex * (a * x + (real)2), E3 = a * a * ex * (a * x + (real)3);
    real u = (real)1 + c * x;
    real L1 = b * c / u, L2 = -b * c * c / (u * u), L3 = (real)2 * b * c * c * c / (u * u * u);
    real L = b * RLOG(u);
    real sw = RSIN(w * x), cw = RCOS(w * x);
    real S = dd * sw, S1 = dd * w * cw, S2 = -dd * w * w * sw, S3 = -dd * w * w * w * cw;
    f = E + L + S + s1 * x + s0;
    f1 = E1 + L1 + S1 + s1;
    f2 = E2 + L2 + S2;
    f3 = E3 + L3 + S3;
}

__device__ __forceinline__ real hh3_step(real f, real f1, real f2, real f3) {
    real num = (real)3 * f * ((real)2 * f1 * f1 - f * f2);
    real den = (real)6 * f * f1 * f2 - f * f * f3 - (real)6 * f1 * f1 * f1;
    real s = num / den;
    return isfinite(s) ? s : (real)0;
}

// G3: 单融合核, 固定步数, 无分支
__global__ void solve_g3(int N, const double* a, const double* b, const double* c, const double* dd,
                         const double* w, const double* s0, const double* s1, const double* x0,
                         double* xhat, int nstep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    real A = (real)a[i], B = (real)b[i], C = (real)c[i], D = (real)dd[i], W = (real)w[i];
    real S0 = (real)s0[i], S1 = (real)s1[i];
    real x = (real)x0[i], f, f1, f2, f3;
    for (int k = 0; k < nstep; k++) {
        derivs(x, A, B, C, D, W, S0, S1, f, f1, f2, f3);
        x = x + hh3_step(f, f1, f2, f3);
    }
    xhat[i] = (double)x;
}

int main(int argc, char** argv) {
    const char* binpath = (argc > 1) ? argv[1] : "../data/inst.bin";
    int nstep = (argc > 2) ? atoi(argv[2]) : 8;
    FILE* fp = fopen(binpath, "rb");
    if (!fp) { fprintf(stderr, "open fail %s\n", binpath); return 1; }
    int N; if (fread(&N, 4, 1, fp) != 1) return 1;
    const int NCOL = 10;
    double* host = (double*)malloc((size_t)NCOL * N * sizeof(double));
    if (fread(host, sizeof(double), (size_t)NCOL * N, fp) != (size_t)NCOL * N) return 1;
    int* cert = (int*)malloc((size_t)N * sizeof(int));
    if (fread(cert, sizeof(int), N, fp) != (size_t)N) return 1;
    fclose(fp);
    double *pa = host, *pb = host + N, *pc = host + 2 * N, *pd = host + 3 * N, *pw = host + 4 * N;
    double *ps0 = host + 5 * N, *ps1 = host + 6 * N, *px0 = host + 7 * N;
    double *pxmach = host + 9 * N;

    double *da, *db, *dc, *dd_, *dw, *ds0, *ds1, *dx0, *dxhat;
    size_t sz = (size_t)N * sizeof(double);
    for (double** p : {&da, &db, &dc, &dd_, &dw, &ds0, &ds1, &dx0, &dxhat}) CK(cudaMalloc(p, sz));
    CK(cudaMemcpy(da, pa, sz, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, pb, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, pc, sz, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd_, pd, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, pw, sz, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds0, ps0, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds1, ps1, sz, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx0, px0, sz, cudaMemcpyHostToDevice));

    int th = 256, bl = (N + th - 1) / th;
    solve_g3<<<bl, th>>>(N, da, db, dc, dd_, dw, ds0, ds1, dx0, dxhat, nstep);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    double* xhat = (double*)malloc(sz);
    CK(cudaMemcpy(xhat, dxhat, sz, cudaMemcpyDeviceToHost));

    // e_repr vs 机器真根, 分认证/未认证
    double* erc = (double*)malloc(sz); int nc = 0, nu = 0; double* eru = (double*)malloc(sz);
    double maxc = 0;
    for (int i = 0; i < N; i++) {
        double e = fabs(xhat[i] - pxmach[i]) / fabs(pxmach[i]);
        if (cert[i]) { erc[nc++] = e; if (e > maxc) maxc = e; }
        else eru[nu++] = e;
    }
    std::sort(erc, erc + nc); std::sort(eru, eru + nu);
    auto pc_ = [&](double* v, int n, double q){ return n ? v[std::min(n-1,(int)(q*n))] : 0.0; };
    printf("[G3 %s] N=%d nstep=%d | 认证 %d: e_repr p50 %.3e p99 %.3e max %.3e | 未认证 %d: e_repr p50 %.3e\n",
           PREC, N, nstep, nc, pc_(erc,nc,0.5), pc_(erc,nc,0.99), maxc, nu, pc_(eru,nu,0.5));
    return 0;
}
