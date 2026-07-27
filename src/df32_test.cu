// df32 精度自检: exp/log/sin/cos 对比 fp64 参考。
#include <cstdio>
#include <cmath>
#include "df32.cuh"

__global__ void test(const double* xin, int n, double* eexp, double* elog, double* esin, double* ecos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = xin[i];
    df dx = df_add(mkdf((float)x), mkdf((float)(x - (double)(float)x)));  // 把 fp64 拆进 df32
    df e = df_exp(dx);
    df l = df_log(dx);
    df s, c; df_sincos(dx, s, c);
    eexp[i] = fabs((double)to_float(e) + (double)e.l * 0 - exp(x)) / exp(x);
    // 用 h+l 双段还原
    double ev = (double)e.h + (double)e.l;
    double lv = (double)l.h + (double)l.l;
    double sv = (double)s.h + (double)s.l;
    double cv = (double)c.h + (double)c.l;
    eexp[i] = fabs(ev - exp(x)) / fabs(exp(x));
    elog[i] = fabs(lv - log(x)) / fabs(log(x));
    esin[i] = fabs(sv - sin(x));
    ecos[i] = fabs(cv - cos(x));
}

int main() {
    const int n = 20000;
    double *hx = new double[n];
    srand(1);
    for (int i = 0; i < n; i++) hx[i] = 0.05 + 12.0 * (rand() / (double)RAND_MAX);  // [0.05,12]
    double *dx, *de, *dl, *ds, *dc;
    cudaMalloc(&dx, n*8); cudaMalloc(&de, n*8); cudaMalloc(&dl, n*8); cudaMalloc(&ds, n*8); cudaMalloc(&dc, n*8);
    cudaMemcpy(dx, hx, n*8, cudaMemcpyHostToDevice);
    test<<<(n+255)/256, 256>>>(dx, n, de, dl, ds, dc);
    cudaDeviceSynchronize();
    double *he=new double[n],*hl=new double[n],*hs=new double[n],*hc=new double[n];
    cudaMemcpy(he,de,n*8,cudaMemcpyDeviceToHost); cudaMemcpy(hl,dl,n*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(hs,ds,n*8,cudaMemcpyDeviceToHost); cudaMemcpy(hc,dc,n*8,cudaMemcpyDeviceToHost);
    auto mx=[&](double*v){double m=0;for(int i=0;i<n;i++)if(v[i]>m)m=v[i];return m;};
    auto md=[&](double*v){double s=0;for(int i=0;i<n;i++)s+=v[i];return s/n;};
    printf("df32 精度 (vs fp64, n=%d, x in [0.05,12]):\n", n);
    printf("  exp 相对: 均 %.2e 最坏 %.2e\n", md(he), mx(he));
    printf("  log 相对: 均 %.2e 最坏 %.2e\n", md(hl), mx(hl));
    printf("  sin 绝对: 均 %.2e 最坏 %.2e\n", md(hs), mx(hs));
    printf("  cos 绝对: 均 %.2e 最坏 %.2e\n", md(hc), mx(hc));
    return 0;
}
