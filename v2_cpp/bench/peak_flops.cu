// 峰值吞吐微基准: 实测本卡 fp32/fp64/df32 的 FMA 吞吐, 求 R=吞吐(fp32)/吞吐(fp64)。
// 高算术强度 + 每线程 NACC 个独立累加器(ILP 打满发射, 隐藏延迟) → 逼近峰值。
// 用于核实"B300 fp64 是否被削减"(预期 R~62 则削减, R~2 则满速)。
// 编译: nvcc -O3 -arch=sm_103 peak_flops.cu -o peak_flops -I../../src
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "df32.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define NACC 32

template<typename T> __global__ void peakK(T* out, T s, int iters){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  T acc[NACC];
  #pragma unroll
  for(int j=0;j<NACC;j++) acc[j]=s+T(j)*T(1e-3);
  T b=s*T(0.9999)+T(1e-4);
  for(int i=0;i<iters;i++){
    #pragma unroll
    for(int j=0;j<NACC;j++) acc[j]=acc[j]*b+T(0.1);   // 1 FMA = 2 flop
  }
  T r=T(0);
  #pragma unroll
  for(int j=0;j<NACC;j++) r+=acc[j];
  out[t]=r;
}
// df32: 每个累加器一次 df_mul + df_add(近似一次"df-FMA"), 记 df-有效 flop = 2
__global__ void peakDF(double* out, double s, int iters){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  df acc[NACC];
  #pragma unroll
  for(int j=0;j<NACC;j++) acc[j]=mkdf((float)(s+j*1e-3));
  df b=mkdf((float)(s*0.9999+1e-4)), c=mkdf(0.1f);
  for(int i=0;i<iters;i++){
    #pragma unroll
    for(int j=0;j<NACC;j++) acc[j]=df_add(df_mul(acc[j],b),c);
  }
  double r=0;
  #pragma unroll
  for(int j=0;j<NACC;j++) r+=(double)acc[j].h+(double)acc[j].l;
  out[t]=r;
}

double med(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}

template<typename F> double timeit(F launch, int reps=20){
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for(int i=0;i<5;i++){launch();} CK(cudaDeviceSynchronize());
  std::vector<double> ts;
  for(int r=0;r<reps;r++){ cudaEventRecord(e0); launch(); cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1)); float ms=0; cudaEventElapsedTime(&ms,e0,e1); ts.push_back(ms);}
  return med(ts);
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("GPU: %s  SM=%d  cc=%d.%d\n",p.name,p.multiProcessorCount,p.major,p.minor);
  int th=256, bl=p.multiProcessorCount*32; long threads=(long)th*bl;
  int iters=4096;
  double flop = (double)threads*iters*NACC*2.0;   // 每 iter 每 acc 1 FMA=2 flop
  double *d; CK(cudaMalloc(&d,threads*8));
  // fp32
  double t32=timeit([&]{peakK<float><<<bl,th>>>((float*)d,1.0001f,iters);});
  // fp64
  double t64=timeit([&]{peakK<double><<<bl,th>>>(d,1.0001,iters);});
  // df32
  double tdf=timeit([&]{peakDF<<<bl,th>>>(d,1.0001,iters);});
  double g32=flop/(t32/1e3)/1e9, g64=flop/(t64/1e3)/1e9, gdf=flop/(tdf/1e3)/1e9;
  printf("threads=%ld iters=%d NACC=%d  total_FMA=%.2e\n",threads,iters,NACC,flop/2);
  printf("fp32 : %8.3f ms  %9.1f GFLOP/s\n",t32,g32);
  printf("fp64 : %8.3f ms  %9.1f GFLOP/s\n",t64,g64);
  printf("df32 : %8.3f ms  %9.1f GFLOP/s (df-有效)\n",tdf,gdf);
  printf(">>> R = 吞吐(fp32)/吞吐(fp64) = %.1f   (若~62=fp64被削减, ~2=满速)\n", g32/g64);
  printf(">>> df32 相对 fp64: %.2fx   df32 相对 fp32: %.3fx\n", gdf/g64, gdf/g32);
  printf(">>> 每个 df32 op 折合 fp32 指令数 C ≈ %.1f\n", g32/gdf);
  return 0;
}
