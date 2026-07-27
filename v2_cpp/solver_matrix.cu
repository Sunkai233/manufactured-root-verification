// solver_matrix.cu — M1–M9 阶次×精度全因子(补"组合C计时对照"): {Newton,Halley,HH3}×{fp64,df32,fp32}=9。
// 每格: 饱和吞吐(REP放大) + 精度 e_repr(对真根 x_mach_star, 固定 8 步) + 确定性。
// 编译: nvcc -O3 -arch=sm_103 solver_matrix.cu -o solver_matrix -I../src
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "df32.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

template<typename T> __device__ __forceinline__ void dv(T x,T a,T b,T c,T dd,T w,T s0,T s1,T&f,T&f1,T&f2,T&f3){
  T ex=exp(a*x); T E=x*ex,E1=ex*(a*x+T(1)),E2=a*ex*(a*x+T(2)),E3=a*a*ex*(a*x+T(3));
  T u=T(1)+c*x; T L1=b*c/u,L2=-b*c*c/(u*u),L3=T(2)*b*c*c*c/(u*u*u),L=b*log(u);
  T sw=sin(w*x),cw=cos(w*x); T S=dd*sw,S1=dd*w*cw,S2=-dd*w*w*sw,S3=-dd*w*w*w*cw;
  f=E+L+S+s1*x+s0; f1=E1+L1+S1+s1; f2=E2+L2+S2; f3=E3+L3+S3;
}
template<typename T> __device__ __forceinline__ T step_ord(int D,T f,T f1,T f2,T f3){
  if(D==1) return -f/f1;
  if(D==2) return -T(2)*f*f1/(T(2)*f1*f1-f*f2);
  T num=T(3)*f*(T(2)*f1*f1-f*f2); T den=T(6)*f*f1*f2-f*f*f3-T(6)*f1*f1*f1; return num/den;
}
struct In{const double*a,*b,*c,*d,*w,*s0,*s1;};
template<typename T,int D> __global__ void k_native(In in,const double*x0,double*xo,int N,int ns){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
  T A=in.a[i],B=in.b[i],C=in.c[i],Dd=in.d[i],W=in.w[i],S0=in.s0[i],S1=in.s1[i],x=(T)x0[i],f,f1,f2,f3;
  for(int k=0;k<ns;k++){ dv<T>(x,A,B,C,Dd,W,S0,S1,f,f1,f2,f3); T s=step_ord<T>(D,f,f1,f2,f3); if(isfinite(s))x+=s; }
  xo[i]=(double)x;
}
// df32 path
__device__ __forceinline__ df dffd(double v){float h=(float)v;return {h,(float)(v-(double)h)};}
__device__ void dvdf(df x,df a,df b,df c,df dd,df w,df s0,df s1,df&f,df&f1,df&f2,df&f3){
  df ax=df_mul(a,x),ex=df_exp(ax);
  df E=df_mul(x,ex),E1=df_mul(ex,df_add(ax,mkdf(1.f)));
  df E2=df_mul(df_mul(a,ex),df_add(ax,mkdf(2.f))),E3=df_mul(df_mul(df_mul(a,a),ex),df_add(ax,mkdf(3.f)));
  df u=df_add(mkdf(1.f),df_mul(c,x)),bc=df_mul(b,c);
  df L1=df_div(bc,u),L2=df_sub(mkdf(0.f),df_div(df_mul(bc,c),df_mul(u,u)));
  df L3=df_div(df_mul(df_mul(mkdf(2.f),bc),df_mul(c,c)),df_mul(df_mul(u,u),u)),L=df_mul(b,df_log(u));
  df wx=df_mul(w,x),sn,cs; df_sincos(wx,sn,cs);
  df S=df_mul(dd,sn),dw=df_mul(dd,w),S1=df_mul(dw,cs);
  df S2=df_sub(mkdf(0.f),df_mul(df_mul(dw,w),sn)),S3=df_sub(mkdf(0.f),df_mul(df_mul(df_mul(dw,w),w),cs));
  f=df_add(df_add(df_add(E,L),S),df_add(df_mul(s1,x),s0));
  f1=df_add(df_add(df_add(E1,L1),S1),s1); f2=df_add(df_add(E2,L2),S2); f3=df_add(df_add(E3,L3),S3);
}
__device__ __forceinline__ df step_df(int D,df f,df f1,df f2,df f3){
  if(D==1) return df_sub(mkdf(0.f),df_div(f,f1));
  if(D==2){ df num=df_mul(mkdf(-2.f),df_mul(f,f1)); df den=df_sub(df_mul(mkdf(2.f),df_mul(f1,f1)),df_mul(f,f2)); return df_div(num,den);}
  df num=df_mul(df_mul(mkdf(3.f),f),df_sub(df_mul(mkdf(2.f),df_mul(f1,f1)),df_mul(f,f2)));
  df den=df_sub(df_sub(df_mul(df_mul(mkdf(6.f),f),df_mul(f1,f2)),df_mul(df_mul(f,f),f3)),df_mul(df_mul(mkdf(6.f),f1),df_mul(f1,f1)));
  return df_div(num,den);
}
// fp32 快速内建路径(与 solver_timed 的 G5 一致: __expf/__logf/__sinf, 否则准确 expf 慢 ~1.4×)
__device__ __forceinline__ void dvf(float x,float a,float b,float c,float dd,float w,float s0,float s1,float&f,float&f1,float&f2,float&f3){
  float ex=__expf(a*x); float E=x*ex,E1=ex*(a*x+1),E2=a*ex*(a*x+2),E3=a*a*ex*(a*x+3);
  float u=1+c*x; float L1=b*c/u,L2=-b*c*c/(u*u),L3=2*b*c*c*c/(u*u*u),L=b*__logf(u);
  float sw=__sinf(w*x),cw=__cosf(w*x); float S=dd*sw,S1=dd*w*cw,S2=-dd*w*w*sw,S3=-dd*w*w*w*cw;
  f=E+L+S+s1*x+s0; f1=E1+L1+S1+s1; f2=E2+L2+S2; f3=E3+L3+S3;
}
template<int D> __global__ void k_f32fast(In in,const double*x0,double*xo,int N,int ns){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
  float A=in.a[i],B=in.b[i],C=in.c[i],Dd=in.d[i],W=in.w[i],S0=in.s0[i],S1=in.s1[i],x=(float)x0[i],f,f1,f2,f3;
  for(int k=0;k<ns;k++){ dvf(x,A,B,C,Dd,W,S0,S1,f,f1,f2,f3); float s=step_ord<float>(D,f,f1,f2,f3); if(isfinite(s))x+=s; }
  xo[i]=(double)x;
}
template<int D> __global__ void k_df(In in,const double*x0,double*xo,int N,int ns){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
  df A=dffd(in.a[i]),B=dffd(in.b[i]),C=dffd(in.c[i]),Dd=dffd(in.d[i]),W=dffd(in.w[i]),S0=dffd(in.s0[i]),S1=dffd(in.s1[i]),x=dffd(x0[i]),f,f1,f2,f3;
  for(int k=0;k<ns;k++){ dvdf(x,A,B,C,Dd,W,S0,S1,f,f1,f2,f3); df s=step_df(D,f,f1,f2,f3); if(isfinite(to_float(s))) x=df_add(x,s);}
  xo[i]=(double)x.h+(double)x.l;
}
double med(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}

int main(int argc,char**argv){
  const char*bin=argv[1]; int ns=argc>2?atoi(argv[2]):8; int REP=argc>3?atoi(argv[3]):256;
  FILE*fp=fopen(bin,"rb");int N0;if(fread(&N0,4,1,fp)!=1)return 1;
  std::vector<double> h((size_t)10*N0); if(fread(h.data(),8,(size_t)10*N0,fp)!=(size_t)10*N0)return 1; fclose(fp);
  long N=(long)N0*REP; std::vector<double> H((size_t)10*N);
  for(int j=0;j<10;j++) for(long i=0;i<N;i++) H[(size_t)j*N+i]=h[(size_t)j*N0+(i%N0)];
  double*dev[8],*dx0,*dxo; size_t sz=(size_t)N*8;
  for(int j=0;j<8;j++){CK(cudaMalloc(&dev[j],sz));CK(cudaMemcpy(dev[j],H.data()+(size_t)j*N,sz,cudaMemcpyHostToDevice));}
  In in{dev[0],dev[1],dev[2],dev[3],dev[4],dev[5],dev[6]}; dx0=dev[7]; CK(cudaMalloc(&dxo,sz));
  int th=256,bl=(int)((N+th-1)/th); cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
  std::vector<double> xstar(N0); for(int i=0;i<N0;i++) xstar[i]=h[(size_t)9*N0+i];
  printf("prec,order,Mroot_s,relerr_p50,relerr_p99,det\n");
  auto bench=[&](const char*pr,int ord,auto launch){
    for(int i=0;i<10;i++){launch();} CK(cudaDeviceSynchronize());
    std::vector<double> tk;
    for(int r=0;r<30;r++){ cudaEventRecord(e0); launch(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms=0;cudaEventElapsedTime(&ms,e0,e1); tk.push_back(ms);}
    double tkm=med(tk);
    std::vector<double> xo(N); CK(cudaMemcpy(xo.data(),dxo,sz,cudaMemcpyDeviceToHost));
    std::vector<double> xo2(N); launch(); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(xo2.data(),dxo,sz,cudaMemcpyDeviceToHost));
    int det=1; for(long i=0;i<N;i++) if(xo[i]!=xo2[i]){det=0;break;}
    std::vector<double> re(N0); for(int i=0;i<N0;i++){double xs=xstar[i]; re[i]=fabs(xo[i]-xs)/fmax(fabs(xs),1.0);} std::sort(re.begin(),re.end());
    printf("%s,%d,%.1f,%.3e,%.3e,%d\n",pr,ord,N/1e6/(tkm/1e3),re[N0/2],re[(size_t)(0.99*(N0-1))],det);
  };
  bench("fp64",2,[&]{k_native<double,1><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  bench("fp64",3,[&]{k_native<double,2><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  bench("fp64",4,[&]{k_native<double,3><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  bench("df32",2,[&]{k_df<1><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  bench("df32",3,[&]{k_df<2><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  bench("df32",4,[&]{k_df<3><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  bench("fp32",2,[&]{k_f32fast<1><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});  // 快速内建, 与 G5 一致
  bench("fp32",3,[&]{k_f32fast<2><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  bench("fp32",4,[&]{k_f32fast<3><<<bl,th>>>(in,dx0,dxo,(int)N,ns);});
  return 0;
}
