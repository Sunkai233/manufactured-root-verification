// multiroot.cpp v2 — 多根扫描(修 v3 三缝: 用 κ 不用 γ; 认证用 long double; 关 FMA 收缩)。
// ★FMA 收缩会让近根 f 的符号判定失真(旧 -march=native 得 38.6%, 关掉/细网格收敛到 ≈76.5%)。
// 编译: g++ -O3 -march=native -ffp-contract=off multiroot.cpp -o multiroot -Iinclude
#pragma STDC FP_CONTRACT OFF
#include "mrv.hpp"
#include "alpha.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
using namespace mrv;
struct Data{ int N; std::vector<double> a,b,c,d,w,s0,s1,x0,gam,xs; std::vector<double> kap; };
Data load(const char* fn,const char* kf){
  FILE* f=fopen(fn,"rb"); int N; if(fread(&N,4,1,f)!=1)exit(1);
  std::vector<double> buf((size_t)10*N); if(fread(buf.data(),8,(size_t)10*N,f)!=(size_t)10*N)exit(1);
  Data D; D.N=N; auto col=[&](int j){return std::vector<double>(buf.begin()+(size_t)j*N,buf.begin()+(size_t)(j+1)*N);};
  D.a=col(0);D.b=col(1);D.c=col(2);D.d=col(3);D.w=col(4);D.s0=col(5);D.s1=col(6);D.x0=col(7);D.gam=col(8);D.xs=col(9);
  fclose(f);
  // ★硬报错: 不静默回退 γ(否则会重蹈 γ 冒充 κ 的 bug)。kappa.bin 由 build_canonical.py 生成。
  D.kap.resize(N); FILE* fk=fopen(kf,"rb");
  if(!fk){ fprintf(stderr,"ERROR: 打不开 %s —— 先跑 build_canonical.py 生成 kappa.bin(parquet 的 kappa 列)\n",kf); exit(1); }
  if(fread(D.kap.data(),8,N,fk)!=(size_t)N){ fprintf(stderr,"ERROR: %s 短读, 期望 %d 个 double\n",kf,N); fclose(fk); exit(1); }
  fclose(fk);
  return D;
}
inline bool in_dom(double x,double c){ return (1.0+c*x)>1e-300; }
inline double Ssum(double x,const Params<double>&p){
  double ex=std::exp(p.a*x);
  return std::fabs(x*ex)+std::fabs(p.b*std::log(std::fabs(1.0+p.c*x)+1e-300))
        +std::fabs(p.d*std::sin(p.w*x))+std::fabs(p.s1*x)+std::fabs(p.s0);
}
// long double 网格扫描(近根 f 相消致漏根, 必须升精度; double 会系统性少数)
int nroots(const Params<long double>& p,long double lo,long double hi,int ng){
  long double fp=0; bool have=false; int cnt=0;
  for(int i=0;i<ng;i++){
    long double x=lo+(hi-lo)*(long double)i/(ng-1); if(!((1.0L+p.c*x)>1e-300L)){ have=false; continue; }
    long double f=resid<long double>(x,p);
    if(have && fp*f<0) cnt++;
    fp=f; have=true;
  }
  return cnt;
}
int main(int argc,char**argv){
  const char* kf=argc>2?argv[2]:"kappa.bin";
  int ng=argc>3?atoi(argv[3]):12001;
  Data D=load(argv[1],kf);
  const double u=std::ldexp(1.0,-53), SNR=1e4; long double thrl=alpha_threshold_t<long double>();
  long wc=0,wc2=0,wc3=0, cdec=0,csec=0,ctot=0;
  for(int i=0;i<D.N;i++){
    Params<double> p{D.a[i],D.b[i],D.c[i],D.d[i],D.w[i],D.s0[i],D.s1[i]};
    Params<long double> pl{(long double)D.a[i],(long double)D.b[i],(long double)D.c[i],(long double)D.d[i],(long double)D.w[i],(long double)D.s0[i],(long double)D.s1[i]};
    double xs=D.xs[i];
    if(D.kap[i]<1e4){                       // ★良态用 κ(不是 γ)
      wc++; int nr=nroots(pl,(long double)xs-0.2L,(long double)xs+0.2L,ng);
      if(nr>=2)wc2++; if(nr>=3)wc3++;
    }
    long double al=alpha_t<long double>((long double)D.x0[i],pl);   // ★long double 认证
    if(al<thrl){
      ctot++;
      double x0=D.x0[i];
      long double beta=beta_t<long double>((long double)x0,pl); double r=2.0*(double)beta;
      double f,f1,f2,f3; derivs(x0,p,f,f1,f2,f3);              // ★在球心 x0 判可判性
      double S=Ssum(x0,p), signal=std::fabs(f1)*r;
      if(r>0 && signal>SNR*u*S){ cdec++;
        // ★Smale 球以 x0 为心、半径 2β(不是以 x* 为心)。检验认证球内是否含非目标零点(≠x*)。
        int nr=nroots(pl,(long double)(x0-r),(long double)(x0+r),201);
        bool xstar_in = std::fabs(xs-x0) < r;                  // x* 是否落在认证球内
        bool nontarget = (nr>=2) || (nr>=1 && !xstar_in);      // 球内存在 ≠x* 的零点=认证收敛到非目标根
        if(nontarget) csec++;
      }
    }
  }
  printf("多根扫描 v2 (N=%d, ng=%d, FMA收缩关闭)\n",D.N,ng);
  printf("(a) 良态 κ<1e4: %ld 个 | x*±0.2 内 ≥2根 %ld (%.1f%%) | ≥3根 %ld (%.1f%%)  ← C4 压力(真值)\n",
         wc,wc2,100.0*wc2/std::max<long>(wc,1),wc3,100.0*wc3/std::max<long>(wc,1));
  printf("(b) 认证子集(long double) %ld | 数值可判 %ld (%.1f%%) | 以x0为心认证球[x0±2β]内含非目标零点(≠x*) %ld (%.2f%%)  ← 应与C5的20.8%%量级对齐\n",
         ctot,cdec,100.0*cdec/std::max<long>(ctot,1),csec,100.0*csec/std::max<long>(cdec,1));
  return 0;
}
