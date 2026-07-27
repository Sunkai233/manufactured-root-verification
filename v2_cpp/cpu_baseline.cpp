// 诚实 CPU 基线: 纠正报告"3917×"(那是 Python 解释器开销)。
// 同算法(HH3 固定步, 与 GPU G3 一致)单核 + 真 OpenMP; 另加 Brent 稳健参照(同 scipy brentq 算法)。
// 编译: g++ -O3 -march=native -fopenmp cpu_baseline.cpp -o cpu_baseline -Iinclude
#include "mrv.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace mrv;
using Clock=std::chrono::high_resolution_clock;

struct Data{ int N; std::vector<double> a,b,c,d,w,s0,s1,x0,gam,xs; std::vector<int> cert; };

Data load(const char* fn){
  FILE* f=fopen(fn,"rb"); if(!f){fprintf(stderr,"open %s fail\n",fn);exit(1);}
  int N; if(fread(&N,4,1,f)!=1){exit(1);}
  std::vector<double> buf((size_t)10*N);
  if(fread(buf.data(),8,(size_t)10*N,f)!=(size_t)10*N){exit(1);}
  Data D; D.N=N;
  auto col=[&](int j){return std::vector<double>(buf.begin()+(size_t)j*N, buf.begin()+(size_t)(j+1)*N);};
  D.a=col(0);D.b=col(1);D.c=col(2);D.d=col(3);D.w=col(4);D.s0=col(5);D.s1=col(6);D.x0=col(7);D.gam=col(8);D.xs=col(9);
  D.cert.resize(N); if(fread(D.cert.data(),4,N,f)!=(size_t)N){/*旧文件可能无*/}
  fclose(f); return D;
}

// Brent(van Wijngaarden–Dekker–Brent), 带定义域感知的括号扩张(不使用真根)
static inline double clamp_domain(double x,double c){
  // 需 1+c*x>0: c>0 => x>-1/c; c<0 => x<-1/c
  if(c>0){ double lo=-1.0/c; if(x<=lo) x=lo*0.999999+1e-12; }
  else if(c<0){ double hi=-1.0/c; if(x>=hi) x=hi*0.999999-1e-12; }
  return x;
}
bool brent(const Params<double>& p,double x0,double& root,long& fev){
  double h=std::fmax(1e-3,1e-3*std::fabs(x0)), lo=0,hi=0,flo=0,fhi=0; bool ok=false;
  for(int i=0;i<80;i++){
    lo=clamp_domain(x0-h,p.c); hi=clamp_domain(x0+h,p.c);
    flo=resid(lo,p); fhi=resid(hi,p); fev+=2;
    if(std::isfinite(flo)&&std::isfinite(fhi)&&flo*fhi<0){ok=true;break;} h*=1.6;
  }
  if(!ok) return false;
  double a=lo,b=hi,fa=flo,fb=fhi,cc=a,fc=fa,d=b-a,e=d;
  for(int it=0;it<100;it++){
    if(fb*fc>0){cc=a;fc=fa;d=b-a;e=d;}
    if(std::fabs(fc)<std::fabs(fb)){a=b;b=cc;cc=a;fa=fb;fb=fc;fc=fa;}
    double tol=2*2.220446049250313e-16*std::fabs(b)+0.5*1e-14;
    double m=0.5*(cc-b);
    if(std::fabs(m)<=tol||fb==0){root=b;return true;}
    if(std::fabs(e)>=tol&&std::fabs(fa)>std::fabs(fb)){
      double s=fb/fa,pp,q;
      if(a==cc){pp=2*m*s;q=1-s;}
      else{q=fa/fc;double r=fb/fc;pp=s*(2*m*q*(q-r)-(b-a)*(r-1));q=(q-1)*(r-1)*(s-1);}
      if(pp>0)q=-q; else pp=-pp;
      if(2*pp<std::fmin(3*m*q-std::fabs(tol*q),std::fabs(e*q))){e=d;d=pp/q;}
      else{d=m;e=m;}
    } else {d=m;e=m;}
    a=b;fa=fb;
    b += (std::fabs(d)>tol)? d : (m>0?tol:-tol);
    fb=resid(b,p); fev++;
    if(!std::isfinite(fb)) return false;
  }
  root=b; return true;
}

template<typename F> double best_time(F fn,int reps=5){
  fn(); // warmup
  double best=1e300;
  for(int r=0;r<reps;r++){ auto t0=Clock::now(); fn(); auto t1=Clock::now();
    best=std::fmin(best,std::chrono::duration<double>(t1-t0).count()); }
  return best;
}

double median_relerr(const std::vector<double>& xh,const std::vector<double>& xs){
  std::vector<double> e; e.reserve(xh.size());
  for(size_t i=0;i<xh.size();i++){ double den=std::fmax(std::fabs(xs[i]),1.0); e.push_back(std::fabs(xh[i]-xs[i])/den); }
  std::sort(e.begin(),e.end()); return e[e.size()/2];
}

int main(int argc,char**argv){
  const char* bin=argv[1]; int nsteps=argc>2?atoi(argv[2]):8;
  int REP=argc>3?atoi(argv[3]):256;   // 复制放大 N, 让 OpenMP 扩展性测量脱离调度噪声
  Data D=load(bin);
  int N0=D.N; long N=(long)N0*REP;
  // 平铺放大(SoA 参数 + 初值)
  std::vector<Params<double>> P(N); std::vector<double> X0(N);
  for(long i=0;i<N;i++){ int j=i%N0; P[i]={D.a[j],D.b[j],D.c[j],D.d[j],D.w[j],D.s0[j],D.s1[j]}; X0[i]=D.x0[j]; }
  std::vector<double> out(N);
  printf("N0=%d  REP=%d  有效N=%ld  nsteps=%d\n",N0,REP,N,nsteps);

  // (1) HH3 固定步, 单核(与 GPU G3 同算法同步数)
  double t1=best_time([&]{ for(long i=0;i<N;i++) out[i]=solve_fixed<double,3>(X0[i],P[i],nsteps); });
  // 精度(取前 N0 个对真根)
  std::vector<double> o0(out.begin(),out.begin()+N0);
  double err1=median_relerr(o0,D.xs);
  double mrs1=N/1e6/t1;

  // (2) HH3 固定步, OpenMP 全核
  int nth=1;
#ifdef _OPENMP
  nth=omp_get_max_threads();
#endif
  double tO=best_time([&]{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(long i=0;i<N;i++) out[i]=solve_fixed<double,3>(X0[i],P[i],nsteps);
  });
  double mrsO=N/1e6/tO;

  // (1b) Newton(2阶) / Halley(3阶) 单核, 同框架对照阶次成本
  double tN=best_time([&]{ for(long i=0;i<N;i++) out[i]=solve_fixed<double,1>(X0[i],P[i],nsteps); });
  double tH=best_time([&]{ for(long i=0;i<N;i++) out[i]=solve_fixed<double,2>(X0[i],P[i],nsteps); });
  double mrsN=N/1e6/tN, mrsH=N/1e6/tH;

  // (3) Brent 从头稳健参照(仅原始 N0, 同 scipy brentq 算法): 诚实脚注——
  //     从 x0 括号扩张会 bracket 到别的根(制造族多根), 非"解目标根"公平基线, 仅列其编译版吞吐远高于 Python 解释器口径。
  std::vector<double> outb(N0); std::vector<char> okb(N0); long fev=0;
  double tb=best_time([&]{ long fe=0; for(int i=0;i<N0;i++){ double r; okb[i]=brent(P[i],D.x0[i],r,fe); outb[i]=okb[i]?r:D.x0[i]; } fev=fe; },3);
  long nok=0; for(int i=0;i<N0;i++) nok+=okb[i];
  double mrsB=N0/1e6/tb;

  printf("[HH3固定步·同GPU算法] 单核  : %.3f ms  %.2f M根/s  rel_err_p50=%.2e\n",t1*1e3,mrs1,err1);
  printf("[HH3固定步·真OpenMP ] %3d核 : %.3f ms  %.2f M根/s  并行效率=%.2f\n",nth,tO*1e3,mrsO,mrsO/mrs1/nth);
  printf("[Newton 2阶 单核]           : %.2f M根/s   [Halley 3阶 单核]: %.2f M根/s   [HH3 4阶 单核]: %.2f M根/s\n",mrsN,mrsH,mrs1);
  printf("[Brent从头·编译版(脚注)]    : %.3f M根/s  成功=%ld/%d(会解到别的根)  fev/根=%.1f\n",mrsB,nok,N0,(double)fev/N0);
  printf("---- 诚实加速比(同 fp64 HH3 算法; GPU 实测 G3 fp64 394 / G4 df32 365 M根/s)----\n");
  printf("GPU满卡 vs CPU单核 = G3:%.0fx  G4:%.0fx  |  GPU满卡 vs CPU%d核 = G3:%.1fx  G4:%.1fx\n",
         394.0/mrs1, 365.0/mrs1, nth, 394.0/mrsO, 365.0/mrsO);
  printf("对比: 报告旧值 3917× = Python 循环调 scipy.brentq 的解释器开销(~0.09 M根/s), 非算法真实差距。\n");
  return 0;
}
