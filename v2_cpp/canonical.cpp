// canonical.cpp — 唯一真源(v3.1)。认证用 long double(修 β=0 相消)。
// ★步数许可: 停机容差改每实例自适应 ε_rel=max(1e-11, C·u·κ_res), κ_res=Ssum/(max(|x|,1)|f'|),
//   否则固定 1e-11 对 u·κ≥ε 的病态实例落在 fp64 地板下停不下来(假违反 k_max)。
// ★输出 moved: 8 步 HH3 是否真的移动过初值(高κ实例 off∝1/γ 太小→不动→污染 e_repr/C2)。
// 编译: g++ -O3 -march=native canonical.cpp -o canonical -Iinclude
#include "mrv.hpp"
#include "alpha.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
using namespace mrv;
struct Data{ int N; std::vector<double> a,b,c,d,w,s0,s1,x0,gam,xs; };
Data load(const char* fn){
  FILE* f=fopen(fn,"rb"); int N; if(fread(&N,4,1,f)!=1)exit(1);
  std::vector<double> buf((size_t)10*N); if(fread(buf.data(),8,(size_t)10*N,f)!=(size_t)10*N)exit(1);
  Data D; D.N=N; auto col=[&](int j){return std::vector<double>(buf.begin()+(size_t)j*N,buf.begin()+(size_t)(j+1)*N);};
  D.a=col(0);D.b=col(1);D.c=col(2);D.d=col(3);D.w=col(4);D.s0=col(5);D.s1=col(6);D.x0=col(7);D.gam=col(8);D.xs=col(9);
  fclose(f); return D;
}
inline double Ssum(double x,const Params<double>&p){
  double ex=std::exp(p.a*x);
  return std::fabs(x*ex)+std::fabs(p.b*std::log(std::fabs(1.0+p.c*x)+1e-300))
        +std::fabs(p.d*std::sin(p.w*x))+std::fabs(p.s1*x)+std::fabs(p.s0);
}
// 自适应容差步数; 返回步数(cap 内未停返回 cap+1)。moved=是否发生过非零步。
template<int Dord>
int steps_adap(double x0,const Params<double>&p,int cap,double C,int&moved){
  double x=x0; const double u=std::ldexp(1.0,-53); moved=0;
  for(int k=0;k<=cap;k++){
    double f,f1,f2,f3; derivs(x,p,f,f1,f2,f3);
    double s=(Dord==1)?step_newton(f,f1):(Dord==2)?step_halley(f,f1,f2):step_hh3(f,f1,f2,f3);
    if(!m_finite(s)) return cap+1;
    double sc=std::max(std::fabs(x),1.0);
    double kres=Ssum(x,p)/(sc*std::max(std::fabs(f1),1e-300));
    double tol_rel=std::max(1e-11, C*u*kres);
    if(std::fabs(s)/sc < tol_rel) return k;
    x+=s; if(s!=0.0) moved=1;
  }
  return cap+1;
}
int main(int argc,char**argv){
  Data D=load(argv[1]);
  const int cap=40; const double C=8.0; const double u=std::ldexp(1.0,-53);
  FILE* fo=fopen(argv[2],"w");
  fprintf(fo,"idx,gamma,alpha,cert,cert_d,kn,kh,kmax,compliant,moved\n");
  long double thrl=alpha_threshold_t<long double>(); double thrd=alpha_threshold_t<double>();
  long cert_l=0,cert_d=0, bd0=0, bl0=0, comp=0, nmoved=0;
  for(int i=0;i<D.N;i++){
    Params<double> p{D.a[i],D.b[i],D.c[i],D.d[i],D.w[i],D.s0[i],D.s1[i]};
    Params<long double> pl{(long double)D.a[i],(long double)D.b[i],(long double)D.c[i],(long double)D.d[i],(long double)D.w[i],(long double)D.s0[i],(long double)D.s1[i]};
    long double xl=(long double)D.x0[i];
    long double al=alpha_t<long double>(xl,pl); int cert=(al<thrl)?1:0;
    double ad=alpha_at(D.x0[i],p); int certd=(ad<thrd)?1:0;
    if(cert)cert_l++; if(certd)cert_d++;
    if(beta_at(D.x0[i],p)==0.0) bd0++;
    if(beta_t<long double>(xl,pl)==0.0L) bl0++;
    int mvn,mvh;
    int kn=steps_adap<1>(D.x0[i],p,cap,C,mvn);
    int kh=steps_adap<3>(D.x0[i],p,cap,C,mvh);
    // kmax 用同一自适应 ε(在 x0 处的 κ_res)
    double sc=std::max(std::fabs(D.x0[i]),1.0);
    double f,f1,f2,f3; derivs(D.x0[i],p,f,f1,f2,f3);
    double kres0=Ssum(D.x0[i],p)/(sc*std::max(std::fabs(f1),1e-300));
    double eps_i=std::max(1e-11, C*u*kres0);
    int km=kmax_newton((double)beta_t<long double>(xl,pl), eps_i);
    int compliant = (cert && kn<=km)?1:0;
    if(cert){ if(kn<=km) comp++; }
    if(mvh) nmoved++;
    fprintf(fo,"%d,%.6e,%.6e,%d,%d,%d,%d,%d,%d,%d\n",i,D.gam[i],(double)al,cert,certd,kn,kh,km,compliant,mvh);
  }
  fclose(fo);
  fprintf(stderr,"N=%d | 认证 LD %ld / double %ld | β=0: double %ld LD %ld | 认证合规(kn<=kmax) %ld/%ld=%.2f%% | HH3移动过 %ld\n",
          D.N,cert_l,cert_d,bd0,bl0,comp,cert_l,100.0*comp/std::max<long>(cert_l,1),nmoved);
  return 0;
}
