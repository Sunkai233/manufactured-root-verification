// alpha.hpp — Smale α 认证(纯 C++, 模板化 T)。阈值 (13−3√17)/4≈0.157671。
// α=βγ; β=|f/f'|; γ=sup_{k≥2}|f^{(k)}/(k! f')|^{1/(k-1)}(有限项 + ln 奇点尾 1/|x+1/c|)。
// 模板化以便在 double/long double 下核对认证的精度稳健性(β 近根相消)。
#pragma once
#include "mrv.hpp"
#include <cmath>

namespace mrv {

template<typename T> inline T alpha_threshold_t(){ return (T(13)-T(3)*std::sqrt(T(17)))/T(4); }
inline double alpha_threshold(){ return alpha_threshold_t<double>(); }  // 0.15767078...

// f^{(k)}(x), k≥2: E=a^{k-1}e^{ax}(ax+k); L=b(-1)^{k-1}(k-1)!c^k/(1+cx)^k; S=d w^k sin(wx+kπ/2)
template<typename T> inline T fk_t(int k, T x, const Params<T>& p){
  const T PI=T(3.14159265358979323846264338327950288L);
  T ex=m_exp(p.a*x);
  T E=std::pow(p.a,k-1)*ex*(p.a*x+T(k));
  T u=T(1)+p.c*x;
  T L=p.b*((k%2==1)?T(1):T(-1))*std::tgamma(T(k))*std::pow(p.c,k)/std::pow(u,k);  // tgamma(k)=(k-1)!
  T S=p.d*std::pow(p.w,k)*m_sin(p.w*x+T(k)*PI/T(2));
  return E+L+S;
}
template<typename T> inline T gamma_t(T x, const Params<T>& p, int K=9){
  T f,f1,f2,f3; derivs(x,p,f,f1,f2,f3);
  T af1=m_abs(f1); if(af1<T(1e-300)) return T(1e300);
  T g=T(0);
  for(int k=2;k<=K;k++){
    T num=m_abs(fk_t<T>(k,x,p));
    T kfact=std::tgamma(T(k)+T(1));                 // k!
    T term=std::pow(num/(kfact*af1), T(1)/T(k-1));
    if(m_finite(term)) g=std::max(g,term);
  }
  if(p.c!=T(0)){ T dist=m_abs(x+T(1)/p.c); if(dist>T(0)) g=std::max(g, T(1)/dist); }
  return g;
}
template<typename T> inline T beta_t(T x, const Params<T>& p){
  T f,f1,f2,f3; derivs(x,p,f,f1,f2,f3); return m_abs(f/f1);
}
template<typename T> inline T alpha_t(T x, const Params<T>& p){ return beta_t<T>(x,p)*gamma_t<T>(x,p); }
template<typename T> inline bool certified_t(T x, const Params<T>& p){ return alpha_t<T>(x,p)<alpha_threshold_t<T>(); }

// double 别名(向后兼容)
inline double fk(int k,double x,const Params<double>&p){ return fk_t<double>(k,x,p); }
inline double gamma_at(double x,const Params<double>&p,int K=9){ return gamma_t<double>(x,p,K); }
inline double beta_at(double x,const Params<double>&p){ return beta_t<double>(x,p); }
inline double alpha_at(double x,const Params<double>&p){ return alpha_t<double>(x,p); }
inline bool   certified(double x,const Params<double>&p){ return certified_t<double>(x,p); }
inline int kmax_newton(double beta,double eps){
  double v=2.0+std::log2(std::max(beta/eps,1.0+1e-300));
  return (int)std::ceil(std::log2(std::max(v,1.0+1e-300)));
}

} // namespace mrv
