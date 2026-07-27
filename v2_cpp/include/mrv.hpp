// mrv.hpp — 制造根核心(纯 C++, 模板化 T=double/__float128/Dual)。
// 制造族 g(x)=x·e^{ax}+b·ln(1+cx)+d·sin(wx); 残差 f=g+s1·x+s0。
// 定义 MRV_QUAD 启用 __float128(需 -lquadmath)。
#pragma once
#include <cmath>
#ifdef MRV_QUAD
#include <quadmath.h>
#endif

namespace mrv {

// —— 数学函数分派(double / __float128) ——
inline double m_exp(double x){return std::exp(x);}
inline double m_log(double x){return std::log(x);}
inline double m_sin(double x){return std::sin(x);}
inline double m_cos(double x){return std::cos(x);}
inline double m_abs(double x){return std::fabs(x);}
inline bool   m_finite(double x){return std::isfinite(x);}

// long double 重载(否则 derivs<long double> 会退化成 double 精度)
inline long double m_exp(long double x){return std::exp(x);}
inline long double m_log(long double x){return std::log(x);}
inline long double m_sin(long double x){return std::sin(x);}
inline long double m_cos(long double x){return std::cos(x);}
inline long double m_abs(long double x){return std::fabs(x);}
inline bool        m_finite(long double x){return std::isfinite(x);}

#ifdef MRV_QUAD
inline __float128 m_exp(__float128 x){return expq(x);}
inline __float128 m_log(__float128 x){return logq(x);}
inline __float128 m_sin(__float128 x){return sinq(x);}
inline __float128 m_cos(__float128 x){return cosq(x);}
inline __float128 m_abs(__float128 x){return fabsq(x);}
inline bool       m_finite(__float128 x){return finiteq(x)!=0;}
#endif

template<typename T> struct Params { T a,b,c,d,w,s0,s1; };

// 残差(仅值), 供 Brent
template<typename T> inline T resid(T x, const Params<T>& p){
  return x*m_exp(p.a*x) + p.b*m_log(T(1)+p.c*x) + p.d*m_sin(p.w*x) + p.s1*x + p.s0;
}

// 残差 + 1..3 阶导(与 solver_timed.cu 的 dv 一致)
template<typename T>
inline void derivs(T x, const Params<T>& p, T& f, T& f1, T& f2, T& f3){
  T ex=m_exp(p.a*x);
  T E=x*ex, E1=ex*(p.a*x+T(1)), E2=p.a*ex*(p.a*x+T(2)), E3=p.a*p.a*ex*(p.a*x+T(3));
  T u=T(1)+p.c*x, bc=p.b*p.c;
  T L1=bc/u, L2=-bc*p.c/(u*u), L3=T(2)*bc*p.c*p.c/(u*u*u), L=p.b*m_log(u);
  T sw=m_sin(p.w*x), cw=m_cos(p.w*x);
  T S=p.d*sw, S1=p.d*p.w*cw, S2=-p.d*p.w*p.w*sw, S3=-p.d*p.w*p.w*p.w*cw;
  f=E+L+S+p.s1*x+p.s0; f1=E1+L1+S1+p.s1; f2=E2+L2+S2; f3=E3+L3+S3;
}

// 迭代步(加法约定 x += step); D=1 Newton(2阶)/2 Halley(3阶)/3 Householder-3(4阶)
template<typename T> inline T step_newton(T f,T f1){ return -f/f1; }
template<typename T> inline T step_halley(T f,T f1,T f2){ return -T(2)*f*f1/(T(2)*f1*f1-f*f2); }
template<typename T> inline T step_hh3(T f,T f1,T f2,T f3){
  T num=T(3)*f*(T(2)*f1*f1-f*f2);
  T den=T(6)*f*f1*f2 - f*f*f3 - T(6)*f1*f1*f1;
  return num/den;   // f 在分子, 收敛后自然趋零, 免除零
}

// 固定步数迭代(与 GPU G3 同算法); D 决定阶次
template<typename T, int D>
inline T solve_fixed(T x, const Params<T>& p, int nsteps){
  for(int k=0;k<nsteps;k++){
    T f,f1,f2,f3; derivs(x,p,f,f1,f2,f3);
    T s = (D==1)? step_newton(f,f1) : (D==2)? step_halley(f,f1,f2) : step_hh3(f,f1,f2,f3);
    if(m_finite(s)) x += s;
  }
  return x;
}

} // namespace mrv
