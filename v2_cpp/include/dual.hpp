// dual.hpp — 纯 C++ 前向自动微分(Ceres Jet 式), 携带值 + 5 个偏导(θ=a,b,c,d,ω)。
// 定义在 namespace mrv, 提供 m_exp/m_log/m_sin/m_cos/m_abs/m_finite 重载,
// 从而 mrv::derivs<Dual5> 与 solve_fixed<Dual5,D> 直接可用 —— unrolled AD 路径。
#pragma once
#include <cmath>

namespace mrv {

struct Dual5 {
  double v; double d[5];
  Dual5(): v(0){ for(int i=0;i<5;i++) d[i]=0; }
  Dual5(double val): v(val){ for(int i=0;i<5;i++) d[i]=0; }   // 常数: 导数=0
};

inline Dual5 dual_var(double val,int idx){ Dual5 r(val); r.d[idx]=1.0; return r; } // 独立变量: e_idx

inline Dual5 operator+(const Dual5&a,const Dual5&b){ Dual5 r; r.v=a.v+b.v; for(int i=0;i<5;i++) r.d[i]=a.d[i]+b.d[i]; return r; }
inline Dual5 operator-(const Dual5&a,const Dual5&b){ Dual5 r; r.v=a.v-b.v; for(int i=0;i<5;i++) r.d[i]=a.d[i]-b.d[i]; return r; }
inline Dual5 operator-(const Dual5&a){ Dual5 r; r.v=-a.v; for(int i=0;i<5;i++) r.d[i]=-a.d[i]; return r; }
inline Dual5 operator*(const Dual5&a,const Dual5&b){ Dual5 r; r.v=a.v*b.v; for(int i=0;i<5;i++) r.d[i]=a.v*b.d[i]+b.v*a.d[i]; return r; } // 乘积法则
inline Dual5 operator/(const Dual5&a,const Dual5&b){ Dual5 r; r.v=a.v/b.v; double ib=1.0/b.v; for(int i=0;i<5;i++) r.d[i]=(a.d[i]-r.v*b.d[i])*ib; return r; } // 商法则
inline Dual5& operator+=(Dual5&a,const Dual5&b){ a=a+b; return a; }

// 数学函数(链式法则写死每个 op)
inline Dual5 m_exp(const Dual5&a){ double e=std::exp(a.v); Dual5 r; r.v=e; for(int i=0;i<5;i++) r.d[i]=e*a.d[i]; return r; }
inline Dual5 m_log(const Dual5&a){ Dual5 r; r.v=std::log(a.v); double ia=1.0/a.v; for(int i=0;i<5;i++) r.d[i]=a.d[i]*ia; return r; }
inline Dual5 m_sin(const Dual5&a){ double c=std::cos(a.v); Dual5 r; r.v=std::sin(a.v); for(int i=0;i<5;i++) r.d[i]=c*a.d[i]; return r; }
inline Dual5 m_cos(const Dual5&a){ double s=std::sin(a.v); Dual5 r; r.v=std::cos(a.v); for(int i=0;i<5;i++) r.d[i]=-s*a.d[i]; return r; }
inline Dual5 m_abs(const Dual5&a){ Dual5 r; r.v=std::fabs(a.v); double sg=(a.v<0?-1.0:1.0); for(int i=0;i<5;i++) r.d[i]=sg*a.d[i]; return r; }
inline bool  m_finite(const Dual5&a){ return std::isfinite(a.v); }

} // namespace mrv
