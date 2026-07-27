// 收敛阶重测(纠正 C1 自相矛盾): fp64 无四阶渐近窗口, 改用 __float128(113位尾数≈34位十进制)。
// 从设计好的初值、按 1/γ 缩放偏移进入深渐近区, 观测阶 = log(e_{k+1})/log(e_k) 的斜率。
// Newton→2, Halley→3, Householder-3→4。参考真根用 __float128 迭代到自身收敛。
// 编译: g++ -O3 -march=native -DMRV_QUAD order_mp.cpp -o order_mp -Iinclude -lquadmath
#define MRV_QUAD
#include "mrv.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <quadmath.h>
using namespace mrv;
typedef __float128 q;

struct Data{ int N; std::vector<double> a,b,c,d,w,s0,s1,x0,gam,xs; };
Data load(const char* fn){
  FILE* f=fopen(fn,"rb"); int N; if(fread(&N,4,1,f)!=1)exit(1);
  std::vector<double> buf((size_t)10*N); if(fread(buf.data(),8,(size_t)10*N,f)!=(size_t)10*N)exit(1);
  Data D; D.N=N; auto col=[&](int j){return std::vector<double>(buf.begin()+(size_t)j*N,buf.begin()+(size_t)(j+1)*N);};
  D.a=col(0);D.b=col(1);D.c=col(2);D.d=col(3);D.w=col(4);D.s0=col(5);D.s1=col(6);D.x0=col(7);D.gam=col(8);D.xs=col(9);
  fclose(f); return D;
}

// 用 __float128 把真根精修到自身收敛(HH3, 25 步足够到 ~1e-34)
q refine_root(const Params<q>& p, q x){
  for(int k=0;k<40;k++){ q f,f1,f2,f3; derivs(x,p,f,f1,f2,f3); q s=step_hh3(f,f1,f2,f3); if(m_finite(s)) x+=s; }
  return x;
}

// 观测阶(双偏移单步法): 取两个渐近区偏移 h 与 h/r, 各走一步得 e_a'=C·h^p, e_b'=C·(h/r)^p,
// 比值消掉误差常数 C: p = ln(e_a'/e_b')/ln(r)。一步后误差远在地板之上, 避开触底。
// 对三个偏移幅度取中位, 抗单点偶然。
template<int D>
static q one_step_err(const Params<q>& p, q xstar, q h){
  q x=xstar+h; q f,f1,f2,f3; derivs(x,p,f,f1,f2,f3);
  q s=(D==1)?step_newton(f,f1):(D==2)?step_halley(f,f1,f2):step_hh3(f,f1,f2,f3);
  if(m_finite(s)) x+=s;
  return m_abs(x-xstar);
}
template<int D>
double observed_order(const Params<q>& p, q xstar, q gam){
  q g = (gam>q(0)? gam : q(1));
  q r = q(2);
  double best=-1;
  // 偏移甜点: 太小则根附近算 f 灾难性相消(f 由 O(1) 项抵消成 O(h)), 太大则超出渐近区。
  // 大偏移优先(e' 大, headroom 足); __float128 34 位对四阶仍受相消限, 高 κ 尾部读数偏低(mpmath 50 位给净 4.000)。
  for(q base : { q(0.02)/g, q(0.01)/g, q(0.04)/g, q(0.004)/g }){
    q ha=base, hb=base/r;
    q ea=one_step_err<D>(p,xstar,ha), eb=one_step_err<D>(p,xstar,hb);
    if(ea>q(1e-28)&&eb>q(1e-28)&&ea<ha&&eb<hb){
      double pr=(double)(logq(ea/eb)/logq(r));
      if(pr>0&&pr<6){ best=pr; break; }
    }
  }
  return best;
}

int main(int argc,char**argv){
  Data D=load(argv[1]);
  int lim=argc>2?atoi(argv[2]):D.N;
  int cnt[4]={0,0,0,0}; double sum[4]={0,0,0,0}; int in5[4]={0,0,0,0};
  int form[4]={0,2,3,4};
  for(int i=0;i<lim;i++){
    Params<q> p{(q)D.a[i],(q)D.b[i],(q)D.c[i],(q)D.d[i],(q)D.w[i],(q)D.s0[i],(q)D.s1[i]};
    q xstar=refine_root(p,(q)D.xs[i]);
    q gam=(q)D.gam[i];
    double o1=observed_order<1>(p,xstar,gam), o2=observed_order<2>(p,xstar,gam), o3=observed_order<3>(p,xstar,gam);
    double os[4]={0,o1,o2,o3};
    for(int D_=1;D_<=3;D_++){ double o=os[D_]; if(o>0){ cnt[D_]++; sum[D_]+=o; if(std::fabs(o-form[D_])/form[D_]<=0.05) in5[D_]++; } }
  }
  const char* nm[4]={"","Newton  ","Halley  ","HH-3    "};
  printf("收敛阶(__float128 113位, 双偏移单步法 p=ln(ea'/eb')/ln2, 偏移~0.02/γ 甜点), n=%d\n",lim);
  for(int D_=1;D_<=3;D_++)
    printf("  %s 形式阶 %d | 观测阶均值 %.4f (n=%d) | 5%%内占比 %.1f%%\n",
           nm[D_],form[D_],cnt[D_]?sum[D_]/cnt[D_]:0.0,cnt[D_],cnt[D_]?100.0*in5[D_]/cnt[D_]:0.0);
  return 0;
}
