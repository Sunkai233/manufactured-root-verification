// diag_alpha.cpp — 审计 α 严格性: 认证失败(Newton 40步不收敛)是 γ 欠估(过度认证)还是越域?
// 对比 K=9 vs K=20 项的 γ; 统计认证翻转; 追踪失败实例是否越 ln 定义域。
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
// Newton, 返回步数; ec=1 越域(1+cx<=0), ec=2 非有限
int newton_steps(double x0,const Params<double>&p,int cap,double eps,int&ec){
  double x=x0; ec=0;
  for(int k=0;k<=cap;k++){
    if(1.0+p.c*x<=0){ ec=1; return cap+1; }
    double f,f1,f2,f3; derivs(x,p,f,f1,f2,f3);
    double s=step_newton(f,f1);
    if(!m_finite(s)){ ec=2; return cap+1; }
    if(std::fabs(s)<eps*std::max(std::fabs(x),1.0)) return k;
    x+=s;
  }
  return cap+1;
}
int main(int argc,char**argv){
  Data D=load(argv[1]); double thr=alpha_threshold(); const int cap=40; const double eps=1e-11;
  long c9=0,c20=0, flip=0, c9fail=0, fail_dom=0, fail_uncert20=0, fail_true=0, undere=0;
  int shown=0;
  for(int i=0;i<D.N;i++){
    Params<double> p{D.a[i],D.b[i],D.c[i],D.d[i],D.w[i],D.s0[i],D.s1[i]};
    double b=beta_at(D.x0[i],p);
    double g9=gamma_at(D.x0[i],p,9), g20=gamma_at(D.x0[i],p,20);
    double a9=b*g9, a20=b*g20;
    bool k9=a9<thr, k20=a20<thr;
    if(g20>g9*1.0000001) undere++;          // K=9 比 K=20 小 => K=9 欠估
    if(k9)c9++; if(k20)c20++;
    if(k9!=k20) flip++;
    if(k9){
      int ec; int kn=newton_steps(D.x0[i],p,cap,eps,ec);
      if(kn>cap){ c9fail++;
        if(ec==1||ec==2) fail_dom++;         // 越域/非有限
        else if(!k20) fail_uncert20++;       // 更多项后其实未认证 => γ 欠估致假认证
        else fail_true++;                    // 认证且更多项仍认证却不收敛 => 真异常
        if(shown<8){ shown++;
          printf("  失败认证 idx=%d kappa=%.2e beta=%.3e g9=%.3e g20=%.3e a9=%.4f a20=%.4f ec=%d kn=%d x0=%.4g\n",
                 i,D.gam[i],b,g9,g20,a9,a20,ec,kn,D.x0[i]); }
      }
    }
  }
  printf("N=%d thr=%.6f cap=%d\n",D.N,thr,cap);
  printf("认证 K=9: %ld | K=20: %ld | 翻转(K9≠K20): %ld | K9欠估(g20>g9): %ld (%.2f%%)\n",
         c9,c20,flip,undere,100.0*undere/D.N);
  printf("K=9 认证中 Newton 失败(>%d步): %ld\n",cap,c9fail);
  printf("  其中 越域/非有限: %ld | 更多项后变未认证(γ欠估致假认证): %ld | 更多项仍认证却不收敛(真异常): %ld\n",
         fail_dom,fail_uncert20,fail_true);
  return 0;
}
