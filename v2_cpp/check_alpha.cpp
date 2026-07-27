// 认证精度稳健性: alpha_t<double> vs alpha_t<long double>(同逻辑, 仅精度差)。
// 翻转实例打印 β/γ 双vs长双, 定位 β 近根相消 or γ。
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
int main(int argc,char**argv){
  Data D=load(argv[1]);
  double thrd=alpha_threshold_t<double>(); long double thrl=alpha_threshold_t<long double>();
  long cd=0,cl=0,flip=0,cx=0; int shown=0;
  for(int i=0;i<D.N;i++){
    Params<double> pd{D.a[i],D.b[i],D.c[i],D.d[i],D.w[i],D.s0[i],D.s1[i]};
    Params<long double> pl{(long double)D.a[i],(long double)D.b[i],(long double)D.c[i],(long double)D.d[i],(long double)D.w[i],(long double)D.s0[i],(long double)D.s1[i]};
    double xd=D.x0[i]; long double xl=(long double)D.x0[i];
    double bd=beta_t<double>(xd,pd), gd=gamma_t<double>(xd,pd), ad=bd*gd;
    long double bl=beta_t<long double>(xl,pl), gl=gamma_t<long double>(xl,pl), al=bl*gl;
    bool kd=ad<thrd, kl=al<thrl;
    if(kd)cd++; if(kl)cl++;
    // gen 口径: β(x0)·γ(xstar) 在 long double
    long double gx=gamma_t<long double>((long double)D.xs[i],pl), ax=bl*gx;
    if(ax<thrl)cx++;
    if(kd!=kl){ flip++;
      if(shown<8){ shown++;
        printf("  翻转 idx=%d x0=%.6g | βd=%.4e βl=%.4e (Δ%.1e) | γd=%.4e γl=%.4e | αd=%.5f αl=%.5f | kd=%d kl=%d\n",
               i,D.x0[i],bd,(double)bl,std::fabs(bd-(double)bl)/((double)bl+1e-300),gd,(double)gl,ad,(double)al,kd,kl); }
    }
  }
  printf("N=%d  认证: double γ(x0):%ld | long double γ(x0)[正确]:%ld | long double γ(xstar)[gen口径]:%ld | double翻转:%ld\n",
         D.N,cd,cl,cx,flip);
  return 0;
}
