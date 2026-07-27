// stepdist.cpp — 认证价值的正确口径(纠正被写死的 warp-waste 0.000)。
// 度量: 从 x0 达 ε 相对精度所需迭代步数分布, 按 Smale α<0.157671 认证与否分组。
// α 理论预测: 认证点 → 步数退化成点质量(≤k_max, 方差≈0); 未认证 → 长尾(线性瞬态/发散)。
// GPU 含义: 认证集步数一致=固定步无分支=零线程束发散; 未认证集步数参差=发散。
// 编译: g++ -O3 -march=native stepdist.cpp -o stepdist -Iinclude
#include "mrv.hpp"
#include "alpha.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace mrv;

struct Data{ int N; std::vector<double> a,b,c,d,w,s0,s1,x0,gam,xs; };
Data load(const char* fn){
  FILE* f=fopen(fn,"rb"); int N; if(fread(&N,4,1,f)!=1)exit(1);
  std::vector<double> buf((size_t)10*N); if(fread(buf.data(),8,(size_t)10*N,f)!=(size_t)10*N)exit(1);
  Data D; D.N=N; auto col=[&](int j){return std::vector<double>(buf.begin()+(size_t)j*N,buf.begin()+(size_t)(j+1)*N);};
  D.a=col(0);D.b=col(1);D.c=col(2);D.d=col(3);D.w=col(4);D.s0=col(5);D.s1=col(6);D.x0=col(7);D.gam=col(8);D.xs=col(9);
  fclose(f); return D;
}

// 达收敛步数(步长判据: |迭代步|<ε·尺度, 与 α 理论/真求解器一致, 不依赖已知真根 → 避开多根歧义)。
// cap 内未收敛返回 cap+1(发散/长瞬态/吸引环)。
template<int Dord>
int steps_to_conv(double x0, const Params<double>& p, int cap, double eps){
  double x=x0;
  for(int k=0;k<=cap;k++){
    double f,f1,f2,f3; derivs(x,p,f,f1,f2,f3);
    double s=(Dord==1)?step_newton(f,f1):(Dord==2)?step_halley(f,f1,f2):step_hh3(f,f1,f2,f3);
    if(!m_finite(s)) return cap+1;
    if(std::fabs(s) < eps*std::max(std::fabs(x),1.0)) return k;
    x+=s;
  }
  return cap+1;
}
double pctl(std::vector<int> v,double q){ if(v.empty())return NAN; std::sort(v.begin(),v.end()); return v[(size_t)(q*(v.size()-1)+0.5)]; }
double mean(const std::vector<int>&v){ if(v.empty())return NAN; double s=0; for(int x:v)s+=x; return s/v.size(); }
// 线程束浪费(锁步): 每 32 路一束, 束内 = 1 - 平均步数/最大步数, 全束取平均。
// = 若用"逐实例早退核"(如 G2), 束内快车道等最慢车道的浪费占比。
double warp_waste(const std::vector<int>& s){
  double acc=0; long g=0;
  for(size_t i=0;i+32<=s.size();i+=32){
    int mx=0; double sum=0; for(int j=0;j<32;j++){ mx=std::max(mx,s[i+j]); sum+=s[i+j]; }
    if(mx>0){ acc += 1.0-(sum/32.0)/mx; g++; }
  }
  return g? acc/g : 0.0;
}

int main(int argc,char**argv){
  Data D=load(argv[1]);
  int M=argc>2?atoi(argv[2]):D.N; if(M>D.N)M=D.N;
  const int cap=30; const double eps=1e-11;
  std::vector<int> nC,nU;      // Newton 步数: 认证/未认证
  std::vector<int> hC,hU;      // HH3   步数: 认证/未认证
  long nfail_C=0,nfail_U=0; std::vector<int> kmaxs;
  int histN_C[35]={0},histN_U[35]={0};
  std::vector<int> allH; std::vector<int> allH_cert;   // HH3 步数(全体自然序 / 仅认证), 供 warp-waste
  for(int i=0;i<M;i++){
    Params<double> p{D.a[i],D.b[i],D.c[i],D.d[i],D.w[i],D.s0[i],D.s1[i]};
    double x0=D.x0[i], xs=D.xs[i];
    (void)xs;
    double a=alpha_at(x0,p); bool cert=(a<alpha_threshold());
    int kn=steps_to_conv<1>(x0,p,cap,eps);
    int kh=steps_to_conv<3>(x0,p,cap,eps);
    allH.push_back(kh); if(cert) allH_cert.push_back(kh);
    if(cert){ nC.push_back(kn); hC.push_back(kh); if(kn>cap)nfail_C++; histN_C[std::min(kn,34)]++;
              kmaxs.push_back(kmax_newton(beta_at(x0,p),eps)); }
    else    { nU.push_back(kn); hU.push_back(kh); if(kn>cap)nfail_U++; histN_U[std::min(kn,34)]++; }
  }
  // 真实 warp-waste(HH3 步数): 自然序(≈随机排布) / 按难度排序 / 仅认证子集 / 固定步(定义值0)
  std::vector<int> sortedH=allH; std::sort(sortedH.begin(),sortedH.end());
  double wShuf=warp_waste(allH), wSort=warp_waste(sortedH), wCert=warp_waste(allH_cert);
  printf("步数分布(达 ε=1e-11, cap=%d)  n=%d  认证 %zu / 未认证 %zu  (阈值 α<%.6f)\n",
         cap,M,nC.size(),nU.size(),alpha_threshold());
  printf("Newton 步数  认证: 均值 %.2f | p50 %.0f | p90 %.0f | p99 %.0f | 失败(>cap) %ld (%.2f%%)\n",
         mean(nC),pctl(nC,.5),pctl(nC,.9),pctl(nC,.99),nfail_C,100.0*nfail_C/std::max<size_t>(nC.size(),1));
  printf("Newton 步数 未认证: 均值 %.2f | p50 %.0f | p90 %.0f | p99 %.0f | 失败(>cap) %ld (%.2f%%)\n",
         mean(nU),pctl(nU,.5),pctl(nU,.9),pctl(nU,.99),nfail_U,100.0*nfail_U/std::max<size_t>(nU.size(),1));
  printf("HH3    步数  认证: 均值 %.2f | p99 %.0f     未认证: 均值 %.2f | p99 %.0f\n",
         mean(hC),pctl(hC,.99),mean(hU),pctl(hU,.99));
  printf("α 理论 k_max(Newton) 认证集: 均值 %.2f | max %.0f  (实测 Newton p99=%.0f 应≤此)\n",
         mean(kmaxs),pctl(kmaxs,1.0),pctl(nC,.99));
  printf("真实 warp-waste(HH3 实测步数, 若用早退核): 随机排布 %.3f → 按难度排序 %.3f → 仅认证 %.3f → 固定步(无早退,定义值) 0.000\n",
         wShuf,wSort,wCert);
  // 导出直方图(供画图: 步数 0..20 的认证/未认证计数)
  FILE* fo=fopen("results_stepdist.csv","w"); fprintf(fo,"steps,certified,uncertified\n");
  for(int k=0;k<=20;k++) fprintf(fo,"%d,%d,%d\n",k,histN_C[k],histN_U[k]);
  fclose(fo);
  printf("-> results_stepdist.csv (Newton 步数直方图, 0..20)\n");
  return 0;
}
