#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唯一真源合并器 v3: parquet(kappa,x*,x0) + 新鲜 xhat(唯一 e_repr) + canonical.cpp(LD认证/自适应ε步数/moved) -> instances.csv。
所有图/criteria 从它派生。新增: 构造偏置(按κ分带的不动比例)、e_repr/C2 分全体 vs 求解器移动过子集、尾部收敛到别根、多键 warp-waste。"""
import numpy as np, pandas as pd, os, sys
B=os.path.expanduser("~/exp1_mrv")
# 可参数化多数据集: argv = parquet, xhat前缀(""或"floor_"), 输出csv, alpha_data
PARQUET=sys.argv[1] if len(sys.argv)>1 else B+"/data/B_N16384.parquet"
XT     =sys.argv[2] if len(sys.argv)>2 else ""
OUT    =sys.argv[3] if len(sys.argv)>3 else B+"/v2_cpp/instances.csv"
ALPHA  =sys.argv[4] if len(sys.argv)>4 else B+"/v2_cpp/alpha_data.csv"
STATS  =OUT.replace("instances","canonical_stats").replace(".csv",".txt")
df=pd.read_parquet(PARQUET)
kap=df.kappa.values.astype(float); xs=df.x_mach_star_f64.values.astype(float); x0=df.x0.values.astype(float)
al=pd.read_csv(ALPHA); N=len(kap); assert len(al)==N
e={}
for m in ["G3","G4","G5"]:
    xh=np.fromfile(B+"/results/xhat_%s%s.bin"%(XT,m))[:N]; e[m]=np.abs(xh-xs)/np.maximum(np.abs(xs),1.0)
xhG3=np.fromfile(B+"/results/xhat_%sG3.bin"%XT)[:N]
moved_G3=(np.abs(xhG3-x0)>0.0)            # CUDA G3 是否真移动过初值
init_rel=np.abs(x0-xs)/np.maximum(np.abs(xs),1.0)
improved_G3=(e["G3"]<init_rel*(1-1e-9))  # 更强: 8步是否把初值改进了
kap.astype(np.float64).tofile(B+"/v2_cpp/kappa.bin")   # 供 multiroot 用 κ
gen_cert=df.certified.values.astype(int) if "certified" in df.columns else np.full(N,-1)
inst=pd.DataFrame({"idx":np.arange(N),"kappa":kap,"gamma":al.gamma.values,"alpha":al.alpha.values,
  "cert":al.cert.values.astype(int),"cert_d":al.cert_d.values.astype(int),"cert_gen":gen_cert,
  "compliant":al.compliant.values.astype(int),"moved":al.moved.values.astype(int),"moved_G3":moved_G3.astype(int),
  "e_G3":e["G3"],"e_G4":e["G4"],"e_G5":e["G5"],"kn":al.kn.values,"kh":al.kh.values,"kmax":al.kmax.values})
inst.to_csv(OUT,index=False)

cert=inst.cert.values.astype(bool); u=2.0**-53; L=[]
L.append("== 唯一真源 instances.csv (N=%d, u=2^-53, 自适应ε=max(1e-11,8uκ)) =="%N)
L.append("认证对账: long double(主用) %d (%.2f%%) | double(β=0 bug) %d | gen mpmath %s"
         %(cert.sum(),100*cert.mean(),int(inst.cert_d.sum()),(str(int((gen_cert==1).sum())) if (gen_cert>=0).any() else "NA")))
comp=inst.compliant.values[cert]
L.append("★步数许可 kn<=kmax(认证子集, 自适应ε): %.2f%% (%d/%d)"%(100*comp.mean(),comp.sum(),cert.sum()))

bands=[(1,1e4),(1e4,1e6),(1e6,1e8),(1e8,1e10),(1e10,1e12),(1e12,1e99)]
def band_mask(lo,hi): return (kap>=lo)&(kap<hi)
L.append("★构造偏置: 各 κ 带 CUDA-G3 未移动初值的比例(off∝1/γ 致高κ不动):")
for lo,hi in bands:
    m=band_mask(lo,hi)
    if m.sum(): L.append("   κ[%.0e,%.0e) n=%d 未移动 %.1f%%"%(lo,hi,m.sum(),100*(~moved_G3[m]).mean()))
L.append("全体未移动 %.1f%% | 未改进初值 %.1f%% | 认证子集未改进 %.1f%%"
         %(100*(~moved_G3).mean(),100*(~improved_G3).mean(),100*(~improved_G3[cert]).mean()))
L.append("   各带未改进比例: "+" ".join("%.0f%%"%(100*(~improved_G3[band_mask(lo,hi)]).mean()) for lo,hi in bands))

L.append("-- e_repr 认证子集 [唯一来源]: 全体 vs 仅求解器移动过 --")
cm=cert&moved_G3
for m in ["G3","G4","G5"]:
    ea=inst["e_"+m].values[cert]; eb=inst["e_"+m].values[cm]
    L.append("   %s 全认证 p50 %.3e p99 %.3e | 移动过认证(n=%d) p50 %.3e p99 %.3e"
             %(m,np.percentile(ea,50),np.percentile(ea,99),cm.sum(),np.percentile(eb,50),np.percentile(eb,99)))
L.append("-- C2 误差≤4κu (fp64) 全体 vs 移动过, 及按 κ 分带 --")
c2=lambda mask: 100*(inst.e_G3.values[mask]<=4*kap[mask]*u).mean() if mask.sum() else float('nan')
L.append("   全体 %.2f%% | 求解器移动过 %.2f%%"%(c2(np.isfinite(inst.e_G3.values)),c2(moved_G3)))
for lo,hi in bands:
    m=band_mask(lo,hi)
    if m.sum(): L.append("   κ[%.0e,%.0e) C2 全 %.2f%% / 移动过 %.2f%%(n移%d)"%(lo,hi,c2(m),c2(m&moved_G3),(m&moved_G3).sum()))

L.append("-- 尾部: 认证集 e_G3>1e-6 是否收敛到别的根(非精度地板) --")
big=cert&(inst.e_G3.values>1e-6); r=inst.e_G4.values[big]/np.maximum(inst.e_G3.values[big],1e-300)
if big.sum(): L.append("   n=%d, e_G4/e_G3 p10/p50/p90 = %.3f/%.3f/%.3f (≈1=两精度收敛到同一错根,非地板)"
                       %(big.sum(),np.percentile(r,10),np.percentile(r,50),np.percentile(r,90)))

def ww(order,ns=8):
    s=np.minimum(inst.kh.values[order],ns); m=len(s)//32*32; s=s[:m].reshape(-1,32)
    mx=s.max(1); mn=s.mean(1); return float(np.mean(1-mn/np.maximum(mx,1)))
L.append("-- warp-waste(HH3步,封顶ns=8) 多调度键 --")
L.append("   随机 %.3f | κ排序 %.3f | α排序 %.3f | 认证标志排序 %.3f | 步数oracle %.3f | 固定步 0.000"
         %(ww(np.arange(N)),ww(np.argsort(kap)),ww(np.argsort(inst.alpha.values)),
           ww(np.argsort(inst.cert.values)),ww(np.argsort(inst.kh.values))))
txt="\n".join(L); open(STATS,"w").write(txt+"\n"); print(txt)
