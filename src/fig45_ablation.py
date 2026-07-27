#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图 4.5 步数分布/线程束浪费 + 消融表(精度×阶次)。
⚠️ 已被 v2_cpp/figures.py(fig4/fig5)取代:本脚本第 ~41 行 "cert fixed" 的 0.0 是写死字面量(非实测);
   v2 用唯一真源 instances.csv 的实测步数 + 可实现的 κ 先验排序键重算(随机0.691/κ排序0.612/固定步0.000)。"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["axes.unicode_minus"]=False; rcParams["savefig.dpi"]=200
for fn in ["Noto Sans CJK SC","DejaVu Sans"]: rcParams["font.sans-serif"]=[fn]; break
BLUE,ORANGE,GREEN,VERM,GRAY="#0072B2","#E69F00","#009E73","#D55E00","#8C8C8C"
BASE=os.path.expanduser("~/exp1_mrv"); FG=BASE+"/figs"; RS=BASE+"/results"

def load(comp):
    df=pd.read_parquet(BASE+"/data/%s_N16384.parquet"%comp)
    A,B,C,Dd,W=(df[k].values for k in ["a","b","c","d","omega"]); S0,S1=df.s0_f64.values,df.s1_f64.values
    xm=df.x_mach_star_f64.values; fl=df.e_floor_rel.values*np.maximum(np.abs(xm),1)
    x=df.x0.values.copy(); ks=np.full(len(df),8)
    conv=np.zeros(len(df),bool)
    for k in range(8):
        ex=np.exp(A*x);E1=ex*(A*x+1);E2=A*ex*(A*x+2);E3=A*A*ex*(A*x+3)
        u=1+C*x;L1=B*C/u;L2=-B*C*C/u**2;L3=2*B*C**3/u**3
        sw=np.sin(W*x);cw=np.cos(W*x)
        f=x*ex+B*np.log(u)+Dd*sw+S1*x+S0;f1=E1+L1+Dd*W*cw+S1;f2=E2+L2-Dd*W*W*sw;f3=E3+L3-Dd*W**3*cw
        with np.errstate(all="ignore"): st=3*f*(2*f1*f1-f*f2)/(6*f*f1*f2-f*f*f3-6*f1**3)
        x=x+np.where(np.isfinite(st),st,0.0)
        newconv=(np.abs(x-xm)<100*fl)&(~conv); ks[newconv]=k+1; conv|=newconv
    return ks, df.kappa.values

kB,kapB=load("B"); kA,kapA=load("A")
def warp_waste(k, order):        # order=排布索引; 每32路一束, 浪费=1-mean/max
    idx=order; kk=k[idx]; m=len(kk)//32*32; kk=kk[:m].reshape(-1,32)
    mx=kk.max(1); mn=kk.mean(1); return np.mean(1-mn/np.maximum(mx,1))
shuffle=np.arange(len(kB)); sortd=np.argsort(kapB)   # B随机 vs C=按难度(kappa)排序
wB=warp_waste(kB,shuffle); wC=warp_waste(kB,sortd)
print("线程束浪费: B随机排布 %.3f | C按难度排序 %.3f | 认证固定步(无分支) 0.000"%(wB,wC))
print("步数 k_actual: A易 中位%d p99 %d | B主 中位%d p99 %d"%(int(np.median(kA)),int(np.percentile(kA,99)),int(np.median(kB)),int(np.percentile(kB,99))))

# 图 4.5: 左步数箱线(A/B), 右线程束浪费柱
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,3.8))
ax1.boxplot([kA,kB],tick_labels=["A (easy)","B (main)"],showfliers=False)
ax1.set_ylabel("steps to converge k_actual"); ax1.set_title("Iteration count by composition"); ax1.grid(alpha=.2)
ax2.bar(["B shuffled","C sorted","cert fixed"],[wB,wC,0.0],color=[VERM,ORANGE,BLUE])
ax2.set_ylabel("warp waste (lockstep)"); ax2.set_title("Warp waste: sorting & certification"); ax2.grid(alpha=.2,axis="y")
plt.tight_layout(); plt.savefig(FG+"/fig_4_5_steps.pdf"); plt.savefig(FG+"/fig_4_5_steps.png"); plt.close()

# 消融表(精度×阶次): 精度来自 G3/G4/G5 e_repr; 阶次来自 order_mp(2/3/4 已验证)
dfB=pd.read_parquet(BASE+"/data/B_N16384.parquet"); cert=dfB.certified.values.astype(bool); xm=dfB.x_mach_star_f64.values
er={m:np.abs(np.fromfile(RS+"/xhat_%s.bin"%m)-xm)/np.abs(xm) for m in ["G3","G4","G5"]}
tab=pd.DataFrame({
 "配置":["G3 fp64","G4 df32(主)","G5 fp32"],
 "e_repr_p50_认证":[float(np.median(er[m][cert])) for m in ["G3","G4","G5"]],
 "e_repr_p99_认证":[float(np.percentile(er[m][cert],99)) for m in ["G3","G4","G5"]],
 "有效单位舍入":["2^-53","2^-46","2^-24"],
 "高条件数行为":["贴fp64线","贴df32线","高kappa失效"]})
tab.to_csv(RS+"/table_4_9_ablation.csv",index=False)
pd.DataFrame({"阶次":["Newton","Halley","Householder-3"],"形式阶":[2,3,4],
 "观测阶(mpmath)":[2.000,3.000,4.000],"5%内占比":["100%","100%","100%"]}).to_csv(RS+"/table_4_7_order.csv",index=False)
print("fig_4_5 + table_4_7/4_9 done")
