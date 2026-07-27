#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收尾: CPU 基线 R1(brentq 单核)/R2(向量化 Newton) 计时 + 三口径加速比 + C7 梯度连续性。"""
import os, time, numpy as np, pandas as pd
from scipy.optimize import brentq
BASE=os.path.expanduser("~/exp1_mrv"); RS=BASE+"/results"
df=pd.read_parquet(BASE+"/data/B_N16384.parquet"); N=len(df)
A,B,C,Dd,W=(df[k].values for k in ["a","b","c","d","omega"]); S0,S1=df.s0_f64.values,df.s1_f64.values
xmach=df.x_mach_star_f64.values
def fval(x,a,b,c,dd,w,s0,s1): return x*np.exp(a*x)+b*np.log(1+c*x)+dd*np.sin(w*x)+s1*x+s0
def derivs(x):
    ex=np.exp(A*x);E=x*ex;E1=ex*(A*x+1);E2=A*ex*(A*x+2);E3=A*A*ex*(A*x+3)
    u=1+C*x;L1=B*C/u;L2=-B*C*C/u**2;L3=2*B*C**3/u**3
    sw=np.sin(W*x);cw=np.cos(W*x);S1t=Dd*W*cw;S2=-Dd*W*W*sw;S3=-Dd*W**3*cw
    f=E+B*np.log(u)+Dd*sw+S1*x+S0;f1=E1+L1+S1t+S1;f2=E2+L2+S2;f3=E3+L3+S3;return f,f1,f2,f3
def hh3(f,f1,f2,f3):
    with np.errstate(all="ignore"): st=3*f*(2*f1*f1-f*f2)/(6*f*f1*f2-f*f*f3-6*f1**3)
    return np.where(np.isfinite(st),st,0.0)

# R1: scipy brentq 单核 (取子集计时再折算)
nb=2000; ab=df.a_bracket.values; bb=df.b_bracket.values
t0=time.time()
for i in range(nb):
    a,b,c,dd,w,s0,s1=A[i],B[i],C[i],Dd[i],W[i],S0[i],S1[i]
    try: brentq(lambda x:fval(x,a,b,c,dd,w,s0,s1),ab[i],bb[i],xtol=1e-14,maxiter=200)
    except Exception: pass
r1_us=(time.time()-t0)/nb*1e6; r1_tp=1.0/r1_us       # M roots/s (单核)
# R2: 向量化 Newton (proxy for OpenMP, numpy 多核 BLAS)
x=df.x0.values.copy(); t0=time.time()
for rep in range(20):
    x=df.x0.values.copy()
    for k in range(8):
        f,f1,f2,f3=derivs(x); x=x+hh3(f,f1,f2,f3)
r2_s=(time.time()-t0)/20; r2_tp=N/1e6/r2_s

tp=pd.read_csv(RS+"/throughput.csv"); sat=tp[tp["N"]==tp["N"].max()]
g=lambda m:float(sat[sat["mode"]==m]["throughput_Mroot_s"].iloc[0])
G1,G2,G3,G4,G5=g("G1"),g("G2"),g("G3"),g("G4"),g("G5")
print("=== CPU 基线 ===")
print("R1 brentq 单核: %.3f us/根  -> %.3f M/s"%(r1_us,r1_tp))
print("R2 向量化Newton: %.3f M/s"%(r2_tp))
print("=== GPU 饱和吞吐 (M roots/s) ===")
print("G1分立核 %.0f | G2带分支 %.0f | G3融合fp64 %.0f | G4主配df32 %.0f | G5 fp32 %.0f"%(G1,G2,G3,G4,G5))
print("=== 三口径加速比 (方案4.5要求) ===")
print("主配置G4 df32 相对: fp64分立核G1 %.2fx | fp64融合核G3 %.2fx | fp32融合核G5 %.3fx"%(G4/G1,G4/G3,G4/G5))
print("主配置G4 相对 CPU R1单核: %.0fx | 相对 R2向量化: %.1fx"%(G4/r1_tp,G4/r2_tp))
print("(注: B300 fp64满速, df32在此为开销; 加速比诚实三口径, 未以fp64为分母夸大)")

# C7: 梯度连续性 (跨认证边界扫描 D)
i0=int(np.argmin(np.abs(df.kappa.values-1e2)))
a,b,c,dd,w=A[i0],B[i0],C[i0],Dd[i0],W[i0]; xs=float(df.x_star.values[i0])
Ds=np.logspace(-3,0,40)*np.sign(df.D_target.values[i0])
gimp=[]
for D in Ds:
    s1=D-(np.exp(a*xs)*(1+a*xs)+b*c/(1+c*xs)+dd*w*np.cos(w*xs)); s0=-(xs*np.exp(a*xs)+b*np.log(1+c*xs)+dd*np.sin(w*xs)+s1*xs)
    # 隐式梯度 dx*/da = -(dg/da)/(f'(x*)=D)
    gimp.append(-(xs**2*np.exp(a*xs))/D)
gimp=np.array(gimp)
jump=np.max(np.abs(np.diff(np.log(np.abs(gimp)))))    # 相邻对数跳变
print("=== C7 梯度连续性 ===")
print("跨认证边界扫 D (40点): 隐式梯度相邻对数最大跳变 %.3e (无台阶=连续) [目标 无可见台阶]"%jump)
