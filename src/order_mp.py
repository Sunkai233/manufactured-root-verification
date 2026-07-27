#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1 观测阶验证 (mpmath 50 位, 去浮点地板)。方法在精确算术里收敛到精确制造根,
50 个数量级足够看清高阶渐近率。用存的 50 位 s0/s1。"""
import sys, numpy as np, pandas as pd, mpmath as mp
mp.mp.dps = 50
df = pd.read_parquet(sys.argv[1] if len(sys.argv)>1 else __import__("os").path.expanduser("~/exp1_mrv/data/B_N16384.parquet"))
sub = df[df.kappa < 1e3].head(80).reset_index(drop=True)

def make(row):
    a,b,c,dd,w = map(mp.mpf,(row.a,row.b,row.c,row.d,row.omega))
    s0,s1 = mp.mpf(row.s0), mp.mpf(row.s1); xs = mp.mpf(row.x_star)
    gam = mp.mpf(row.gamma_ref)
    def f(x):  return x*mp.e**(a*x)+b*mp.log(1+c*x)+dd*mp.sin(w*x)+s1*x+s0
    def d1(x): return mp.e**(a*x)*(1+a*x)+b*c/(1+c*x)+dd*w*mp.cos(w*x)+s1
    def d2(x): return mp.e**(a*x)*(a*(2+a*x))-b*c*c/(1+c*x)**2-dd*w*w*mp.sin(w*x)
    def d3(x): return mp.e**(a*x)*(a*a*(3+a*x))+2*b*c**3/(1+c*x)**3-dd*w**3*mp.cos(w*x)
    return f,d1,d2,d3,xs,gam

def step(order,f,f1,f2,f3):
    if order==1: return -f/f1
    if order==2: return -2*f*f1/(2*f1*f1-f*f2)
    return 3*f*(2*f1*f1-f*f2)/(6*f*f1*f2-f*f*f3-6*f1**3)

print("C1 观测阶 (mpmath 50位, kappa<1e3, x0=x*(1+0.5), n=%d):"%len(sub))
for order,name,formal in [(1,"Newton",2),(2,"Halley",3),(3,"Householder-3",4)]:
    ords=[]
    for _,row in sub.iterrows():
        f,f1,f2,f3,xs,gam = make(row)
        x = xs + mp.mpf("0.01")/gam; e=[mp.fabs(x-xs)]   # 偏移按 1/gamma 缩放 -> 深渐近区
        for k in range(7):
            x = x + step(order,f(x),f1(x),f2(x),f3(x)); e.append(mp.fabs(x-xs))
        # 首个 三点都>1e-40 且严格递减 的三元组
        for k in range(len(e)-2):
            if e[k]>mp.mpf("1e-40") and e[k+1]>mp.mpf("1e-40") and e[k+2]>mp.mpf("1e-40") and e[k]>e[k+1]>e[k+2]>0:
                ords.append(float(mp.log(e[k+2]/e[k+1])/mp.log(e[k+1]/e[k]))); break
    ords=np.array(ords)
    within=np.mean(np.abs(ords-formal)/formal<0.05)
    print("  [%-13s] 形式阶=%d  观测阶 中位 %.3f  均 %.3f (n=%d)  |偏差<5%%占比 %.1f%%"%
          (name,formal,np.median(ords),np.mean(ords),len(ords),100*within))
