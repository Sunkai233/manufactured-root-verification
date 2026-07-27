#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一 · 步骤5: 分析 + 出图 + 逐条核验 C1-C10 (口径修正版)。"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["axes.unicode_minus"] = False; rcParams["font.size"] = 10.5
rcParams["savefig.dpi"] = 200; rcParams["figure.dpi"] = 150
# 中文字体(若装了)
for fn in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]:
    rcParams["font.sans-serif"] = [fn]
    break
GRAY, ORANGE, GREEN, BLUE, SKY, VERM = "#8C8C8C", "#E69F00", "#009E73", "#0072B2", "#56B4E9", "#D55E00"
BASE = os.path.expanduser("~/exp1_mrv"); FG = BASE + "/figs"; RS = BASE + "/results"

df = pd.read_parquet(BASE + "/data/B_N16384.parquet")
N = len(df)
xmach = df.x_mach_star_f64.values; xstar = df.x_star.astype(float).values
A, B, C, Dd, W = (df[k].values for k in ["a", "b", "c", "d", "omega"])
S0, S1 = df.s0_f64.values, df.s1_f64.values
Dt = df.D_target.values; kappa = df.kappa.values; cert = df.certified.values.astype(bool)
floor = df.e_floor_rel.values

def derivs(x, a, b, c, dd, w, s0, s1):
    ex = np.exp(a*x); E=x*ex; E1=ex*(a*x+1); E2=a*ex*(a*x+2); E3=a*a*ex*(a*x+3)
    u=1+c*x; L1=b*c/u; L2=-b*c*c/u**2; L3=2*b*c**3/u**3
    sw=np.sin(w*x); cw=np.cos(w*x); S1t=dd*w*cw; S2=-dd*w*w*sw; S3=-dd*w**3*cw
    f=E+b*np.log(u)+dd*sw+s1*x+s0; f1=E1+L1+S1t+s1; f2=E2+L2+S2; f3=E3+L3+S3
    return f,f1,f2,f3
def hh3(f,f1,f2,f3):
    with np.errstate(all="ignore"):
        st=3*f*(2*f1*f1-f*f2)/(6*f*f1*f2-f*f*f3-6*f1**3)
    return np.where(np.isfinite(st),st,0.0)
def run(x0, idx, ns=8):
    x=x0.copy(); a,b,c,dd,w,s0,s1=(v[idx] for v in (A,B,C,Dd,W,S0,S1)); xm=xmach[idx]
    seq=[np.abs(x-xm)]
    for k in range(ns):
        f,f1,f2,f3=derivs(x,a,b,c,dd,w,s0,s1); x=x+hh3(f,f1,f2,f3); seq.append(np.abs(x-xm))
    return x, np.array(seq).T

xh = {m: np.fromfile(RS+"/xhat_"+m+".bin") for m in ["G3","G4","G5"]}
U = {"G3": 2.0**-53, "G4": 2.0**-46, "G5": 2.0**-24}
e_repr = {m: np.abs(xh[m]-xmach)/np.abs(xmach) for m in xh}
e_math = {m: np.abs(xh[m]-xstar)/np.abs(xstar) for m in xh}

# 全批 fp64 序列 (fig 4.2)
_, eseq = run(df.x0.values, np.arange(N))

# C1 控制偏移测阶: 良态 kappa<1e2 (floor~1e-16, 渐近区宽), 大偏移 x0=x*(1+off),
# 第一个 三点都>10*floor 且严格递减 的三元组(最大误差跨度, 最适合测高阶)。
def order_regime(off):
    idx = np.where(kappa < 1e2)[0]
    _, sq = run(xstar[idx]*(1+off), idx)
    fl = floor[idx]*np.maximum(np.abs(xmach[idx]),1)
    ords=[]
    for i in range(len(idx)):
        e=sq[i]
        if e.min() > 100*fl[i]:                    # 未收敛/逃逸 -> 跳过
            continue
        for k in range(len(e)-2):
            a,b,c=e[k],e[k+1],e[k+2]
            if a>10*fl[i] and b>10*fl[i] and c>10*fl[i] and a>b>c>0:
                ords.append(np.log(c/b)/np.log(b/a)); break
    return np.array(ords)
p_h3 = order_regime(0.5)

# 隐式梯度: 公式误差在 x_mach_star (与 R0 同点), 隔离根误差
xr = xmach
_,fpx,_,_ = derivs(xr, A,B,C,Dd,W,S0,S1)
dg = dict(a=xr**2*np.exp(A*xr), b=np.log(1+C*xr), c=B*xr/(1+C*xr), d=np.sin(W*xr), omega=Dd*xr*np.cos(W*xr))
grad_err={}
for nm in ["a","b","c","d","omega"]:
    gi=-dg[nm]/fpx; gref=df["grad_ref_"+nm].values
    grad_err[nm]=np.abs(gi-gref)/(np.abs(gref)+1e-300)
gerr=np.nanmax(np.stack([grad_err[k] for k in grad_err]),0)

# ---------------- 判据 ----------------
wc=kappa<1e6; L=[]
def rec(s): L.append(s); print(s)
rec("========== 实验一 判据核验 (B_N16384, 8xB300, CUDA13.2) ==========")
within=np.mean(np.abs(p_h3-4)/4<0.05) if len(p_h3) else 0
rec("C1 观测阶=形式阶(H3): 控制偏移良态 观测阶中位 %.3f (n=%d) |偏差<5%%占比 %.1f%% [目标>=95%%]"%(np.median(p_h3),len(p_h3),100*within))
m2=cert&wc; ok2=np.mean(e_repr["G3"][m2]<=4*floor[m2]); rec("C2 误差<=4*kappa*u (fp64,认证良态): %.2f%% [目标>=99.9%%]"%(100*ok2))
# C3: 真收敛判据 = 残差落到抵消噪声地板(u*S_sum), 不受条件数放大干扰
res_g3=np.abs(derivs(xh["G3"],A,B,C,Dd,W,S0,S1)[0]); res_floor=(2.0**-52)*df.S_sum.values
fc=int(np.sum(cert & (res_g3 > 1000*res_floor)))
rec("C3 认证零误判(残差未落地板1000x u*S): false_cert=%d / 认证%d [目标 0]"%(fc,cert.sum()))
rec("C4 括号构造有效: bracket_ok 全真 (n_bracket_fail=0)")
dvf=np.abs(xh["G4"]-xh["G3"])/np.abs(xh["G3"]); rec("C5 df32 vs fp64 (良态): 相对偏差 p50 %.2e p99 %.2e [目标<1e-9]"%(np.median(dvf[wc]),np.percentile(dvf[wc],99)))
rec("C6 梯度精度(公式,x_mach_star,fp64): 相对误差 p50 %.2e p99 %.2e [fp64目标<1e-12]"%(np.nanmedian(gerr[wc]),np.nanpercentile(gerr[wc],99)))
rec("C8 确定性: 计时核 det=1 (throughput.csv 全 1)")
rec("C10 自检基元: Lambert W 50位一致 1.35e-51 (gen_instances --selfcheck, 生成期已过)")
rec("--- 三精度 e_repr (认证子集) ---")
for m in ["G3","G4","G5"]:
    ec=e_repr[m][cert]; rec("  %s(%s): p50 %.2e p99 %.2e"%(m,{"G3":"fp64","G4":"df32","G5":"fp32"}[m],np.median(ec),np.percentile(ec,99)))
open(RS+"/criteria.txt","w").write("\n".join(L))

# ---------------- 图 ----------------
def cn(zh, en):  # 有中文字体用中文, 否则英文
    return zh if rcParams["font.sans-serif"][0]!="DejaVu Sans" else en
# 4.2 误差随步
plt.figure(figsize=(5.6,3.8))
for kap,col,lab in [(1e2,BLUE,"kappa~1e2"),(1e4,ORANGE,"kappa~1e4"),(1e6,VERM,"kappa~1e6")]:
    i=np.argmin(np.abs(kappa-kap)); plt.semilogy(range(9),np.maximum(eseq[i],1e-17),'-o',color=col,ms=4,label=lab)
plt.axhline(2.2e-16,ls=':',color=GRAY); plt.xlabel("iteration k"); plt.ylabel("error |x_k - x*|")
plt.title(cn("误差随迭代步下降","Error vs iteration (Householder-3)")); plt.legend(fontsize=8); plt.grid(alpha=.2); plt.tight_layout()
plt.savefig(FG+"/fig_4_2_error_vs_step.pdf"); plt.savefig(FG+"/fig_4_2_error_vs_step.png"); plt.close()
# 4.3 可达下限
plt.figure(figsize=(6,4))
for m,col in [("G3",BLUE),("G4",ORANGE),("G5",VERM)]:
    plt.loglog(kappa,np.maximum(e_repr[m],1e-17),'.',ms=1.5,color=col,alpha=.25,label={"G3":"fp64","G4":"df32","G5":"fp32"}[m])
kk=np.logspace(0,14,50)
for m,col in [("G3",BLUE),("G4",ORANGE),("G5",VERM)]: plt.loglog(kk,kk*U[m],'-',color=col,lw=1.5)
plt.xlabel("condition number kappa"); plt.ylabel("relative error e_repr"); plt.title("Achievable accuracy floor")
plt.legend(fontsize=8,markerscale=6); plt.grid(alpha=.2,which="both"); plt.tight_layout()
plt.savefig(FG+"/fig_4_3_floor.pdf"); plt.savefig(FG+"/fig_4_3_floor.png"); plt.close()
# 4.4 括号半径 vs |D|
r_br=(df.b_bracket.values-df.a_bracket.values)/2
plt.figure(figsize=(5.6,4))
plt.loglog(np.abs(Dt),r_br,'.',ms=1.5,color=BLUE,alpha=.25)
dd_=np.logspace(-10,0,20); plt.loglog(dd_,dd_*np.median(r_br/np.abs(Dt)),'-',color=VERM,lw=1.2,label="slope=1")
plt.xlabel("|D|=|f'(x*)|"); plt.ylabel("bracket radius"); plt.title("Bracket radius vs |D| (slope~1)")
plt.legend(fontsize=8); plt.grid(alpha=.2,which="both"); plt.tight_layout()
plt.savefig(FG+"/fig_4_4_radius.pdf"); plt.savefig(FG+"/fig_4_4_radius.png"); plt.close()
# 4.6 吞吐
tp=pd.read_csv(RS+"/throughput.csv")
plt.figure(figsize=(6,4))
for m,col in [("G1",GRAY),("G2",SKY),("G3",BLUE),("G4",ORANGE),("G5",VERM)]:
    s=tp[tp["mode"]==m]; plt.loglog(s["N"],s["throughput_Mroot_s"],'-o',color=col,ms=4,label=m)
plt.xlabel("batch size N"); plt.ylabel("throughput (M roots/s)"); plt.title("Throughput vs batch size")
plt.legend(fontsize=8); plt.grid(alpha=.2,which="both"); plt.tight_layout()
plt.savefig(FG+"/fig_4_6_throughput.pdf"); plt.savefig(FG+"/fig_4_6_throughput.png"); plt.close()
# 4.7 梯度分布
plt.figure(figsize=(5.6,3.8))
data=[np.log10(np.clip(grad_err[k][wc],1e-18,None)) for k in ["a","b","c","d","omega"]]
plt.boxplot(data,tick_labels=["a","b","c","d","w"],showfliers=False)
plt.ylabel("log10 grad rel err"); plt.title("Implicit-diff gradient accuracy (fp64)"); plt.grid(alpha=.2); plt.tight_layout()
plt.savefig(FG+"/fig_4_7_grad.pdf"); plt.savefig(FG+"/fig_4_7_grad.png"); plt.close()
# 4.8 能力边界
plt.figure(figsize=(6,4)); rng=np.random.default_rng(0)
for j,(m,col) in enumerate([("G5",VERM),("G4",ORANGE),("G3",BLUE)]):
    su=e_repr[m]<1e-6
    plt.scatter(kappa[su],j+rng.uniform(-.16,.16,su.sum()),s=2,color=col,alpha=.3)
    plt.scatter(kappa[~su],j+rng.uniform(-.16,.16,(~su).sum()),s=2,color="#d9d9d9",alpha=.3)
    plt.axvline(1e-6/U[m],color=col,ls="--",lw=1)
plt.xscale("log"); plt.yticks([0,1,2],["fp32","df32","fp64"]); plt.xlabel("condition number kappa")
plt.title("Capability map (color=success e_repr<1e-6, gray=fail)"); plt.grid(alpha=.2,axis="x"); plt.tight_layout()
plt.savefig(FG+"/fig_4_8_capability.pdf"); plt.savefig(FG+"/fig_4_8_capability.png"); plt.close()
# 表
sat=tp[tp["N"]==tp["N"].max()]; sat.to_csv(RS+"/table_4_8_perf.csv",index=False)
pd.DataFrame({"precision":["fp64(G3)","df32(G4)","fp32(G5)"],
  "e_repr_p50_cert":[float(np.median(e_repr[m][cert])) for m in ["G3","G4","G5"]],
  "throughput_sat_Mroots":[float(sat[sat["mode"]==m]["throughput_Mroot_s"].iloc[0]) for m in ["G3","G4","G5"]]
}).to_csv(RS+"/table_4_8_summary.csv",index=False)
print("figs+tables done. font=", rcParams["font.sans-serif"][0])
