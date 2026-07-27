#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一 v2 · 出版级八图重画(本地 matplotlib, 微软雅黑中文, Okabe-Ito 色盲友好)。
数据: floor_data.csv / diff3_data.csv / results_stepdist.csv + 汇总实测数(见各图注释)。"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
rcParams["axes.unicode_minus"]=False
rcParams["savefig.dpi"]=200; rcParams["figure.dpi"]=120
rcParams["axes.spines.top"]=False; rcParams["axes.spines.right"]=False
rcParams["font.size"]=11
# Okabe-Ito
BLUE,ORANGE,GREEN,VERM,SKY,YEL,PUR,GRAY="#0072B2","#E69F00","#009E73","#D55E00","#56B4E9","#F0E442","#CC79A7","#999999"
HERE=os.path.dirname(os.path.abspath(__file__)); FG=os.path.join(HERE,"figs_v2"); os.makedirs(FG,exist_ok=True)
def save(fig,name):
    fig.tight_layout(); fig.savefig(os.path.join(FG,name+".pdf")); fig.savefig(os.path.join(FG,name+".png")); plt.close(fig)
    print("  ->",name)

# ---------- 图1 收敛阶(__float128 双偏移单步法) ----------
def fig_order():
    meth=["Newton\n(2阶)","Halley\n(3阶)","Householder-3\n(4阶)"]; formal=[2,3,4]; obs=[2.0003,2.9917,3.8845]; pct=[99.8,98.4,93.4]
    x=np.arange(3); w=0.36; fig,ax=plt.subplots(figsize=(6.2,4))
    ax.bar(x-w/2,formal,w,label="形式阶",color=GRAY,alpha=.65)
    ax.bar(x+w/2,obs,w,label="观测阶(__float128)",color=BLUE)
    for i,(o,p) in enumerate(zip(obs,pct)): ax.text(i+w/2,o+0.06,f"{o:.3f}\n{p:.0f}%内5%",ha="center",va="bottom",fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(meth); ax.set_ylabel("收敛阶"); ax.set_ylim(0,4.7)
    ax.set_title("收敛阶验证:观测阶=形式阶(高精度双偏移单步法)"); ax.legend(loc="upper left"); ax.grid(axis="y",alpha=.25)
    save(fig,"fig1_order")

# ---------- 图2 可达精度下限(核心, 唯一真源 instances.csv) ----------
def fig_floor():
    # 地板数据集(floorscaled δ=M·κu): 每实例起于地板上方固定倍数, 真迭代到地板(无构造偏置)
    df=pd.read_csv(os.path.join(HERE,"instances_floor.csv")); k=df.kappa.values
    kfl=1.0/np.sqrt(2.0**-53)   # 基本极限 κ≈1/√u≈1.3e8: 地板 κu 超牛顿盆地 1/κ
    fig,ax=plt.subplots(figsize=(6.8,4.9))
    for m,c,lab in [("e_G3",BLUE,"fp64 (u=2^-53)"),("e_G4",ORANGE,"df32 (u=2^-46)"),("e_G5",GREEN,"fp32 (u=2^-24)")]:
        e=df[m].values; mask=np.isfinite(e)&(e>0)
        ax.scatter(k[mask],e[mask],s=3,c=c,alpha=.25,edgecolors="none",label=lab,rasterized=True)
    ks=np.logspace(np.log10(k.min()),np.log10(k.max()),50)
    for u,c in [(2**-53.,BLUE),(2**-46.,ORANGE),(2**-24.,GREEN)]:
        ax.plot(ks,np.clip(ks*u,1e-17,None),"--",c=c,lw=1.4,alpha=.9)
    ax.axvline(kfl,ls="-.",c=VERM,lw=1.2,alpha=.75)
    ax.text(kfl*0.5,3e0,"基本极限 κ≈1/√u\n(地板 κu > 牛顿盆地 1/κ,\n迭代无法达地板)",fontsize=7.5,color=VERM,ha="right",va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("条件数 κ"); ax.set_ylabel("可达相对误差 e_repr")
    ax.set_title("可达精度下限:floorscaled 初值, 三精度各贴 κ·u 线(无构造偏置)")
    ax.legend(markerscale=4,loc="upper left",framealpha=.9); ax.grid(alpha=.2,which="both"); ax.set_ylim(1e-17,3e1)
    save(fig,"fig2_floor")

# ---------- 图3 可微性三路互证 ----------
def fig_diff():
    df=pd.read_csv(os.path.join(HERE,"diff3_data.csv")); k=df.kappa.values
    fig,ax=plt.subplots(figsize=(6.6,4.6))
    ax.scatter(k,df.rAB.values,s=3,c=BLUE,alpha=.25,edgecolors="none",label="A(前向AD) vs B(隐式闭式)",rasterized=True)
    ax.scatter(k,df.rBC.values,s=3,c=VERM,alpha=.22,edgecolors="none",label="B(隐式) vs C(中心差分)",rasterized=True)
    ks=np.logspace(np.log10(max(k.min(),1)),np.log10(k.max()),50)
    ax.plot(ks,ks*2**-53,"--",c=BLUE,lw=1.4,label="κ·u 参考线 (u=2^-53)")
    ax.axhline(4e-11,ls=":",c=GRAY,lw=1.2,label="中心差分地板 ~u^(2/3)")
    ax.axvline(1e4,ls="-.",c=VERM,lw=1.2,alpha=.7)
    ax.text(3e2,3e1,"B-C 有效域\nκ<1e4 (~34%样本)",fontsize=8.5,color=VERM,ha="center",va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("条件数代理 κ (γ界)"); ax.set_ylabel("梯度相对误差")
    ax.set_title("可微性:A-B 全程随 κ·u;C(差分)有效域 κ<1e4"); ax.legend(markerscale=4,loc="lower right",fontsize=8.5,framealpha=.9)
    ax.grid(alpha=.2,which="both"); ax.set_ylim(1e-16,1e2)
    save(fig,"fig3_diff3")

# ---------- 图4 认证 vs 未认证 步数分布(唯一真源, 全量+溢出桶) ----------
def fig_stepdist():
    df=pd.read_csv(os.path.join(HERE,"instances_cert.csv"))  # kindep 认证数据集(认证收敛/未认证不收敛的干净分离)
    cap=40; DISP=15   # 显示 0..DISP, 其余(含失败>cap)并入溢出桶
    kn=df.kn.values; cert=df.cert.values.astype(bool)   # cert = long double 稳健认证
    def hist(mask):
        h=np.zeros(DISP+2)
        for v in kn[mask]: h[v if v<=DISP else DISP+1]+=1
        return h/mask.sum()
    def stat(mask):
        v=kn[mask]; return int(np.median(v)), v.mean(), 100*np.mean(v>cap), mask.sum()
    hc=hist(cert); hu=hist(~cert)
    mc,ac,fc,nc=stat(cert); mu,au,fu,nu=stat(~cert)
    x=np.arange(DISP+2); w=0.42; fig,ax=plt.subplots(figsize=(7.2,4.3))
    ax.bar(x-w/2,hc,w,color=GREEN,label="认证 α<0.1577 (n=%d, 中位%d, 均值%.2f, 失败%.2f%%)"%(nc,mc,ac,fc))
    ax.bar(x+w/2,hu,w,color=VERM,label="未认证 (n=%d, 中位%d, 均值%.2f, 失败%.2f%%)"%(nu,mu,au,fu))
    ax.set_xticks(list(range(0,DISP+1,2))+[DISP+1])
    ax.set_xticklabels([str(i) for i in range(0,DISP+1,2)]+[f">{DISP}"])
    ax.set_xlabel("达 ε 所需 Newton 迭代步数 (全量含溢出桶; 数据集 kindep)"); ax.set_ylabel("占比")
    ax.set_title("认证价值=步数分布(kindep):认证点步数短 vs 未认证长尾")
    ax.text(0.98,0.60,"注: 干净分离在“步数”,\n非“认证点必达 x*”\n(kindep 大偏移下 20.8%\n认证点收敛到邻根)",
            transform=ax.transAxes,ha="right",va="top",fontsize=7.5,color=GRAY)
    ax.legend(loc="upper right"); ax.grid(axis="y",alpha=.25)
    save(fig,"fig4_stepdist")

# ---------- 图5 线程束浪费(实测步数) ----------
def fig_warp():
    # 运行时从 floorscaled 数据集算(有真实难度方差, 调度键才有意义; kindep 步数近均匀不适合)
    df=pd.read_csv(os.path.join(HERE,"instances_floor.csv"))
    kh=df.kh.values; kap=df.kappa.values; al=df.alpha.values; ce=df.cert.values
    def ww(order,ns=8):
        s=np.minimum(kh[order],ns); m=len(s)//32*32; s=s[:m].reshape(-1,32)
        return float(np.mean(1-s.mean(1)/np.maximum(s.max(1),1)))
    keys=[("随机排布",np.arange(len(kh)),VERM),("κ先验排序",np.argsort(kap),ORANGE),
          ("α先验排序\n(最优可实现)",np.argsort(al),GREEN),("认证标志排序",np.argsort(ce),SKY),
          ("步数oracle\n(不可实现)",np.argsort(kh),GRAY),("固定步\n(无早退)",None,BLUE)]
    labs=[k[0] for k in keys]; vals=[ww(k[1]) if k[1] is not None else 0.0 for k in keys]; cols=[k[2] for k in keys]
    fig,ax=plt.subplots(figsize=(7.6,4.2))
    b=ax.bar(labs,vals,color=cols)
    for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.012,f"{v:.3f}",ha="center",fontsize=9)
    ax.text(5,0.05,"定义值",ha="center",fontsize=8,color=GRAY)
    ax.set_ylabel("线程束浪费 (锁步, 步数封顶 ns=8, floorscaled)"); ax.set_ylim(0,max(vals)*1.25)
    ax.set_title("warp-waste:κ/α/认证 先验键均大幅降(0.63→~0.3),固定步靠构造归零")
    ax.grid(axis="y",alpha=.25); plt.setp(ax.get_xticklabels(),fontsize=8.5)
    save(fig,"fig5_warp")

# ---------- 图6 硬件峰值吞吐 跨卡对照 ----------
def fig_peak():
    # 实测: B300 fp32/fp64/df32 = 70933/993/3780; 5090 = 64106/1578/3151 GFLOP/s
    b300=[70933,993,3780]; r5090=[64106,1578,3151]; labs=["fp32","fp64","df32"]
    x=np.arange(3); w=0.36; fig,ax=plt.subplots(figsize=(6.4,4.3))
    ax.bar(x-w/2,b300,w,color=BLUE,label="B300 (R=71.4)")
    ax.bar(x+w/2,r5090,w,color=ORANGE,label="RTX5090 (R=40.6)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("峰值吞吐 GFLOP/s (log)")
    for i,(a,b) in enumerate(zip(b300,r5090)):
        ax.text(i-w/2,a*1.15,f"{a/1000:.0f}T",ha="center",fontsize=8); ax.text(i+w/2,b*1.15,f"{b/1000:.1f}T",ha="center",fontsize=8)
    ax.set_title("两卡 fp64 皆重度削减;df32 纯FMA 反超 fp64"); ax.legend(); ax.grid(axis="y",alpha=.25,which="both")
    ax.set_ylim(500,1.6e5)
    save(fig,"fig6_peak")

# ---------- 图7 能耗每根 ----------
def fig_energy():
    labs=["G3 fp64","G4 df32(主)","G5 fp32"]; net=[0.2574,0.6741,0.0203]; cols=[BLUE,ORANGE,GREEN]
    fig,ax=plt.subplots(figsize=(5.8,4))
    b=ax.bar(labs,net,color=cols)
    for r,v in zip(b,net): ax.text(r.get_x()+r.get_width()/2,v+0.012,f"{v:.3f}",ha="center",fontsize=10)
    ax.set_ylabel("净能耗 J / 百万根(扣空载)"); ax.set_title("能耗(修正协议):df32 在 B300 最费电")
    ax.grid(axis="y",alpha=.25); ax.set_ylim(0,0.78)
    save(fig,"fig7_energy")

# ---------- 图8 CFG 示意(汇编证据: g3 直链 vs g2 菱形) ----------
def fig_cfg():
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,4.4))
    def box(ax,xy,txt,w=0.5,h=0.13,fc="white",ec=BLUE):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((xy[0]-w/2,xy[1]-h/2),w,h,boxstyle="round,pad=0.02",fc=fc,ec=ec,lw=1.5))
        ax.text(xy[0],xy[1],txt,ha="center",va="center",fontsize=8.5)
    def arrow(ax,p,q,c="#333"): ax.annotate("",xy=q,xytext=p,arrowprops=dict(arrowstyle="-|>",color=c,lw=1.4))
    for ax in (a1,a2): ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    # g3 直链单回边
    a1.set_title("k_g3 认证固定步\nSASS:循环体直链 DFMA·无早退分支",fontsize=10)
    box(a1,(0.5,0.86),"入口 x=x0")
    box(a1,(0.5,0.60),"derivs + HH3 步\n(DFMA 链)",ec=GREEN,fc="#eaf7f0")
    box(a1,(0.5,0.34),"k<8 ?",ec=GRAY)
    box(a1,(0.5,0.10),"写回 x_hat")
    arrow(a1,(0.5,0.79),(0.5,0.67)); arrow(a1,(0.5,0.53),(0.5,0.41)); arrow(a1,(0.5,0.27),(0.5,0.17))
    a1.annotate("",xy=(0.62,0.60),xytext=(0.72,0.34),arrowprops=dict(arrowstyle="-|>",color=GRAY,lw=1.2,connectionstyle="arc3,rad=-0.5"))
    a1.text(0.83,0.47,"统一回边\n(全束同步)",fontsize=7.5,color=GRAY,ha="center")
    # g2 菱形(早退分支)
    a2.set_title("k_g2 带早退\nSASS:多 1 条数据相关分支 (+1 BSSY)",fontsize=10)
    box(a2,(0.5,0.88),"入口 x=x0")
    box(a2,(0.5,0.64),"derivs + HH3 步",ec=GREEN,fc="#eaf7f0")
    from matplotlib.patches import Polygon
    a2.add_patch(Polygon([(0.5,0.52),(0.68,0.42),(0.5,0.32),(0.32,0.42)],closed=True,fc="#fdeee3",ec=VERM,lw=1.5))
    a2.text(0.5,0.42,"|step|<tol ?\n(数据相关)",ha="center",va="center",fontsize=8,color=VERM)
    box(a2,(0.5,0.10),"写回 x_hat")
    arrow(a2,(0.5,0.81),(0.5,0.71)); arrow(a2,(0.5,0.57),(0.5,0.53))
    a2.annotate("",xy=(0.5,0.17),xytext=(0.5,0.32),arrowprops=dict(arrowstyle="-|>",color=VERM,lw=1.4))
    a2.text(0.60,0.24,"break→发散",fontsize=7.5,color=VERM)
    a2.annotate("",xy=(0.80,0.64),xytext=(0.68,0.42),arrowprops=dict(arrowstyle="-|>",color=GRAY,lw=1.2,connectionstyle="arc3,rad=-0.5"))
    a2.text(0.88,0.53,"回边",fontsize=7.5,color=GRAY,ha="center")
    save(fig,"fig8_cfg")

# ---------- 图9 M1–M9 阶次×精度全因子(吞吐 + 精度) ----------
def fig_matrix():
    p=os.path.join(HERE,"matrix_floor.csv")
    if not os.path.exists(p): print("  (跳过 fig9: 无 matrix_floor.csv)"); return
    df=pd.read_csv(p)
    precs=["fp64","df32","fp32"]; orders=[2,3,4]; onm={2:"Newton",3:"Halley",4:"HH3"}
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.2))
    x=np.arange(3); w=0.25; cols={2:BLUE,3:ORANGE,4:GREEN}
    for j,o in enumerate(orders):
        thr=[float(df[(df.prec==p)&(df.order==o)].Mroot_s.iloc[0]) for p in precs]
        ax1.bar(x+(j-1)*w,thr,w,color=cols[o],label="%d阶 %s"%(o,onm[o]))
    ax1.set_yscale("log"); ax1.set_xticks(x); ax1.set_xticklabels(precs); ax1.set_ylabel("饱和吞吐 M根/s (log)")
    ax1.set_title("M1–M9 吞吐:阶次↑→每步更贵吞吐↓"); ax1.legend(fontsize=8); ax1.grid(axis="y",alpha=.25,which="both")
    for j,o in enumerate(orders):
        er=[float(df[(df.prec==p)&(df.order==o)].relerr_p50.iloc[0]) for p in precs]
        ax2.bar(x+(j-1)*w,er,w,color=cols[o],label="%d阶"%o)
    ax2.set_yscale("log"); ax2.set_xticks(x); ax2.set_xticklabels(precs); ax2.set_ylabel("e_repr p50 (log, 8步固定)")
    ax2.set_title("M1–M9 精度:阶次↑→8步内更准"); ax2.legend(fontsize=8); ax2.grid(axis="y",alpha=.25,which="both")
    save(fig,"fig9_matrix")

if __name__=="__main__":
    print("生成图 ->",FG)
    fig_order(); fig_floor(); fig_diff(); fig_stepdist(); fig_warp(); fig_peak(); fig_energy(); fig_cfg(); fig_matrix()
    print("完成")
