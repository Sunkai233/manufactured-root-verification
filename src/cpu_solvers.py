#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一 · 步骤3(CPU部分): R1 brentq / R2 Newton + 算法核心向量化验证。

核心 = 热启动 + Householder-d 高阶迭代 + alpha 认证 (与 CUDA 的 SIMT 同一算法),
fp64 先验证数学: 逐实例观测收敛阶(式17, 判据C1)、认证零误判(判据C3)、
误差贴可达下限、与 R0 机器真根一致。

用法: python cpu_solvers.py <parquet> [--tol 1e-12]
"""
import argparse, time
import numpy as np, pandas as pd
from scipy.optimize import brentq

ALPHA0 = 0.157671

def derivs(x, a, b, c, dd, w, s0, s1):
    """f, f', f'', f''' (fp64, 逐元素向量化)。"""
    ex = np.exp(a * x)
    E  = x * ex
    E1 = ex * (a * x + 1.0)
    E2 = a * ex * (a * x + 2.0)
    E3 = a * a * ex * (a * x + 3.0)
    u  = 1.0 + c * x
    L  = b * np.log(u)
    L1 = b * c / u
    L2 = -b * c * c / u ** 2
    L3 = 2.0 * b * c ** 3 / u ** 3
    sw = np.sin(w * x); cw = np.cos(w * x)
    S  = dd * sw
    S1 = dd * w * cw
    S2 = -dd * w * w * sw
    S3 = -dd * w ** 3 * cw
    f  = E + L + S + s1 * x + s0
    f1 = E1 + L1 + S1 + s1
    f2 = E2 + L2 + S2
    f3 = E3 + L3 + S3
    return f, f1, f2, f3

def hh_step(order, f, f1, f2, f3):
    """Householder-d 步长, 直接有理式 (f 在分子, 收敛时步长->0, 无除零)。
    order=1 Newton / 2 Halley / 3 三阶。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        if order == 1:
            step = -f / f1
        elif order == 2:
            step = -2.0 * f * f1 / (2.0 * f1 * f1 - f * f2)
        else:  # H3
            num = 3.0 * f * (2.0 * f1 * f1 - f * f2)
            den = 6.0 * f * f1 * f2 - f * f * f3 - 6.0 * f1 ** 3
            step = num / den
    # 已收敛(f=0)或非有限 -> 冻结
    return np.where(np.isfinite(step), step, 0.0)

def run_batch(df, order, tol, nstep=8, x0_override=None):
    """向量化跑一批, 返回 x_hat, 误差序列, k_actual, 认证等。"""
    a, b, c, dd, w = (df[k].values.astype(np.float64) for k in ["a", "b", "c", "d", "omega"])
    s0, s1 = df.s0_f64.values, df.s1_f64.values
    x = (df.x0.values.astype(np.float64).copy() if x0_override is None
         else np.asarray(x0_override, np.float64).copy())
    xmach = df.x_mach_star_f64.values
    xstar = df.x_star.astype(float).values
    gamma = df.gamma_bound.values
    # 认证: 在初值处 beta=|f/f'|, alpha=beta*gamma
    f0, f1_0, _, _ = derivs(x, a, b, c, dd, w, s0, s1)
    beta0 = np.abs(f0 / f1_0)
    alpha = beta0 * gamma
    certified = alpha < ALPHA0
    # 固定步数迭代(无分支), 记录每步误差 vs 机器真根
    eseq = np.zeros((len(df), nstep + 1))
    eseq[:, 0] = np.abs(x - xmach)
    for k in range(nstep):
        f, f1, f2, f3 = derivs(x, a, b, c, dd, w, s0, s1)
        step = hh_step(order, f, f1, f2, f3)
        x = x + step
        eseq[:, k + 1] = np.abs(x - xmach)
    x_hat = x
    res = np.abs(derivs(x_hat, a, b, c, dd, w, s0, s1)[0])
    # k_actual: 首次误差 < tol*max(|xmach|,1)
    thr = tol * np.maximum(np.abs(xmach), 1.0)
    below = eseq <= thr[:, None]
    k_actual = np.where(below.any(1), below.argmax(1), nstep)
    e_repr_rel = np.abs(x_hat - xmach) / np.abs(xmach)
    e_math_rel = np.abs(x_hat - xstar) / np.abs(xstar)
    return dict(x_hat=x_hat, res=res, eseq=eseq, k_actual=k_actual,
                certified=certified, e_repr_rel=e_repr_rel, e_math_rel=e_math_rel)

def observed_order(eseq, floor):
    """式(17): 只在收敛实例的干净渐近区估阶。
    要求: 该实例最终收敛(min<100*floor); 取所有 严格递减且三点都>3*floor 的
    三元组, 取其观测阶的中位数(对噪声稳健)。跳出收敛域/未收敛者返回 nan。"""
    p = np.full(len(eseq), np.nan)
    nstep = eseq.shape[1]
    for i in range(len(eseq)):
        e = eseq[i]; fl = 3.0 * floor[i]
        if e.min() > 100.0 * floor[i]:      # 未收敛/跳出域 -> 跳过
            continue
        ps = []
        for k in range(1, nstep - 1):
            a3, b3, c3 = e[k-1], e[k], e[k+1]
            if a3 > fl and b3 > fl and c3 > fl and a3 > b3 > c3 > 0:
                ps.append(np.log(c3 / b3) / np.log(b3 / a3))
        if ps:
            p[i] = np.median(ps)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("--tol", type=float, default=1e-12)
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    cert = df.certified.values
    print("=== A) 认证实例应快速收敛到机器精度 (fp64, N=%d, 认证 %d) ===" % (len(df), cert.sum()))
    for order, name in [(1, "Newton"), (2, "Halley"), (3, "Householder-3")]:
        r = run_batch(df, order, args.tol)
        ec = r["e_repr_rel"][cert]; kc = r["k_actual"][cert]
        eu = r["e_repr_rel"][~cert]
        print("[%-13s] 认证子集: e_repr p50 %.2e p99 %.2e | 步数 p50 %d p99 %d | 未认证 e_repr p50 %.2e" %
              (name, np.nanmedian(ec), np.nanpercentile(ec, 99), int(np.median(kc)),
               int(np.percentile(kc, 99)), np.nanmedian(eu)))

    print("=== B) 观测收敛阶 C1 (良态 kappa<1e3, 固定偏移 +0.15|x*|, 干净渐近) ===")
    mask = df.kappa.values < 1e3
    sub = df[mask].reset_index(drop=True)
    xs = sub.x_star.astype(float).values
    x0o = xs * 1.15
    floor_sub = sub.e_floor_rel.values * np.maximum(np.abs(sub.x_mach_star_f64.values), 1.0)
    for order, name, formal in [(1, "Newton", 2), (2, "Halley", 3), (3, "Householder-3", 4)]:
        r = run_batch(sub, order, args.tol, x0_override=x0o)
        p = observed_order(r["eseq"], floor_sub)
        pv = p[np.isfinite(p)]
        within = np.mean(np.abs(pv - formal) / formal < 0.05) if len(pv) else 0
        print("[%-13s] 形式阶=%d 观测阶: p50 %.3f p05 %.3f (n=%d) |偏差<5%%占比 %.1f%% (C1)" %
              (name, formal, np.nanmedian(pv) if len(pv) else float('nan'),
               np.nanpercentile(pv, 5) if len(pv) else float('nan'), len(pv), 100*within))

    print("=== 诊断: 5 个最良态实例的 H3 误差序列 (+0.15 偏移) ===")
    idx = np.argsort(sub.kappa.values)[:5]
    r = run_batch(sub, 3, args.tol, x0_override=x0o)
    for i in idx:
        gxs = abs(sub.s0_f64.values[i])  # |s0|~|g(x*)| 量级, 抵消来源
        seq = "  ".join("%.1e" % e for e in r["eseq"][i])
        print("  kappa=%.1e |s0|=%.1e floor=%.1e | eseq: %s" %
              (sub.kappa.values[i], gxs, floor_sub[i], seq))

    print("=== C) 认证零误判 C3 ===")
    r3 = run_batch(df, 3, args.tol)
    kpred = df.k_pred.values
    false_cert = np.sum(cert & (r3["res"] > args.tol) & (r3["e_repr_rel"] > 1e-6))
    print("认证率=%.1f%%  false_cert(通过认证却未收敛, 须0)=%d" % (100*cert.mean(), int(false_cert)))
    # R1 brentq / R2 Newton 一致性 (抽样, vs 机器真根)
    t0 = time.time(); nb = min(len(df), 512)
    xr1 = np.zeros(nb)
    for i in range(nb):
        a,b,c,dd,w,s0,s1 = (df[k].values[i] for k in ["a","b","c","d","omega","s0_f64","s1_f64"])
        fb = lambda x: derivs(np.array([x]),a,b,c,dd,w,s0,s1)[0][0]
        try: xr1[i] = brentq(fb, df.a_bracket.values[i], df.b_bracket.values[i], xtol=1e-15, rtol=1e-15, maxiter=200)
        except Exception: xr1[i] = np.nan
    d_r1 = np.abs(xr1 - df.x_mach_star_f64.values[:nb]) / np.abs(df.x_mach_star_f64.values[:nb])
    print("R1 brentq vs 机器真根: 中位 %.2e p99 %.2e  (%.2fs, %d实例)" %
          (np.nanmedian(d_r1), np.nanpercentile(d_r1, 99), time.time()-t0, nb))

if __name__ == "__main__":
    main()
