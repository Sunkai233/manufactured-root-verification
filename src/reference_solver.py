#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一 · 步骤2: 高精度参考实现 R0 (mpmath 50 位, CPU 多核并行)。

对每个实例产出:
  x_mach_star : 舍入后机器方程 f(x)=g(x;fp64参数)+s1_f64*x+s0_f64 的真根 (50 位) —— e_repr 基准
  grad_ref_{a,b,c,d,omega} : 隐函数定理梯度 -dg/dtheta / f'(x*)  —— 隐式微分梯度基准 (20)

用法: python reference_solver.py <in.parquet> [--procs 128]
就地把新列写回 parquet。
"""
import argparse, time
import numpy as np, pandas as pd
import mpmath as mp
from multiprocessing import Pool

mp.mp.dps = 50

def _row(args):
    mp.mp.dps = 60                              # 每进程内设精度, 留足余量
    a, b, c, d, w, s0f, s1f, xstar_str, ab, bb = args
    a, b, c, d, w = map(mp.mpf, (a, b, c, d, w))
    s0, s1 = mp.mpf(s0f), mp.mpf(s1f)          # 工作精度舍入后的源项
    x0 = mp.mpf(xstar_str); ab, bb = mp.mpf(ab), mp.mpf(bb)
    def g(x):  return x * mp.e ** (a * x) + b * mp.log(1 + c * x) + d * mp.sin(w * x)
    def gp(x): return mp.e ** (a * x) * (1 + a * x) + b * c / (1 + c * x) + d * w * mp.cos(w * x)
    def f(x):  return g(x) + s1 * x + s0
    def fp(x): return gp(x) + s1
    # 机器方程真根: 从制造根出发手写高精度 Newton, 相对步长停机 (对小 D 浅根稳健)
    xr = x0
    for _ in range(200):
        fpx = fp(xr)
        if fpx == 0:
            break
        dx = f(xr) / fpx
        xr = xr - dx
        if mp.fabs(dx) <= mp.fabs(xr) * mp.mpf("1e-45") + mp.mpf("1e-55"):
            break
    if not (ab <= xr <= bb):                    # 逃出括号则二分兜底
        lo, hi = ab, bb
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0: hi = mid
            else: lo = mid
            if hi - lo <= mp.fabs(mid) * mp.mpf("1e-45"):
                break
        xr = (lo + hi) / 2
    fpx = fp(xr)
    dg_da = xr ** 2 * mp.e ** (a * xr)
    dg_db = mp.log(1 + c * xr)
    dg_dc = b * xr / (1 + c * xr)
    dg_dd = mp.sin(w * xr)
    dg_dw = d * xr * mp.cos(w * xr)
    gr = [-v / fpx for v in (dg_da, dg_db, dg_dc, dg_dd, dg_dw)]
    return (mp.nstr(xr, 50), float(xr), [float(x) for x in gr])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--procs", type=int, default=64)
    args = ap.parse_args()
    df = pd.read_parquet(args.inp)
    argrows = list(zip(df.a, df.b, df.c, df.d, df.omega, df.s0_f64, df.s1_f64,
                       df.x_star, df.a_bracket, df.b_bracket))
    t0 = time.time()
    with Pool(args.procs) as pool:
        res = pool.map(_row, argrows, chunksize=max(1, len(argrows) // (args.procs * 4)))
    df["x_mach_star"] = [r[0] for r in res]
    df["x_mach_star_f64"] = [r[1] for r in res]
    for j, name in enumerate(["a", "b", "c", "d", "omega"]):
        df["grad_ref_" + name] = [r[2][j] for r in res]
    df.to_parquet(args.inp, index=False)
    # e_repr 参考下限自检: 机器根与制造根之差应 <= 数倍 e_floor
    dev = np.abs(df.x_mach_star_f64.values - df.x_star.astype(float).values) / np.abs(df.x_star.astype(float).values)
    print("[R0 完成] %d 实例  %.1fs  (procs=%d)" % (len(df), time.time() - t0, args.procs))
    print("  机器根 vs 制造根 相对偏移: 中位 %.2e  p99 %.2e  max %.2e" %
          (np.median(dev), np.percentile(dev, 99), dev.max()))
    print("  该偏移应受条件数限制 (~kappa*u): e_floor_rel 中位 %.2e p99 %.2e" %
          (np.median(df.e_floor_rel), np.percentile(df.e_floor_rel, 99)))
    print("  新列:", [c for c in df.columns if c.startswith("grad_ref_") or "mach" in c])

if __name__ == "__main__":
    main()
