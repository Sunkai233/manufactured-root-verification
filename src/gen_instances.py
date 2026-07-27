#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一 · 制造根验证基准 — 数据生成脚本 (方案 4.10 第一步)。

基础族 (8):  g(x) = x*e^(a x) + b*ln(1+c x) + d*sin(w x)
制造残差 (9): f(x) = g(x) + s1*(x - x*) + [g(x*) 项通过 s0 归零]
  取 f(x) = g(x) - g(x*) + s1*(x - x*),  s1 = D - g'(x*)   (10)(11)
  => f(x*)=0, f'(x*)=D, 且 k>=2 阶 f^(k)=g^(k)。
落表字段严格对齐方案表 4.4 (S1 生成期部分)。

用法:
  python gen_instances.py --comp B --N 1024 --seed 12345 --out ../data/B_N1024.parquet
  python gen_instances.py --selfcheck        # 只跑 Lambert W 自检基元 (判据 C10)
"""
import argparse, hashlib, json, sys, time
import numpy as np
import mpmath as mp

mp.mp.dps = 50
ALPHA0 = mp.mpf("0.157671")     # 认证阈值 (方案指定)
K_ONLINE = 8                    # 在线 gamma 上界取 k=2..8
U_FP64 = mp.mpf(2) ** -53       # 单位舍入 (round-to-nearest)

# ---------------- 基础族与导数 (mpmath 50 位) ----------------
def g_terms(x, a, b, c, d, w):
    x = mp.mpf(x)
    te = x * mp.e ** (a * x)
    tl = b * mp.log(1 + c * x)
    ts = d * mp.sin(w * x)
    return te, tl, ts

def g_val(x, a, b, c, d, w):
    te, tl, ts = g_terms(x, a, b, c, d, w)
    return te + tl + ts

def gp_val(x, a, b, c, d, w):           # g'(x)
    x = mp.mpf(x)
    return mp.e ** (a * x) * (1 + a * x) + b * c / (1 + c * x) + d * w * mp.cos(w * x)

def gpp_val(x, a, b, c, d, w):          # g''(x)
    x = mp.mpf(x)
    return mp.e ** (a * x) * (a * (2 + a * x)) - b * c * c / (1 + c * x) ** 2 - d * w * w * mp.sin(w * x)

def Ak_bound(x, a, b, c, d, w, k):
    """|g^(k)(x)| 的逐项闭式上界 (12), k>=2。"""
    x = mp.mpf(x)
    Aexp = mp.e ** (a * x) * mp.fabs(a) ** (k - 1) * (k + mp.fabs(a * x))
    Alog = mp.fabs(b) * mp.factorial(k - 1) * mp.fabs(c) ** k / mp.fabs(1 + c * x) ** k
    Asin = mp.fabs(d) * mp.fabs(w) ** k
    return Aexp + Alog + Asin

def gamma_from_bounds(x, a, b, c, d, w, D, kmax):
    """gamma 上界 (13): max_{k=2..kmax} (A_k/(k! |D|))^(1/(k-1))。"""
    D = mp.fabs(D)
    best = mp.mpf(0)
    for k in range(2, kmax + 1):
        val = (Ak_bound(x, a, b, c, d, w, k) / (mp.factorial(k) * D)) ** (mp.mpf(1) / (k - 1))
        if val > best:
            best = val
    return best

# ---------------- 单实例构造 ----------------
def build_instance(a, b, c, d, w, xstar, D, delta, sign_off, tol=mp.mpf("1e-12"),
                   offset_mode="kindep", M=100.0):
    # offset_mode: "kindep"=κ无关相对偏移 delta(认证价值实验); "floorscaled"=M·κ·u·|x*|(地板实验, 每实例起于地板上方固定倍数M, 强制迭代M步到地板)
    a, b, c, d, w = map(mp.mpf, (a, b, c, d, w))
    xstar = mp.mpf(xstar); D = mp.mpf(D)
    gpx = gp_val(xstar, a, b, c, d, w)
    gx  = g_val(xstar, a, b, c, d, w)
    s1 = D - gpx                       # 线性源项 (10)
    s0 = -(gx + s1 * xstar)            # 常数源项: f(x*)=g(x*)+s1*x*+s0=0

    def f(x):   return g_val(x, a, b, c, d, w) + s1 * x + s0
    def fp(x):  return gp_val(x, a, b, c, d, w) + s1

    # 对数定义域安全半径: 保证 [x*-r, x*+r] 内 1+c*x>0 (奇点在 x=-1/c)
    domain_r = mp.fabs(1 + c * xstar) / (mp.fabs(c) + mp.mpf("1e-30")) if c != 0 else mp.mpf("1e30")
    rmax = mp.mpf("0.9") * domain_r

    # gamma: 在线上界(K=8) 与 参考(k->30)
    gamma_bound = gamma_from_bounds(xstar, a, b, c, d, w, D, K_ONLINE)
    gamma_ref   = gamma_from_bounds(xstar, a, b, c, d, w, D, 30)

    # 括号区间 (14): r < |D| / max|g''|, 收缩因子 rho=1/2, 迭代一次自洽
    rho = mp.mpf("0.5")
    r = rho * mp.fabs(D) / (mp.fabs(gpp_val(xstar, a, b, c, d, w)) + mp.mpf("1e-30"))
    for _ in range(3):
        M2 = max(mp.fabs(gpp_val(xstar - r, a, b, c, d, w)),
                 mp.fabs(gpp_val(xstar + r, a, b, c, d, w)),
                 mp.fabs(gpp_val(xstar, a, b, c, d, w)))
        r = rho * mp.fabs(D) / (M2 + mp.mpf("1e-30"))
    if r > rmax:
        r = rmax
    a_b, b_b = xstar - r, xstar + r
    for _ in range(60):                           # 收缩到端点异号(单根附近必成立)
        if f(a_b) * f(b_b) < 0:
            break
        r = r * mp.mpf("0.5"); a_b, b_b = xstar - r, xstar + r
    bracket_ok = bool(f(a_b) * f(b_b) < 0)

    # 量级和 S 与相对条件数 (16)
    te, tl, ts = g_terms(xstar, a, b, c, d, w)
    S_sum = mp.fabs(te) + mp.fabs(tl) + mp.fabs(ts) + mp.fabs(s0) + mp.fabs(s1 * xstar)
    kappa = S_sum / (mp.fabs(D) * mp.fabs(xstar))

    # ★初值(v4): 两种 κ 无偏构造(均消除旧 off∝1/γ 的构造偏置)
    if offset_mode == "floorscaled":
        # 地板缩放: |x0-x*| = M·(κu)·max(|x*|,1) = 固定 M 倍地板 → 每实例都真迭代~log_p(M)步到地板
        off = sign_off * mp.mpf(M) * kappa * U_FP64 * max(mp.fabs(xstar), mp.mpf(1))
    else:
        # κ 无关相对偏移: |x0-x*| = delta·max(|x*|,1), delta∈[1e-3,1e-1]
        off = sign_off * delta * max(mp.fabs(xstar), mp.mpf(1))
    if mp.fabs(off) > rmax:                       # 初值留在实定义域内(极端 c 才触发)
        off = mp.sign(off) * rmax
    x0 = xstar + off
    beta0 = mp.fabs(f(x0) / fp(x0))
    alpha0_val = beta0 * gamma_ref
    certified = bool(alpha0_val < ALPHA0)

    # k_pred: Newton alpha-理论步数上界  |e_n|<=(1/2)^(2^n-1)|e0|, 到 tol*max(|x*|,1)
    e0 = mp.fabs(off); target = tol * max(mp.fabs(xstar), mp.mpf(1))
    k_pred = 0
    if e0 > 0:
        e = e0
        while e > target and k_pred < 100:
            e = mp.mpf("0.5") ** (2 ** k_pred) * e0  # 粗上界
            k_pred += 1
    return dict(a=float(a), b=float(b), c=float(c), d=float(d), omega=float(w),
                x_star=mp.nstr(xstar, 50), D_target=float(D),
                s0=mp.nstr(s0, 50), s1=mp.nstr(s1, 50),
                s0_f64=float(s0), s1_f64=float(s1),
                a_bracket=float(a_b), b_bracket=float(b_b), bracket_ok=bracket_ok,
                S_sum=float(S_sum), kappa=float(kappa),
                gamma_bound=float(gamma_bound), gamma_ref=float(gamma_ref),
                x0=float(x0), beta0=float(beta0), alpha0_val=float(alpha0_val),
                certified=certified, k_pred=int(k_pred),
                e_floor_rel=float(kappa * U_FP64))

# ---------------- 参数采样 (表 4.1) ----------------
def sample_params(rng, comp):
    xstar = 10 ** rng.uniform(np.log10(0.2), np.log10(3.0))     # log-uniform
    a = rng.uniform(0.5, 3.0)
    b = rng.uniform(-1, 1)
    u = 10 ** rng.uniform(np.log10(0.05), np.log10(2.0))        # 1+c*x* in [0.05,2]
    c = (u - 1.0) / xstar
    d = rng.uniform(-0.5, 0.5)
    w = 10 ** rng.uniform(np.log10(1.0), np.log10(20.0))
    # D: 分层对数, 符号随机(κ 由 D 分层扫描, 与初值偏移解耦)
    # ★δ 采样放在旧 alpha_t 的同一 RNG 位置(同为一次 uniform), 使同 seed 下 a..D 全同、仅初值变
    if comp == "A":                      # 易解: 条件数~1 => |D| 较大
        logD = rng.uniform(-1.0, 0.0)
        delta = 10 ** rng.uniform(-3.0, -1.0)   # κ 无关相对偏移 [1e-3,1e-1]
    else:                                 # B/C: |D| 10 数量级分层
        logD = rng.uniform(-10.0, 0.0)
        delta = 10 ** rng.uniform(-3.0, -1.0)
    D = (10 ** logD) * (1 if rng.random() < 0.5 else -1)
    sign_off = 1 if rng.random() < 0.5 else -1
    return dict(a=a, b=b, c=c, d=d, w=w, xstar=xstar, D=D,
                delta=mp.mpf(delta), sign_off=sign_off)

# ---------------- 自检基元 (4.2.6, 判据 C10) ----------------
def selfcheck(n=20):
    print("[自检基元] b=d=0, a=1, D=g'(x*) => s1=0, f(x)=x e^x - x* e^(x*), 根应 = x* (=W(x* e^x*))")
    rng = np.random.default_rng(2024)
    worst = mp.mpf(0)
    for i in range(n):
        xstar = mp.mpf(10 ** rng.uniform(np.log10(0.2), np.log10(3.0)))
        a = mp.mpf(1); b = d = mp.mpf(0); c = mp.mpf(1); w = mp.mpf(1)
        D = gp_val(xstar, a, b, c, d, w)         # 使 s1=0
        inst = build_instance(a, b, c, d, w, xstar, D, mp.mpf("1e-3"), 1)
        # 独立用 lambertw 求 x e^x = x* e^x* 的根, 主分支 = x*
        arg = xstar * mp.e ** xstar
        root_lw = mp.lambertw(arg, 0)
        err = mp.fabs(root_lw - xstar) / mp.fabs(xstar)
        worst = max(worst, err)
    ok = worst < mp.mpf("1e-45")
    print("  最坏相对偏差 = %s   -> %s" % (mp.nstr(worst, 5), "PASS (C10)" if ok else "FAIL"))
    return ok

# ---------------- 主流程 ----------------
def generate(comp, N, seed, out, offset_mode="kindep", M=100.0):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    rows = []
    n_bad = 0
    for i in range(N):
        p = sample_params(rng, comp)
        inst = build_instance(p["a"], p["b"], p["c"], p["d"], p["w"],
                              p["xstar"], p["D"], p["delta"], p["sign_off"],
                              offset_mode=offset_mode, M=M)
        inst["run_id"] = 0; inst["inst_id"] = i; inst["batch_comp"] = comp; inst["seed"] = seed
        if not inst["bracket_ok"]:
            n_bad += 1
        rows.append(inst)
        if (i + 1) % 2000 == 0:
            print("  %d/%d  (%.1fs)" % (i + 1, N, time.time() - t0), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    # 内容哈希
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    manifest = dict(out=out, comp=comp, N=N, seed=seed, sha256=h,
                    n_bracket_fail=n_bad, cert_rate=float(df["certified"].mean()),
                    kappa_min=float(df["kappa"].min()), kappa_max=float(df["kappa"].max()),
                    gen_seconds=round(time.time() - t0, 2))
    json.dump(manifest, open(out + ".manifest.json", "w"), indent=2, ensure_ascii=False)
    print("[生成完成] %s" % json.dumps(manifest, ensure_ascii=False))
    # 判据 C4: 括号构造有效 (全部端点异号)
    print("  判据 C4 (bracket_ok 全真): %s (失败 %d)" % ("PASS" if n_bad == 0 else "FAIL", n_bad))
    return manifest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="B", choices=["A", "B", "C"])
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="../data/inst.parquet")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--offset_mode", default="kindep", choices=["kindep", "floorscaled"])
    ap.add_argument("--M", type=float, default=100.0)
    args = ap.parse_args()
    if args.selfcheck:
        sys.exit(0 if selfcheck() else 1)
    generate(args.comp, args.N, args.seed, args.out, offset_mode=args.offset_mode, M=args.M)
