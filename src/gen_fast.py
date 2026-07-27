#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""吞吐批量快速生成器 (float64 向量化, 直出 .bin)。
只用于计时/吞吐曲线, 不含 50 位参考 (gamma_bound/x_mach_star/certified 为占位)。
布局同 export_bin: int32 N; 10*float64 (a,b,c,d,omega,s0,s1,x0,占位,占位); int32 certified。
用法: python gen_fast.py <N> <out.bin> [seed] [comp]
"""
import sys, struct
import numpy as np

N = int(sys.argv[1]); out = sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
comp = sys.argv[4] if len(sys.argv) > 4 else "B"
rng = np.random.default_rng(seed)

xstar = 10 ** rng.uniform(np.log10(0.2), np.log10(3.0), N)
a = rng.uniform(0.5, 3.0, N)
b = rng.uniform(-1, 1, N)
u = 10 ** rng.uniform(np.log10(0.05), np.log10(2.0), N)
c = (u - 1.0) / xstar
d = rng.uniform(-0.5, 0.5, N)
w = 10 ** rng.uniform(np.log10(1.0), np.log10(20.0), N)
if comp == "A":
    logD = rng.uniform(-1.0, 0.0, N); alpha_t = 10 ** rng.uniform(-4, -2, N)
else:
    logD = rng.uniform(-10.0, 0.0, N); alpha_t = 10 ** rng.uniform(-4, 1, N)
D = (10.0 ** logD) * np.where(rng.random(N) < 0.5, 1.0, -1.0)

def g(x):  return x * np.exp(a * x) + b * np.log(1 + c * x) + d * np.sin(w * x)
def gp(x): return np.exp(a * x) * (1 + a * x) + b * c / (1 + c * x) + d * w * np.cos(w * x)
def gpp(x):return np.exp(a*x)*(a*(2+a*x)) - b*c*c/(1+c*x)**2 - d*w*w*np.sin(w*x)

s1 = D - gp(xstar)
s0 = -(g(xstar) + s1 * xstar)
gamma = np.abs(gpp(xstar) / (2 * D))            # 粗 gamma (仅热启动定标)
domain_r = np.abs(1 + c * xstar) / (np.abs(c) + 1e-30)
off = np.sign(rng.uniform(-1, 1, N)) * np.minimum(alpha_t / (gamma + 1e-30), 0.9 * domain_r)
x0 = xstar + off

arr = np.stack([a, b, c, d, w, s0, s1, x0, gamma, xstar]).astype(np.float64)
cert = np.ones(N, np.int32)
with open(out, "wb") as f:
    f.write(struct.pack("i", N)); arr.tofile(f); cert.tofile(f)
print("fast bin N=%d -> %s (comp %s)" % (N, out, comp))
