#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parquet -> 二进制 (SoA), 供 CUDA 求解器读取。
布局: int32 N; 然后 10 个 float64 数组(各 N): a,b,c,d,omega,s0,s1,x0,gamma_bound,x_mach_star;
      然后 int32 数组(N): certified。
用法: python export_bin.py <in.parquet> <out.bin>
"""
import sys, struct
import numpy as np, pandas as pd

inp, out = sys.argv[1], sys.argv[2]
df = pd.read_parquet(inp)
N = len(df)
cols = ["a", "b", "c", "d", "omega", "s0_f64", "s1_f64", "x0", "gamma_bound", "x_mach_star_f64"]
arr = np.stack([df[c].values.astype(np.float64) for c in cols])   # (10, N) SoA
cert = df["certified"].values.astype(np.int32)
with open(out, "wb") as f:
    f.write(struct.pack("i", N))
    arr.tofile(f)
    cert.tofile(f)
print("wrote %s  N=%d  (%d cols f64 + certified i32)" % (out, N, len(cols)))
