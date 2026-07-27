#!/usr/bin/env bash
# 生成画图所需紧凑 CSV: 可达下限(kappa,e_repr G3/G4/G5) + 重跑 xhat。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/dev/null
for m in G3 G4 G5; do src/solver_timed data/B_N16384.bin $m 8 results/xhat_$m.bin >/dev/null 2>&1; done
./venv/bin/python - <<'PY'
import numpy as np, pandas as pd, os
B=os.path.expanduser("~/exp1_mrv")
df=pd.read_parquet(B+"/data/B_N16384.parquet")
xm=df.x_mach_star_f64.values.astype(float); kap=df.kappa.values.astype(float)
cert=df.certified.values.astype(int)
out={"kappa":kap,"certified":cert}
for m in ["G3","G4","G5"]:
    xh=np.fromfile(B+"/results/xhat_%s.bin"%m)
    n=min(len(xh),len(xm))
    e=np.abs(xh[:n]-xm[:n])/np.maximum(np.abs(xm[:n]),1.0)
    out["erepr_%s"%m]=np.concatenate([e,np.full(len(xm)-n,np.nan)]) if n<len(xm) else e
pd.DataFrame(out).to_csv(B+"/v2_cpp/floor_data.csv",index=False)
print("floor_data.csv rows",len(kap),"kappa range %.2e..%.2e"%(kap.min(),kap.max()))
PY
