cd ~/exp1_mrv
echo "=== ncu 可用性 ==="
NCU=$(which ncu 2>/dev/null || ls /usr/local/cuda/bin/ncu 2>/dev/null || echo "")
echo "ncu=$NCU"
echo "=== 能耗: 后台采功率, 前台跑 G4 持续负载 ==="
( for i in $(seq 1 400); do nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -i 0; done > /tmp/pw.log ) &
PID=$!
for i in $(seq 1 120); do src/solver_timed data/B_N16384.bin G4 8 >/dev/null; done
kill $PID 2>/dev/null
echo "采样点 $(wc -l < /tmp/pw.log)"
echo "功率均值/最大(W) 与 SM时钟均值(MHz):"
awk -F',' '{p+=$1;if($1>mx)mx=$1;c+=$2;n++} END{printf "P_mean=%.1f P_max=%.1f clk_mean=%.0f (n=%d)\n",p/n,mx,c/n,n}' /tmp/pw.log
echo "=== J/Mroot (G4主配, 用 P_mean/throughput) ==="
awk -F',' 'BEGIN{tp=365.5} {p+=$1;n++} END{printf "P_mean=%.1fW / %.1f Mroots/s = %.3f J/Mroot\n",p/n,tp,(p/n)/tp}' /tmp/pw.log
if [ -n "$NCU" ]; then
  echo "=== Nsight 剖析 G3(fp64) 与 G4(df32) SpeedOfLight+Occupancy ==="
  $NCU --set basic --launch-count 1 --kernel-name-base demangled --print-summary per-kernel \
       src/solver_timed data/B_N16384.bin G3 8 2>&1 | grep -iE "Duration|Compute .SM|Memory .%|Achieved Occupancy|Registers Per Thread" | head -12
  echo "--- G4 ---"
  $NCU --set basic --launch-count 1 src/solver_timed data/B_N16384.bin G4 8 2>&1 | grep -iE "Duration|Compute .SM|Memory .%|Achieved Occupancy|Registers Per Thread" | head -12
else
  echo "ncu 不可用(共享机常见,需权限); 硬件计数器 S2 细项标注为待补"
fi
