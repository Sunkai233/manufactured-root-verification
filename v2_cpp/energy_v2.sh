#!/usr/bin/env bash
# 修正能耗协议(纠 0.438 乐观~4×): ①大N饱和(REP=256→~4.19M) ②进程内持续满载(核背靠背, 无malloc/H2D进窗口)
# ③扣空载底噪 ④用配套的饱和吞吐(非另测的值)。对 G4(df32主配)/G3(fp64)/G5(fp32) 各测。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/tmp/nvcc.err && echo BUILT || { echo BUILD_FAIL; tail -5 /tmp/nvcc.err; exit 1; }
GPU=0
echo "采空载功率(4s)..."
IDLE=$(for i in $(seq 1 20); do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i $GPU; sleep 0.2; done | awk '{s+=$1;n++}END{printf "%.1f",s/n}')
echo "P_idle = $IDLE W"

measure(){
  local mode=$1
  REP=256 SUSTAIN_SEC=18 src/solver_timed data/B_N16384.bin $mode 8 >/tmp/thr_$mode.txt 2>/tmp/sus_$mode.txt &
  local PID=$!
  sleep 4                       # 等上量+时钟爬坡
  local PL=$(for i in $(seq 1 50); do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i $GPU; sleep 0.2; done | awk '{s+=$1;n++;if($1>mx)mx=$1}END{printf "%.1f %.1f",s/n,mx}')
  wait $PID
  local THR=$(tail -1 /tmp/thr_$mode.txt)
  local PMEAN=$(echo $PL|awk '{print $1}'); local PMAX=$(echo $PL|awk '{print $2}')
  local NETJ=$(awk -v p=$PMEAN -v id=$IDLE -v t=$THR 'BEGIN{printf "%.4f",(p-id)/t}')
  local RAWJ=$(awk -v p=$PMEAN -v t=$THR 'BEGIN{printf "%.4f",p/t}')
  printf "[%s] 饱和吞吐 %s Mroot/s | P_load %sW(峰 %s) P_idle %sW | 净能耗 %s J/Mroot (不扣空载 %s)\n" \
    "$mode" "$THR" "$PMEAN" "$PMAX" "$IDLE" "$NETJ" "$RAWJ"
}
measure G4
measure G3
measure G5
echo "对比: 旧 energy.sh = 0.438 J/Mroot(N=16384采功率÷365.5饱和吞吐, 含进程重启, 未扣空载)"
