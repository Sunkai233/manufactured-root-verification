set -e
cd ~/exp1_mrv
/usr/local/cuda/bin/nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed
echo "=== 时钟状态 (是否降频参考) ==="
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw --format=csv,noheader -i 0
echo "mode,N,nstep,launches,t_kernel_med_ms,t_kernel_iqr,t_e2e_med_ms,t_e2e_iqr,throughput_Mroot_s,det"
for m in G1 G2 G3 G4 G5; do CUDA_VISIBLE_DEVICES=0 src/solver_timed data/B_N16384.bin $m 8; done
