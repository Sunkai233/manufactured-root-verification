set -e
cd ~/exp1_mrv
PY=venv/bin/python
echo "mode,N,nstep,launches,t_kernel_med_ms,t_kernel_iqr,t_e2e_med_ms,t_e2e_iqr,throughput_Mroot_s,det" > results/throughput.csv
for e in 10 12 14 16 18 20 22; do
  N=$((1<<e))
  $PY src/gen_fast.py $N data/fast_$N.bin 7 B >/dev/null
  for m in G1 G2 G3 G4 G5; do
    CUDA_VISIBLE_DEVICES=0 src/solver_timed data/fast_$N.bin $m 8 >> results/throughput.csv
  done
  rm -f data/fast_$N.bin
  echo "  N=$N done"
done
echo "=== throughput.csv ==="
cat results/throughput.csv
