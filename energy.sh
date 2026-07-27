pkill -f ncu 2>/dev/null; pkill -f solver_timed 2>/dev/null; sleep 1
cd ~/exp1_mrv
( for i in $(seq 1 40); do src/solver_timed data/B_N16384.bin G4 8 >/dev/null; done ) &
LP=$!
for i in $(seq 1 30); do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0; sleep 0.15; done > /tmp/pw2.log
kill $LP 2>/dev/null
awk '{p+=$1; if($1>mx)mx=$1; n++} END{printf "P_mean=%.1fW  P_max=%.1fW  n=%d  ->  J/Mroot(G4主配@365.5Mrs)=%.3f\n", p/n, mx, n, (p/n)/365.5}' /tmp/pw2.log
