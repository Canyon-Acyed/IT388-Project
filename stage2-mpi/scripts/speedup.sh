#!/bin/bash
# Measures speedup on this computer: trains to the target with the serial
# program and with 1, 2, 4, ... MPI processes (up to the number of cores),
# and prints time, speedup and efficiency.
#
# OUR CODE (whole file). Speedup S(p) = T(serial) / T(p) and efficiency
# E(p) = S(p) / p are the definitions in Pacheco, "An Introduction to Parallel
# Programming", 2011, Ch. 2.6.
#
#   scripts/speedup.sh
set -e
cd "$(dirname "$0")/.."
A="--quiet"
MAXP=${MAXP:-$(nproc)}
t_of() { awk '/^Done:/ {for (i = 1; i <= NF; i++) if ($i == "in") print $(i + 1)}'; }

ts=$(./bin/es_serial $A | t_of)
printf "%-10s %8s %8s %10s\n" "run" "time(s)" "speedup" "efficiency"
printf "%-10s %8.2f %8s %10s\n" "serial" "$ts" "1.00" "100%"
p=1
while (( p <= MAXP )); do
    tp=$(mpirun -np $p ./bin/es_mpi $A | t_of)
    awk -v s="$ts" -v t="$tp" -v p="$p" 'BEGIN { sp = s / t; printf "%-10s %8.2f %8.2f %9.0f%%\n", "mpi " p, t, sp, 100 * sp / p }'
    p=$((p * 2))
done
