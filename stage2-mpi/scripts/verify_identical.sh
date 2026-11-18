#!/bin/bash
# Correctness check: the serial program and the MPI program with any number of
# processes must produce exactly the same trained controller ("theta hash").
# If processes ever disagreed about theta, the hashes would differ.
#
# OUR CODE (whole file). mpirun and its --oversubscribe option are Open MPI's
# standard launcher (https://docs.open-mpi.org/en/main/man-openmpi/man1/mpirun.1.html).
#
#   scripts/verify_identical.sh
set -e
cd "$(dirname "$0")/.."
A="--gens 12 --eval-every 4 --target 0 --quiet"
RUNS=(
  "./bin/es_serial"
  "mpirun --oversubscribe -np 1 ./bin/es_mpi"
  "mpirun --oversubscribe -np 2 ./bin/es_mpi"
  "mpirun --oversubscribe -np 3 ./bin/es_mpi"
  "mpirun --oversubscribe -np 5 ./bin/es_mpi"
)
first=""
fail=0
for cmd in "${RUNS[@]}"; do
    h=$($cmd $A | awk '/theta hash/ {print $NF}')
    printf "%-50s %s\n" "$cmd" "$h"
    [[ -z "$first" ]] && first=$h
    [[ "$h" != "$first" ]] && fail=1
done
if (( fail )); then echo "MISMATCH: runs disagree"; exit 1; else echo "All runs produced the identical controller."; fi
