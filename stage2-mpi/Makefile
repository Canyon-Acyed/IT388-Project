# Stage 2: serial and MPI versions of ES on the inverted double pendulum
#
# OUR CODE (whole file). MuJoCo include/link flags per MuJoCo's programming
# docs (https://mujoco.readthedocs.io/en/stable/programming/index.html);
# mpicc is the standard Open MPI / MPICH compiler wrapper.
#
#   make            build bin/es_serial, bin/es_mpi and bin/env_test
#   make check      compare the C environment with Gymnasium (needs Python)
#   make verify     check that every process count trains the identical controller
#   make clean

MUJOCO_DIR ?= third_party/mujoco
CC         ?= gcc
MPICC      ?= mpicc
CFLAGS     ?= -O3 -march=native -Wall -Wextra -Wno-unused-parameter
CPPFLAGS   := -Isrc -I$(MUJOCO_DIR)/include
LDLIBS     := -L$(MUJOCO_DIR)/lib -lmujoco -lm -Wl,-rpath,$(abspath $(MUJOCO_DIR)/lib)

SRC  := src/es.c src/env.c src/policy.c
HDRS := src/env.h src/policy.h src/rng.h

all: bin/es_serial bin/es_mpi bin/env_test

bin:
	mkdir -p bin

bin/es_serial: $(SRC) $(HDRS) | bin
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SRC) -o $@ $(LDLIBS)

bin/es_mpi: $(SRC) $(HDRS) | bin
	$(MPICC) $(CFLAGS) -DUSE_MPI $(CPPFLAGS) $(SRC) -o $@ $(LDLIBS)

bin/env_test: src/env_test.c src/env.c $(HDRS) | bin
	$(CC) $(CFLAGS) $(CPPFLAGS) src/env_test.c src/env.c -o $@ $(LDLIBS)

check: bin/env_test
	python3 scripts/check_env.py

verify: bin/es_serial bin/es_mpi
	scripts/verify_identical.sh

clean:
	rm -rf bin

.PHONY: all check verify clean
