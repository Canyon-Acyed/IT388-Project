# Stage 1: serial ES on the inverted double pendulum
#
# OUR CODE (whole file). Compiler and linker flags for MuJoCo follow MuJoCo's
# "Getting started / Building" docs: include <mujoco/mujoco.h> and link
# libmujoco (https://mujoco.readthedocs.io/en/stable/programming/index.html).
#
#   make            build bin/es_serial and bin/env_test
#   make check      compare the C environment with Gymnasium (needs Python)
#   make clean
#
# MUJOCO_DIR must contain include/mujoco/mujoco.h and lib/libmujoco.so.
# scripts/get_mujoco.sh downloads it into third_party/mujoco.

MUJOCO_DIR ?= third_party/mujoco
CC         ?= gcc
CFLAGS     ?= -O3 -march=native -Wall -Wextra -Wno-unused-parameter
CPPFLAGS   := -Isrc -I$(MUJOCO_DIR)/include
LDLIBS     := -L$(MUJOCO_DIR)/lib -lmujoco -lm -Wl,-rpath,$(abspath $(MUJOCO_DIR)/lib)

SRC  := src/es.c src/env.c src/policy.c
HDRS := src/env.h src/policy.h src/rng.h

all: bin/es_serial bin/env_test

bin:
	mkdir -p bin

bin/es_serial: $(SRC) $(HDRS) | bin
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SRC) -o $@ $(LDLIBS)

bin/env_test: src/env_test.c src/env.c $(HDRS) | bin
	$(CC) $(CFLAGS) $(CPPFLAGS) src/env_test.c src/env.c -o $@ $(LDLIBS)

check: bin/env_test
	python3 scripts/check_env.py

clean:
	rm -rf bin

.PHONY: all check clean
