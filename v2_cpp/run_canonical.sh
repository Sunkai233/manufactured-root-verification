#!/usr/bin/env bash
cd ~/exp1_mrv/v2_cpp || exit 1
g++ -O3 -march=native canonical.cpp -o canonical -Iinclude 2>/tmp/c.err || { echo BUILD_FAIL; cat /tmp/c.err; exit 1; }
./canonical ../data/B_N16384.bin alpha_data.csv
../venv/bin/python build_canonical.py
