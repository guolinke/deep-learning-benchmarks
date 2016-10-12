#!/bin/bash
sudo ./tensorflow_build.sh
./tensorflow_run_benchmarks.sh "$@"
