#!/bin/sh
lmp -in stockmayer_base.lmp -sf gpu

# Some manual intervention required before this step (ideally, but failsafe should work)
python3 polarizer.py -t OW stockmayer_base.data stockmayer_drude.data

# Clean base file
rm stockmayer_base.data

# Final run
OMP_NUM_THREADS=32 lmp -in stockmayer_drude.lmp

# Analysis
python3 dipole_analysis.py
