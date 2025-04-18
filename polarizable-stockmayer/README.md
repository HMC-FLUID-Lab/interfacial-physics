# Polarizable Stockmayer Fluid Simulation

This repository contains simulation code using the LAMMPS Molecular Dynamics Simulator to model a polarizable Stockmayer fluid and calculate resulting dielectric constants. The simulation uses Drude oscillators to model molecular polarizability.

## Overview

The Stockmayer fluid is a model of polar molecules with Lennard-Jones interactions and point dipoles. This implementation extends the model to include polarizability using the Drude oscillator approach, allowing for accurate calculation of dielectric properties.

## Prerequisites

- LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)
- Python 3.x
- Required LAMMPS packages:
  - DRUDE (for polarizable models)
  - GPU (for GPU acceleration)
  - KSPACE (for long-range electrostatics)

## Simulation Workflow

1. **Create base system**:
   ```bash
   lmp -in stockmayer_base.lmp -sf gpu
   ```
   This generates a non-polarizable Stockmayer fluid configuration.

2. **Add Drude oscillators**:
   ```bash
   python3 polarizer.py -t OW -f drude.dff stockmayer_base.data stockmayer_drude.data
   ```
   The `polarizer.py` script adds Drude particles to the base configuration using parameters from `drude.dff`.

3. **Run the polarizable simulation**:
   ```bash
   OMP_NUM_THREADS=32 lmp -in stockmayer_drude.lmp
   ```
   This runs the main simulation with Drude oscillators and outputs dipole moment data.

4. **Analyze dielectric properties**:
   ```bash
   python3 dipole_analysis.py
   ```
   This script calculates the dielectric constant from dipole moment fluctuations using the Kirkwood-Fröhlich reaction field formula.

   Note that this script currently assumes the volume of of the box, this may need to be changed.

## Script Descriptions

- `stockmayer_base.lmp`: Creates the initial non-polarizable fluid configuration
- `stockmayer_drude.lmp`: Main simulation script for the polarizable fluid
- `polarizer.py`: Utility to add Drude oscillators to LAMMPS data files
- `dipole_analysis.py`: Analyzes dipole moment time series to calculate dielectric constants
- `script.sh`: Convenience script that runs the full workflow

## Parameter Files

- `drude.dff`: Drude oscillator parameters for the fluid model
- `pair-drude.lmp`: Generated file with pair interactions for Drude particles

## Advanced Usage

To customize the polarizability properties, edit the `drude.dff` file:
```
# type  dm/u  dq/e  k/(kJ/molA2)  alpha/A3  thole
OW     0.4   -1.0   2092.0        1.45      2.6
```

Where:
- `dm` is the mass to place on the Drude particle (taken from its core)
- `dq` is the charge to place on the Drude particle (taken from its core)
- `k` is the harmonic force constant of the bond between core and Drude
- `alpha` is the polarizability
- `thole` is a parameter of the Thole damping function

## Analysis Output

The dielectric constant calculation uses the Kirkwood-Fröhlich reaction field formula:
```
ε = (1 + 2χ)/(1 - χ)
```
where χ is the susceptibility calculated from dipole moment fluctuations.

## Notes

- For accurate dielectric calculations, ensure adequate equilibration and sampling time
- The thermostatted temperature for Drude particles should be around 1K
- Results are saved in `dielectric_results.txt`
