# polarizable-stockmayer

Simulation code using LAMMPS Molecular Dynamics Simulator to simulate a polarizable stockmayer fluid and calculate resulting dielectric constants.

## To Run

1. Download and compile LAMMPS. Make sure to include all the packages required when compiling. DRUDE, GPU, and more.
2. Use `init.lammps` to generate `non_polarizable.data`.
3. Use `polarizer.py` to generate `polarizable.data`. Also use with `params.dff`. Run with `python3 polarizer.py -q -f params.dff non_polarizable.data polarizeable.data`.
4. We use the file `pair-drude-new.lmp` for params when running simulation.
5. Run simulation with `lmp -in polarization.lammps -sf gpu`.
