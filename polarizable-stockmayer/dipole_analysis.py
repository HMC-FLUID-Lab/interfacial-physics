#!/usr/bin/env python3
# Simple, accurate dielectric constant calculator using reaction field formula
# Based on Neumann, M. (1983) "Dipole moment fluctuation formulas in computer simulations of polar systems"

import numpy as np
import argparse

def calculate_dielectric_constant(filename, volume, temperature=298.0, num_molecules=1000):
    """Calculate dielectric constant from dipole fluctuations using reaction field formula in SI units"""
    # SI constants
    kb_SI = 1.380649e-23  # Boltzmann constant in J/K
    e_SI = 1.602176634e-19  # elementary charge in C
    epsilon0_SI = 8.8541878128e-12  # vacuum permittivity in F/m
    angstrom_to_meter = 1e-10  # conversion from Å to m
    
    # Convert volume to SI
    volume_SI = volume * (angstrom_to_meter**3)  # Å³ to m³
    
    # Load the dipole data
    try:
        data = np.loadtxt(filename)
        print(f"Loaded {data.shape[0]} frames from {filename}")
    except:
        print(f"Standard loading failed, trying manual parsing...")
        with open(filename, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:  # We need at least 3 values (MUX, MUY, MUZ)
                try:
                    mx, my, mz = map(float, parts[-3:])  # Last 3 values
                    data.append([mx, my, mz])
                except ValueError:
                    pass  # Skip lines that can't be converted to floats
        data = np.array(data)
        print(f"Manually loaded {data.shape[0]} frames")
    
    # Extract dipole components (MUX, MUY, MUZ columns)
    dipoles = data[:, 0:3]
    
    # Calculate average dipole vector
    avg_dipole = np.mean(dipoles, axis=0)
    print(f"Average dipole vector: [{avg_dipole[0]:.4f}, {avg_dipole[1]:.4f}, {avg_dipole[2]:.4f}] e·Å")
    
    # Calculate |⟨M⟩|²
    avg_dipole_squared = np.sum(avg_dipole**2)
    print(f"|⟨M⟩|²: {avg_dipole_squared:.4f} (e·Å)²")
    
    # Calculate ⟨|M|²⟩
    m_squared = np.sum(dipoles**2, axis=1)
    avg_m_squared = np.mean(m_squared)
    print(f"⟨|M|²⟩: {avg_m_squared:.4f} (e·Å)²")
    
    # Calculate fluctuation ⟨|M|²⟩ - |⟨M⟩|²
    fluctuation = avg_m_squared - avg_dipole_squared
    print(f"Dipole fluctuation (⟨|M|²⟩ - |⟨M⟩|²): {fluctuation:.4f} (e·Å)²")
    
    # Convert dipole fluctuation from (e·Å)² to C²·m²
    # 1 e·Å = 1.602176634e-19 C * 1e-10 m = 1.602176634e-29 C·m
    dipole_factor = (e_SI * angstrom_to_meter)**2  # (e·Å)² to C²·m²
    fluctuation_SI = fluctuation * dipole_factor
    
    # Calculate susceptibility: χ = M²/(3ε₀kTV)
    susceptibility = fluctuation_SI / (3.0 * epsilon0_SI * kb_SI * temperature * volume_SI)
    
    # Calculate dielectric constant using Kirkwood-Fröhlich equation (reaction field formula)
    # ε = (1 + 2χ)/(1 - χ)
    epsilon_RF = (1.0 + 2.0 * susceptibility) / (1.0 - susceptibility)
    
    # Print results
    print("\n=== SI Unit Calculations ===")
    print(f"Volume: {volume:.2f} Å³ = {volume_SI:.4e} m³")
    print(f"Dipole fluctuation: {fluctuation:.4f} (e·Å)² = {fluctuation_SI:.4e} C²·m²")
    print(f"Susceptibility (χ): {susceptibility:.6f}")
    print(f"Dielectric constant (reaction field formula): {epsilon_RF:.6f}")
    
    # Save results to file
    with open("dielectric_results.txt", "w") as f:
        f.write("=== Dielectric Constant Analysis ===\n\n")
        f.write(f"System volume: {volume:.2f} Å³ = {volume_SI:.4e} m³\n")
        f.write(f"Temperature: {temperature:.2f} K\n")
        f.write(f"Number of molecules: {num_molecules}\n\n")
        
        f.write(f"Average dipole vector: [{avg_dipole[0]:.4f}, {avg_dipole[1]:.4f}, {avg_dipole[2]:.4f}] e·Å\n")
        f.write(f"|⟨M⟩|²: {avg_dipole_squared:.4f} (e·Å)²\n")
        f.write(f"⟨|M|²⟩: {avg_m_squared:.4f} (e·Å)²\n")
        f.write(f"Dipole fluctuation (⟨|M|²⟩ - |⟨M⟩|²): {fluctuation:.4f} (e·Å)²\n\n")
        
        f.write("=== Kirkwood-Fröhlich Analysis (Reaction Field) ===\n")
        f.write(f"Susceptibility (χ): {susceptibility:.6f}\n")
        f.write(f"Dielectric constant (ε = (1+2χ)/(1-χ)): {epsilon_RF:.6f}\n")
    
    print(f"\nResults saved to dielectric_results.txt")
    
    return epsilon_RF

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dielectric constant from dipole data using reaction field formula")
    parser.add_argument("-f", "--file", default="dipole.dat", help="Dipole data file from LAMMPS (default: dipole.dat)")
    parser.add_argument("-v", "--volume", type=float, default=29218.112, help="System volume in Å³ (default: 29218.112)")
    parser.add_argument("-t", "--temperature", type=float, default=298.0, help="Temperature in K (default: 298.0)")
    parser.add_argument("-n", "--nmol", type=int, default=1000, help="Number of molecules (default: 1000)")
    
    args = parser.parse_args()
    
    dielectric = calculate_dielectric_constant(args.file, args.volume, args.temperature, args.nmol)
    print(f"Final dielectric constant: {dielectric:.6f}")
