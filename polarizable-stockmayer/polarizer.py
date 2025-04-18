#!/usr/bin/env python
# polarizer.py - add Drude oscillators to LAMMPS data file.
# Modified version to handle missing headers/type labels
# Originally by: Agilio Padua <agilio.padua@univ-bpclermont.fr>
# Alain Dequidt <alain.dequidt@univ-bpclermont.fr>

import sys
import argparse
import random
from copy import deepcopy

usage = """Add Drude oscillators to LAMMPS data file.

Format of file containing specification of Drude oscillators:

  # type  dm/u  dq/e  k/(kJ/molA2)  alpha/A3  thole
  OW      0.4   -1.0   2092.0        1.45     2.6
  ...

* dm is the mass to place on the Drude particle (taken from its core),
* dq is the charge to place on the Drude particle (taken from its core),
* k is the harmonic force constant of the bond between core and Drude,
* alpha is the polarizability,
* thole is a parameter of the Thole damping function.

A Drude particle is created for each atom in the LAMMPS data file
that corresponds to an atom type given in the Drude file.
Since LAMMPS uses numbers for atom types in the data file, a comment
after each line in the Masses section has to be introduced to allow
identification of the atom types within the force field database:

  Masses

  1   12.011  # C3H
  2   12.011  # CTO
  ...

===== MODIFICATIONS =====
This modified version can handle:
- Missing type labels in the Masses section (using a command line flag)
- Missing sections by creating them if needed
- More robust handling of input files

Command line flags:
  -t TYPE, --type TYPE   Add "# TYPE" label to all mass entries (if missing)
"""

# keywords of header and main sections (from data.py in Pizza.py)

hkeywords = ["atoms", "ellipsoids", "lines", "triangles", "bodies",
             "bonds", "angles", "dihedrals", "impropers",
             "atom types", "bond types", "angle types", "dihedral types",
             "improper types", "xlo xhi", "ylo yhi", "zlo zhi", "xy xz yz"]

skeywords = [["Masses", "atom types"],
             ["Pair Coeffs", "atom types"],
             ["Bond Coeffs", "bond types"], ["Angle Coeffs", "angle types"],
             ["Dihedral Coeffs", "dihedral types"],
             ["Improper Coeffs", "improper types"],
             ["BondBond Coeffs", "angle types"],
             ["BondAngle Coeffs", "angle types"],
             ["MiddleBondTorsion Coeffs", "dihedral types"],
             ["EndBondTorsion Coeffs", "dihedral types"],
             ["AngleTorsion Coeffs", "dihedral types"],
             ["AngleAngleTorsion Coeffs", "dihedral types"],
             ["BondBond13 Coeffs", "dihedral types"],
             ["AngleAngle Coeffs", "improper types"],
             ["Atoms", "atoms"], ["Velocities", "atoms"],
             ["Ellipsoids", "ellipsoids"],
             ["Lines", "lines"], ["Triangles", "triangles"],
             ["Bodies", "bodies"],
             ["Bonds", "bonds"],
             ["Angles", "angles"], ["Dihedrals", "dihedrals"],
             ["Impropers", "impropers"], ["Molecules", "atoms"]]


def massline(att):
    return "{0:4d} {1:8.3f}  # {2}\n".format(att['id'], att['m'], att['type'])

def bdtline(bdt):
    return "{0:4d} {1:12.6f} {2:12.6f}  {3}\n".format(bdt['id'], bdt['k'],
                                                     bdt['r0'], bdt['note'])

def atomline(at):
    return "{0:7d} {1:7d} {2:4d} {3:8.4f} {4:13.6e} {5:13.6e} {6:13.6e} "\
           " {7}\n".format(at['n'], at['mol'], at['id'], at['q'],
                           at['x'], at['y'], at['z'], at['note'])

def bondline(bd):
    return "{0:7d} {1:4d} {2:7d} {3:7d}  {4}\n".format(bd['n'], bd['id'],
                                            bd['i'], bd['j'], bd['note'])

def velline(at):
    return "{0:7d} {1:13.6e} {2:13.6e} {3:13.6e} \n".format(at['n'],
                                       at['vx'], at['vy'], at['vz'])

# --------------------------------------


class Data(object):

    def __init__(self, datafile, default_type=None):
        '''read LAMMPS data file (from data.py in Pizza.py)'''

        # for extract method
        self.atomtypes = []
        self.bondtypes = []
        self.atoms = []
        self.bonds = []
        self.idmap = {}
        self.default_type = default_type

        self.nselect = 1

        f = open(datafile, "r")

        self.title = f.readline()
        self.names = {}

        headers = {}
        while 1:
            line = f.readline().strip()
            if '#' in line:
                line = line[:line.index('#')].strip()
            if len(line) == 0:
                continue
            found = 0
            for keyword in hkeywords:
                if keyword in line:
                    found = 1
                    words = line.split()
                    if keyword == "xlo xhi" or keyword == "ylo yhi" or \
                      keyword == "zlo zhi":
                        headers[keyword] = (float(words[0]), float(words[1]))
                    elif keyword == "xy xz yz":
                        headers[keyword] = \
                          (float(words[0]), float(words[1]), float(words[2]))
                    else:
                        headers[keyword] = int(words[0])
            if not found:
                break

        sections = {}
        while 1:
            if len(line) > 0:
                found = 0
                for pair in skeywords:
                    keyword, length = pair[0], pair[1]
                    if keyword == line:
                        found = 1
                        if length not in headers:
                            if length == "atoms" and keyword == "Atoms":
                                print("Warning: Missing 'atoms' header. Will attempt to count from Atoms section.")
                                line_count = count_atoms_in_section(f)
                                headers[length] = line_count
                                f.seek(0)  # Reset file position
                                # Skip to Atoms section again
                                while True:
                                    line = f.readline()
                                    if not line or line.strip() == "Atoms":
                                        break
                            else:
                                print(f"Warning: '{length}' header not found for {keyword} section")
                                headers[length] = 0  # Default to 0 for missing headers
                        f.readline()  # Skip header line
                        list_ = []
                        for _ in range(headers[length]):
                            list_.append(f.readline())
                        sections[keyword] = list_
                if not found:
                    # Try to continue on unknown section
                    print(f"Warning: skipping unrecognized section: {line}")
                    line = f.readline().strip()
                    continue
            #f.readline()
            line = f.readline()
            if not line:
                break
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()

        f.close()
        
        # Initialize empty sections that might be needed later
        if "Bond Coeffs" not in sections and "bond types" in headers:
            sections["Bond Coeffs"] = []
        if "Bonds" not in sections and "bonds" in headers:
            sections["Bonds"] = []
            
        self.headers = headers
        self.sections = sections

    def write(self, filename):
        '''write out a LAMMPS data file (from data.py in Pizza.py)'''

        with open(filename, "w") as f:
            f.write(self.title + '\n')
            for keyword in hkeywords:
                if keyword in self.headers:
                    if keyword == "xlo xhi" or keyword == "ylo yhi" or \
                       keyword == "zlo zhi":
                        f.write("{0:f} {1:f} {2}\n".format(
                            self.headers[keyword][0],
                            self.headers[keyword][1], keyword))
                    elif keyword == "xy xz yz":
                        f.write("{0:f} {1:f} {2:f} {3}\n".format(
                            self.headers[keyword][0],
                            self.headers[keyword][1],
                            self.headers[keyword][2], keyword))
                    else:
                        f.write("{0:d} {1}\n".format(self.headers[keyword],
                                                   keyword))
            for pair in skeywords:
                keyword = pair[0]
                if keyword in self.sections:
                    f.write("\n{}\n\n".format(keyword))
                    for line in self.sections[keyword]:
                        f.write(line)

    def extract_nonpol(self):
        """extract atom and bond info from nonpolarizable data"""

        # extract atom IDs
        missinglabels = False
        for line in self.sections['Masses']:
            tok = line.split()
            atomtype = {}
            atomtype['id'] = int(tok[0])
            atomtype['m'] = float(tok[1])
            
            # Check if there's a type label in the comment
            if len(tok) < 4:
                if self.default_type:
                    # Use default type if provided
                    atomtype['type'] = self.default_type
                    print(f"Using default type '{self.default_type}' for atom ID {tok[0]}")
                else:
                    print("Warning: missing type for atom ID " + tok[0] +
                          " in Masses section")
                    missinglabels = True
                    atomtype['type'] = f"TYPE{tok[0]}"  # Default name based on ID
                    continue
            else:
                atomtype['type'] = tok[3]
                
            self.atomtypes.append(atomtype)

        if missinglabels and not self.default_type:
            print("Warning: some atom types have no labels. Using generated labels.")
            print("Consider using -t/--type flag to set a default type.")

        # extract atom registers
        for line in self.sections['Atoms']:
            tok = line.split()
            if len(tok) < 7:
                print(f"Warning: invalid atom line: {line}")
                continue
                
            atom = {}
            atom['n'] = int(tok[0])
            atom['mol'] = int(tok[1])
            atom['id'] = int(tok[2])
            atom['q'] = float(tok[3])
            atom['x'] = float(tok[4])
            atom['y'] = float(tok[5])
            atom['z'] = float(tok[6])
            #atom['note'] = ''.join([s + ' ' for s in tok[7:]]).strip()
            atom['note'] = ' '.join(tok[7:]) if len(tok) > 7 else ''
            self.atoms.append(atom)
            self.idmap[atom['n']] = atom

        if 'Velocities' in self.sections:
            for line in self.sections['Velocities']:
                tok = line.split()
                if len(tok) < 4:
                    print(f"Warning: invalid velocity line: {line}")
                    continue
                    
                atom_id = int(tok[0])
                if atom_id not in self.idmap:
                    print(f"Warning: velocity for unknown atom ID: {atom_id}")
                    continue
                    
                atom = self.idmap[atom_id]
                atom['vx'] = float(tok[1])
                atom['vy'] = float(tok[2])
                atom['vz'] = float(tok[3])

    def polarize(self, drude):
        """add Drude particles"""

        if 'Pair Coeffs' in self.sections:
            raise RuntimeError("cannot polarize a data with Pair Coeffs")

        self.extract_nonpol()

        # Initialize headers if missing
        if 'atoms' not in self.headers:
            self.headers['atoms'] = len(self.atoms)
            print(f"Warning: 'atoms' header not found. Setting to {len(self.atoms)}.")
            
        if 'bonds' not in self.headers:
            self.headers['bonds'] = 0
            print("Warning: 'bonds' header not found. Setting to 0.")
            
        if 'atom types' not in self.headers:
            self.headers['atom types'] = max(at['id'] for at in self.atomtypes)
            print(f"Warning: 'atom types' header not found. Setting to {self.headers['atom types']}.")
            
        if 'bond types' not in self.headers:
            self.headers['bond types'] = 0
            print("Warning: 'bond types' header not found. Setting to 0.")

        natom = self.headers['atoms']
        nbond = self.headers['bonds']
        nattype = self.headers['atom types']
        nbdtype = self.headers['bond types']

        # create new atom types (IDs) for Drude particles and modify cores
        newattypes = []
        for att in self.atomtypes:
            att['dflag'] = 'n'
            for ddt in drude.types:
                if ddt['type'] == att['type']:
                    nattype += 1
                    newid = {}
                    newid['id'] = ddt['id'] = nattype
                    newid['m'] = ddt['dm']
                    att['m'] -= ddt['dm']
                    # label drude particles and cores
                    att['dflag'] = 'c'
                    newid['dflag'] = 'd'
                    newid['type'] = att['type'] + ' DP'
                    att['type'] += ' DC'
                    ddt['type'] += ' DC'
                    newattypes.append(newid)
                    break

        self.headers['atom types'] += len(newattypes)
        self.sections['Masses'] = []
        for att in self.atomtypes + newattypes:
            self.sections['Masses'].append(massline(att))

        # create new bond types for core-drude bonds
        newbdtypes = []
        for att in self.atomtypes:
            for ddt in drude.types:
                if ddt['type'] == att['type']:
                    nbdtype += 1
                    newbdtype = {}
                    newbdtype['id'] = ddt['bdid'] = nbdtype
                    newbdtype['k'] = ddt['k']
                    newbdtype['r0'] = 0.0
                    newbdtype['note'] = '# ' + ddt['type'] + '-DP'
                    newbdtypes.append(newbdtype)
                    break

        self.headers['bond types'] += len(newbdtypes)
        
        # Create Bond Coeffs section if it doesn't exist
        if 'Bond Coeffs' not in self.sections:
            self.sections['Bond Coeffs'] = []
            
        for bdt in newbdtypes:
            self.sections['Bond Coeffs'].append(bdtline(bdt))

        # create new atoms for Drude particles and new bonds with their cores
        random.seed(123)
        newatoms = []
        newbonds = []
        for atom in self.atoms:
            atom['dflag'] = ''           # [c]ore, [d]rude, [n]on-polarizable
            atom['dd'] = 0               # partner drude or core
            for att in self.atomtypes:
                if att['id'] == atom['id']:
                    break
            for ddt in drude.types:
                if ddt['type'] == att['type']:
                    natom += 1
                    newatom = deepcopy(atom)
                    newatom['n'] = natom
                    self.idmap[natom] = newatom
                    newatom['id'] = ddt['id']
                    newatom['q'] = ddt['dq']
                    newatom['note'] = atom['note']
                    if '#' not in newatom['note']:
                        newatom['note'] += ' #'
                    newatom['note'] += ' DP'
                    newatom['dflag'] = 'd'
                    newatom['dd'] = atom['n']

                    # avoid superposition of cores and Drudes
                    newatom['x'] += 0.1 * (random.random() - 0.5)
                    newatom['y'] += 0.1 * (random.random() - 0.5)
                    newatom['z'] += 0.1 * (random.random() - 0.5)
                    if 'Velocities' in self.sections:
                        newatom['vx'] = atom['vx']
                        newatom['vy'] = atom['vy']
                        newatom['vz'] = atom['vz']

                    newatoms.append(newatom)
                    atom['q'] -= ddt['dq']
                    atom['dflag'] = 'c'
                    atom['dd'] = natom
                    if '#' not in atom['note']:
                        atom['note'] += ' #'
                    atom['note'] += ' DC'

                    nbond += 1
                    newbond = {}
                    newbond['n'] = nbond
                    newbond['id'] = ddt['bdid']
                    newbond['i'] = atom['n']
                    newbond['j'] = newatom['n']
                    newbond['note'] = '# ' + ddt['type'] + '-DP'
                    newbonds.append(newbond)
                    break

        self.headers['atoms'] += len(newatoms)
        self.headers['bonds'] += len(newbonds)
        self.sections['Atoms'] = []
        for atom in self.atoms + newatoms:
            self.sections['Atoms'].append(atomline(atom))
            
        if 'Bonds' not in self.sections:
            self.sections['Bonds'] = []
            
        for bond in newbonds:
            self.sections['Bonds'].append(bondline(bond))
            
        if 'Velocities' in self.sections:
            self.sections['Velocities'] = []
            for atom in self.atoms + newatoms:
                self.sections['Velocities'].append(velline(atom))

        # update list of atom IDs
        for att in newattypes:
            self.atomtypes.append(att)


    def extract_pol(self, drude):
        """extract atom, drude, bonds info from polarizable data"""

        # extract atom IDs
        for line in self.sections['Masses']:
            tok = line.split()
            atomtype = {}
            atomtype['id'] = int(tok[0])
            atomtype['m'] = float(tok[1])
            if len(tok) >= 4:
                atomtype['type'] = tok[3]
                atomtype['dflag'] = 'n'
                if tok[-1] == "DC":
                    atomtype['dflag'] = 'c'
                elif tok[-1] == "DP":
                    atomtype['dflag'] = 'd'
                print(atomtype['dflag'])
            else:
                if self.default_type:
                    atomtype['type'] = self.default_type
                    atomtype['dflag'] = 'n'
                else:
                    raise RuntimeError("comments in Masses section required "\
                                   "to identify cores (DC) and Drudes (DP)")
            self.atomtypes.append(atomtype)
                        
        # extract bond type data
        if 'Bond Coeffs' in self.sections:
            for line in self.sections['Bond Coeffs']:
                tok = line.split()
                bondtype = {}
                bondtype['id'] = int(tok[0])
                bondtype['k'] = float(tok[1])
                bondtype['r0'] = float(tok[2])
                bondtype['note'] = ''.join([s + ' ' for s in tok[3:]]).strip()
                self.bondtypes.append(bondtype)

        # extract atom registers
        for line in self.sections['Atoms']:
            tok = line.split()
            atom = {}
            atom['n'] = int(tok[0])
            atom['mol'] = int(tok[1])
            atom['id'] = int(tok[2])
            atom['q'] = float(tok[3])
            atom['x'] = float(tok[4])
            atom['y'] = float(tok[5])
            atom['z'] = float(tok[6])
            # atom['note'] = ''.join([s + ' ' for s in tok[7:-1]]).strip()
            if len(tok) > 7:
                if tok[-1] == 'DC':
                    atom['note'] = ' '.join(tok[7:-1])
                else:
                    atom['note'] = ' '.join(tok[7:])
            else:
                atom['note'] = ''
            self.atoms.append(atom)
            self.idmap[atom['n']] = atom

        if 'Velocities' in self.sections:
            for line in self.sections['Velocities']:
                tok = line.split()
                if int(tok[0]) in self.idmap:
                    atom = self.idmap[int(tok[0])]
                    atom['vx'] = float(tok[1])
                    atom['vy'] = float(tok[2])
                    atom['vz'] = float(tok[3])
                else:
                    print(f"Warning: velocity data for non-existent atom ID {tok[0]}")

        # extract bond data
        if 'Bonds' in self.sections:
            for line in self.sections['Bonds']:
                tok = line.split()
                bond = {}
                bond['n'] = int(tok[0])
                bond['id'] = int(tok[1])
                bond['i'] = int(tok[2])
                bond['j'] = int(tok[3])
                bond['note'] = ''.join([s + ' ' for s in tok[4:]]).strip()
                self.bonds.append(bond)


    def depolarize(self, drude):
        """remove Drude particles"""

        self.extract_pol(drude)

        atom_tp_map = {}
        bond_tp_map = {}
        atom_map = {}
        bond_map = {}
        q = {}
        atom_tp = {}
        m = {}

        for att in self.atomtypes:
            if att['dflag'] != 'd':
                atom_tp_map[att['id']] = len(atom_tp_map) + 1
            m[att['id']] = att['m']
        print(atom_tp_map)
        for atom in self.atoms:
            if atom['id'] in atom_tp_map:
                atom_map[atom['n']] = len(atom_map) + 1
            q[atom['n']] = atom['q']
            atom_tp[atom['n']] = atom['id']
        for bond in self.bonds:
            if bond['i'] in atom_map and bond['j'] in atom_map:
                bond_map[bond['n']] = len(bond_map) + 1
                if bond['id'] not in bond_tp_map:
                    bond_tp_map[bond['id']] = len(bond_tp_map) + 1
            else:
                if bond['i'] in atom_map:
                    q[bond['i']] += q[bond['j']]
                    if atom_tp[bond['j']] in m:
                        m[atom_tp[bond['i']]] += m.pop(atom_tp[bond['j']])
                else:
                    q[bond['j']] += q[bond['i']]
                    if atom_tp[bond['i']] in m:
                        m[atom_tp[bond['j']]] += m.pop(atom_tp[bond['i']])

        size = len(self.atomtypes)
        for iatom_tp in reversed(range(size)):
            att = self.atomtypes[iatom_tp]
            if att['id'] not in atom_tp_map:
                del self.atomtypes[iatom_tp]
            else:
                att['m'] = m[att['id']]
                att['id'] = atom_tp_map[att['id']]

        size = len(self.bondtypes)
        for ibond_tp in reversed(range(size)):
            bdt = self.bondtypes[ibond_tp]
            if bdt['id'] not in bond_tp_map:
                del self.bondtypes[ibond_tp]
            else:
                bdt['id'] = bond_tp_map[bdt['id']]

        size = len(self.atoms)
        for iatom in reversed(range(size)):
            atom = self.atoms[iatom]
            if atom['n'] not in atom_map:
                del self.atoms[iatom]
            else:
                atom['q'] = q[atom['n']]
                atom['n'] = atom_map[atom['n']]
                atom['id'] = atom_tp_map[atom['id']]

        size = len(self.bonds)
        for ibond in reversed(range(size)):
            bond = self.bonds[ibond]
            if bond['n'] not in bond_map:
                del self.bonds[ibond]
            else:
                bond['n'] = bond_map[bond['n']]
                bond['id'] = bond_tp_map[bond['id']]
                bond['i'] = atom_map[bond['i']]
                bond['j'] = atom_map[bond['j']]

        self.sections['Atoms'] = []
        for atom in self.atoms:
            self.sections['Atoms'].append(atomline(atom))
        self.headers['atoms'] = len(self.atoms)
        self.sections['Masses'] = []
        for att in self.atomtypes:
            self.sections['Masses'].append(massline(att))
        self.headers['atom types'] = len(self.atomtypes)
        self.sections['Bonds'] = []
        for bond in self.bonds:
            self.sections['Bonds'].append(bondline(bond))
        self.headers['bonds'] = len(self.bonds)
        self.sections['Bond Coeffs'] = []
        for bdt in self.bondtypes:
            self.sections['Bond Coeffs'].append(bdtline(bdt))
        self.headers['bond types'] = len(self.bondtypes)

        if 'Velocities' in self.sections:
            self.sections['Velocities'] = []
            for atom in self.atoms:
                self.sections['Velocities'].append(velline(atom))

    def lmpscript(self, drude, outfile, thole = 2.6, cutoff = 12.0):
        """print lines for input script, including pair_style thole"""

        pairfile = "pair-drude.lmp"
        
        dfound = False
        for att in self.atomtypes:
            if att['dflag'] == 'd':
                dfound = True
                break
        if not dfound:
            print("# No polarizable atoms found.")
            return

        print("# Commands to include in the LAMMPS input script\n")

        print("# adapt the pair_style command as needed")
        print("pair_style hybrid/overlay ... coul/long/cs {0:.1f} "\
              "thole {1:.3f} {0:.1f}\n".format(cutoff, thole))

        print("# data file with Drude oscillators added")
        print("read_data {0}\n".format(outfile))

        print("# pair interactions with Drude particles written to file")
        print("# Thole damping recommended if more than 1 Drude per molecule")
        print("include {0}\n".format(pairfile))

        with open(pairfile, "w") as f:
            f.write("# interactions involving Drude particles\n")
            f.write("pair_coeff    * {0:3d}* coul/long/cs\n".format(att['id']))

            f.write("# Thole damping if more than 1 Drude per molecule\n")
            # Thole parameters for I,J pairs
            ifound = False
            for atti in self.atomtypes:
                itype = atti['type'].split()[0]
                for ddt in drude.types:
                    dtype = ddt['type'].split()[0]
                    if dtype == itype:
                        alphai = ddt['alpha']
                        tholei = ddt['thole']
                        ifound = True
                        break
                jfound = False
                for attj in self.atomtypes:
                    if attj['id'] < atti['id']:
                        continue
                    jtype = attj['type'].split()[0]
                    for ddt in drude.types:
                        dtype = ddt['type'].split()[0]
                        if dtype == jtype:
                            alphaj = ddt['alpha']
                            tholej = ddt['thole']
                            jfound = True
                            break
                    if ifound and jfound:
                        alphaij = (alphai * alphaj)**0.5
                        tholeij = (tholei + tholej) / 2.0
                        if tholeij == thole:
                            f.write("pair_coeff {0:4} {1:4} thole "\
                                  "{2:7.3f}\n".format(atti['id'], attj['id'],
                                                      alphaij))
                        else:
                            f.write("pair_coeff {0:4} {1:4} thole {2:7.3f} "\
                                  "{3:7.3f}\n".format(atti['id'],attj['id'],
                                                        alphaij, tholeij))
                    jfound = False
                ifound = False

        print("# atom groups convenient for thermostats (see package "
              "documentation), etc.")
        gatoms = gcores = gdrudes = ""
        for att in self.atomtypes:
            if att['dflag'] != 'd':
                gatoms += " {0}".format(att['id'])
            if att['dflag'] == 'c':
                gcores += " {0}".format(att['id'])
            if att['dflag'] == 'd':
                gdrudes += " {0}".format(att['id'])
        print("group ATOMS type" + gatoms)
        print("group CORES type" + gcores)
        print("group DRUDES type" + gdrudes)
        print("")

        print("# flag for each atom type: [C]ore, [D]rude, [N]on-polarizable")
        drudetypes = ""
        for att in self.atomtypes:
            drudetypes += " {0}".format(att['dflag'].upper())
        print("fix DRUDE all drude" + drudetypes)
        print("")

        print("# ATTENTION!")
        print("#  * read_data may need 'extra/special/per/atom' keyword, "
              "LAMMPS will exit with a message.")
        print("#  * If using fix shake the group-ID must not include "
              "Drude particles.")
        print("#    Use group ATOMS for example.")
        print("#  * Give all I<=J pair interactions, no mixing.")
        print("#  * Pair style coul/long/cs from CORESHELL package is used "\
              "for interactions")
        print("#    of Drude particles. Alternatively pair lj/cut/thole/long "\
              "could be used,")
        print("#    avoiding hybrid/overlay and allowing mixing. See doc "\
              "pages.")


# --------------------------------------

def count_atoms_in_section(file_obj):
    """Count the number of atoms in the Atoms section by parsing it"""
    line_count = 0
    # Skip header line
    file_obj.readline()
    
    # Count atom lines until we hit a blank line or a new section
    while True:
        pos = file_obj.tell()
        line = file_obj.readline()
        if not line or line.strip() == "":
            break
        if any(line.strip() == kw[0] for kw in skeywords):
            file_obj.seek(pos)  # Move back before section header
            break
        line_count += 1
            
    return line_count

kcal =  4.184                           # kJ
eV   = 96.485                           # kJ/mol
fpe0 =  0.000719756                     # (4 Pi eps0) in e^2/(kJ/mol A)


class Drude(object):
    """specification of drude oscillator types"""

    def __init__(self, drudefile, polar = '', positive = False, metal = False):
        self.types = []
        with open(drudefile, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    continue
                tok = line.split()
                drude = {}
                drude['type'] = tok[0]
                drude['dm'] = float(tok[1])
                dq = float(tok[2])
                k = float(tok[3])
                drude['alpha'] = alpha = float(tok[4])
                drude['thole'] = float(tok[5])

                if polar == 'q':
                    dq = (fpe0 * k * alpha)**0.5
                elif polar == 'k':
                    k = dq*dq / (fpe0 * alpha)

                if positive:
                    drude['dq'] = abs(dq)
                else:
                    drude['dq'] = -abs(dq)

                if metal:
                    drude['k'] = k / (2.0 * eV)
                else:
                    drude['k'] = k / (2.0 * kcal)

                self.types.append(drude)


# --------------------------------------

def main():
    parser = argparse.ArgumentParser(description = usage,
             formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--ffdrude', default = 'drude.dff',
                        help = 'Drude parameter file (default: drude.dff)')
    parser.add_argument('-t', '--type', 
                        help = 'Default atom type if missing in Masses section')
    parser.add_argument('--thole', type = float, default = 2.6,
                        help = 'Thole damping parameter (default: 2.6)')
    parser.add_argument('-c', '--cutoff', type = float, default = 12.0,
                        help = 'distance cutoff/A (default: 12.0)')
    parser.add_argument('-q', '--qcalc', action = 'store_true',
                        help = 'Drude charges calculated from polarisability '\
                        '(default: q value from parameter file)')
    parser.add_argument('-k', '--kcalc', action = 'store_true',
                        help = 'Drude force constants calculated from '\
                        'polarisability (default: k value from parameter file)')
    parser.add_argument('-p', '--positive', action = 'store_true',
                        help = 'Drude particles have positive charge '\
                        '(default: negative charge)')
    parser.add_argument('-m', '--metal', action = 'store_true',
                        help = 'LAMMPS metal units (default: real units)')
    parser.add_argument('-d', '--depolarize', action = 'store_true',
                        help = 'remove Drude dipole polarization from '\
                        'LAMMPS data file')
    parser.add_argument('infile', help = 'input LAMMPS data file')
    parser.add_argument('outfile', help = 'output LAMMPS data file')
    args = parser.parse_args()

    if args.qcalc:
        polar = 'q'
    elif args.kcalc:
        polar = 'p'
    else:
        polar = ''

    data = Data(args.infile, args.type)
    drude = Drude(args.ffdrude, polar, args.positive, args.metal)
    if not args.depolarize:
        data.polarize(drude)
        data.lmpscript(drude, args.outfile, args.thole, args.cutoff)
    else:
        data.depolarize(drude)
    data.write(args.outfile)

if __name__ == '__main__':
    main()
