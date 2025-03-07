import numpy as np
import matplotlib.pyplot as plt
import sys
from copy import deepcopy

# Original Data class and related functions
class Data(object):
    def __init__(self, datafile):
        """read LAMMPS data file"""
        self.atomtypes = []
        self.bondtypes = []
        self.atoms = []
        self.bonds = []
        self.idmap = {}
        self.nselect = 1
        self.charge_map = {}  # New mapping for atom IDs to charges

        f = open(datafile, "r")
        self.title = f.readline()
        self.names = {}

        headers = {}
        while 1:
            line = f.readline().strip()
            if "#" in line:
                line = line[: line.index("#")].strip()
            if len(line) == 0:
                continue
            found = 0
            for keyword in hkeywords:
                if keyword in line:
                    found = 1
                    words = line.split()
                    if (
                        keyword == "xlo xhi"
                        or keyword == "ylo yhi"
                        or keyword == "zlo zhi"
                    ):
                        headers[keyword] = (float(words[0]), float(words[1]))
                    elif keyword == "xy xz yz":
                        headers[keyword] = (
                            float(words[0]),
                            float(words[1]),
                            float(words[2]),
                        )
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
                            raise RuntimeError(
                                "data section {} "
                                "has no matching header value".format(line)
                            )
                        f.readline()
                        list_ = []
                        for _ in range(headers[length]):
                            list_.append(f.readline())
                        sections[keyword] = list_
                if not found:
                    raise RuntimeError(
                        "invalid section {} in data" " file".format(line)
                    )
            line = f.readline()
            if not line:
                break
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()

        f.close()
        self.headers = headers
        self.sections = sections

    def extract_pol(self):
        """extract atom, drude, bonds info from polarizable data"""
        # extract atom IDs
        for line in self.sections["Masses"]:
            tok = line.split()
            atomtype = {}
            atomtype["id"] = int(tok[0])
            atomtype["m"] = float(tok[1])
            if len(tok) >= 4:
                atomtype["type"] = tok[3]
                atomtype["dflag"] = "n"
                if tok[-1] == "DC":
                    atomtype["dflag"] = "c"
                elif tok[-1] == "DP":
                    atomtype["dflag"] = "d"
            else:
                raise RuntimeError(
                    "comments in Masses section required "
                    "to identify cores (DC) and Drudes (DP)"
                )
            self.atomtypes.append(atomtype)

        # extract bond type data
        for line in self.sections["Bond Coeffs"]:
            tok = line.split()
            bondtype = {}
            bondtype["id"] = int(tok[0])
            bondtype["k"] = float(tok[1])
            bondtype["r0"] = float(tok[2])
            bondtype["note"] = "".join([s + " " for s in tok[3:]]).strip()
            self.bondtypes.append(bondtype)

        # extract atom registers
        for line in self.sections["Atoms"]:
            tok = line.split()
            atom = {}
            atom["n"] = int(tok[0])
            atom["mol"] = int(tok[1])
            atom["id"] = int(tok[2])
            atom["q"] = float(tok[3])
            atom["x"] = float(tok[4])
            atom["y"] = float(tok[5])
            atom["z"] = float(tok[6])
            atom["xu"] = int(tok[7])
            atom["yu"] = int(tok[8])
            atom["zu"] = int(tok[9])
            if tok[-1] == "DC":
                atom["note"] = " ".join(tok[7:-1])
            else:
                atom["note"] = " ".join(tok[7:])
            self.atoms.append(atom)
            print(atom)
            self.idmap[atom["n"]] = atom
            
            # Create mapping from atom ID to charge
            self.charge_map[atom["n"]] = atom["q"]

        # extract bond data
        for line in self.sections["Bonds"]:
            tok = line.split()
            bond = {}
            bond["n"] = int(tok[0])
            bond["id"] = int(tok[1])
            bond["i"] = int(tok[2])
            bond["j"] = int(tok[3])
            bond["note"] = "".join([s + " " for s in tok[4:]]).strip()
            self.bonds.append(bond)

# Constants from your original code
hkeywords = [
    "atoms", "ellipsoids", "lines", "triangles", "bodies",
    "bonds", "angles", "dihedrals", "impropers",
    "atom types", "bond types", "angle types",
    "dihedral types", "improper types",
    "xlo xhi", "ylo yhi", "zlo zhi", "xy xz yz",
]

skeywords = [
    ["Masses", "atom types"],
    ["Pair Coeffs", "atom types"],
    ["Bond Coeffs", "bond types"],
    ["Angle Coeffs", "angle types"],
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
    ["Atoms", "atoms"],
    ["Velocities", "atoms"],
    ["Ellipsoids", "ellipsoids"],
    ["Lines", "lines"],
    ["Triangles", "triangles"],
    ["Bodies", "bodies"],
    ["Bonds", "bonds"],
    ["Angles", "angles"],
    ["Dihedrals", "dihedrals"],
    ["Impropers", "impropers"],
    ["Molecules", "atoms"],
]

class Box:
    def __init__(self, box_bounds):
        """Initialize simulation box from bounds"""
        self.bounds = np.array(box_bounds)
        self.lengths = self.bounds[:, 1] - self.bounds[:, 0]

    def minimum_image(self, r1, r2):
        """Apply minimum image convention using numpy vectorized operations"""
        dr = r2 - r1
        # Apply periodic boundary conditions in one vectorized step
        dr = dr - np.round(dr / self.lengths) * self.lengths
        return dr

class BondDipole:
    def __init__(self, id1, id2, q1, q2, r1, r2):
        self.id1 = id1
        self.id2 = id2
        self.q1 = q1
        self.q2 = q2
        self.r1 = np.array(r1)
        self.r2 = np.array(r2)

    def calculate_dipole(self, box):
        """Calculate dipole moment considering PBC"""
        # dr = box.minimum_image(self.r1, self.r2)

        dipole_moment = self.q2 * self.r2 + self.q1 * self.r1 # big
        # dipole_moment = self.q2 * dr # small

        if np.linalg.norm(dipole_moment) > 10:
            print(np.linalg.norm(dipole_moment))
        return dipole_moment

class Timestep:
    def __init__(self, timestep, box_bounds):
        self.timestep = timestep
        self.box = Box(box_bounds)
        self.atoms = {}
        self.bond_dipoles = []
        self.dipole_moment = None
        self.dipole_magnitude = None

    def add_atom(self, id, mol, type, q, x, y, z):
        self.atoms[id] = {
            'mol': mol,
            'type': type,
            'q': q,
            'pos': np.array([x, y, z])
        }

    def process_bonds(self, bonds):
        """Process bonds to create bond dipoles"""
        self.bond_dipoles = []
        for bond in bonds:
            id1, id2 = bond['i'], bond['j']
            if id1 in self.atoms and id2 in self.atoms:
                atom1 = self.atoms[id1]
                atom2 = self.atoms[id2]
                bond_dipole = BondDipole(
                    id1, id2,
                    atom1['q'], atom2['q'],
                    atom1['pos'], atom2['pos']
                )
                self.bond_dipoles.append(bond_dipole)

    def calculate_total_dipole(self):
        """Calculate total dipole moment from all bond dipoles"""
        if not self.bond_dipoles:
            return np.zeros(3), 0.0

        total_dipole = np.zeros(3)
        for bond_dipole in self.bond_dipoles:
            total_dipole += bond_dipole.calculate_dipole(self.box)

        self.dipole_moment = total_dipole
        self.dipole_magnitude = np.linalg.norm(total_dipole)
        return self.dipole_moment, self.dipole_magnitude

class DumpAnalyzer:
    def __init__(self, dump_file, data_file):
        self.dump_file = dump_file
        self.data = data_file
        self.timesteps = []
        self.bonds = data_file.bonds
        self.charge_map = data_file.charge_map  # Get charge mapping from data file
        self.read_dump()

    def read_dump(self):
        try:
            with open(self.dump_file, 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break

                    if "ITEM: TIMESTEP" not in line:
                        continue

                    timestep_num = int(f.readline().strip())
                    
                    f.readline()  # Skip "ITEM: NUMBER OF ATOMS"
                    num_atoms = int(f.readline().strip())

                    f.readline()  # Skip "ITEM: BOX BOUNDS"
                    box_bounds = []
                    for _ in range(3):
                        lo, hi = map(float, f.readline().split())
                        box_bounds.append([lo, hi])

                    ts = Timestep(timestep_num, box_bounds)

                    f.readline()  # Skip "ITEM: ATOMS"

                    for _ in range(num_atoms):
                        tokens = f.readline().split()
                        atom_id = int(tokens[0])
                        mol_id = int(tokens[1])
                        atom_type = int(tokens[2])
                        
                        # Use charge from data file instead of hardcoded value
                        q = self.charge_map.get(atom_id, 0.0)  # Default to 0 if not found
                        
                        x, y, z = map(float, tokens[4:7])
                        ts.add_atom(atom_id, mol_id, atom_type, q, x, y, z)

                    ts.process_bonds(self.bonds)
                    ts.calculate_total_dipole()
                    self.timesteps.append(ts)
        except FileNotFoundError:
            print(f"Error: Could not find dump file '{self.dump_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing dump file: {e}")
            sys.exit(1)

    def plot_dipole_analysis(self):
        """Create plots of dipole evolution"""
        if not self.timesteps:
            print("No timesteps found to analyze")
            return
            
        timestamps = [ts.timestep for ts in self.timesteps]
        magnitudes = [ts.dipole_magnitude for ts in self.timesteps]
        components = np.array([ts.dipole_moment for ts in self.timesteps])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(timestamps, magnitudes, '-o', label='Total magnitude')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Dipole magnitude')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(timestamps, components[:, 0], '-o', label='x')
        ax2.plot(timestamps, components[:, 1], '-o', label='y')
        ax2.plot(timestamps, components[:, 2], '-o', label='z')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Dipole components')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('dipole_analysis.png')
        
        print("\nDipole Analysis Statistics:")
        print(f"Average magnitude: {np.mean(magnitudes):.4f}")
        print(f"Median magnitude: {np.median(magnitudes):.4f}")
        print(f"Maximum magnitude: {np.max(magnitudes):.4f}")
        print(f"Minimum magnitude: {np.min(magnitudes):.4f}")
        print("\nComponent averages:")
        print(f"X: {np.mean(components[:, 0]):.4f}")
        print(f"Y: {np.mean(components[:, 1]):.4f}")
        print(f"Z: {np.mean(components[:, 2]):.4f}")

def main():
    if len(sys.argv) > 2:
        data_file = sys.argv[1]
        dump_file = sys.argv[2]
    else:
        data_file = "cool_stuff.data"
        dump_file = "dump.lammpstrj"
        print(f"Using default filenames: {data_file} and {dump_file}")
        print("Use: python script.py [data_file] [dump_file] to specify file names")
    
    try:
        data = Data(data_file)
        data.extract_pol()
        analyzer = DumpAnalyzer(dump_file, data)
        analyzer.plot_dipole_analysis()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
