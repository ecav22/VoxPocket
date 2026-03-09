# Save this script as extract_bfactors.py

import sys

input_filename = sys.argv[3]
# Load PDB file
cmd.load(input_filename, 'molecule_name')
cmd.remove('hetatm')
# Extract B-factors
bfactors = []
cmd.iterate_state(1, 'molecule_name', 'bfactors.append(b)')

# Print B-factors
#print(bfactors)
#print(len(bfactors))

# Save B-factors to a file (optional)
with open(input_filename.rstrip(".pdb")+'d_bfactors.txt', 'w') as f:
    for bfactor in bfactors:
        f.write(str(bfactor) + '\n')

# Exit PyMOLa
cmd.quit()