from pymol import cmd

def identify_missing_residues(pdb_file):
    cmd.load(pdb_file, "structure")
    cmd.h_add("structure")
    
    missing_residues = set()
    
    # Iterate through all atoms to identify missing residues
    cmd.iterate_state(1, "structure", "missing_residues.add((chain, resi))", space=locals())

    return sorted(list(missing_residues))

pdb_file = "1qyg_protein_cleaned.pdb"
missing_residues = identify_missing_residues(pdb_file)
print("Missing Residues:", missing_residues)
