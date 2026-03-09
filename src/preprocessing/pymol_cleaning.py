# remove heteroatoms, hydrogens and readd hydrogens
import sys

input_filename = sys.argv[3]

cmd.load(input_filename)
cmd.remove('hetatm')
cmd.remove('hydrogens')
cmd.h_add()
cmd.alter('protein', 'resn = "ALA" if resn == "NILE" else resn') #probably not necessary
cmd.save(input_filename.replace('.pdb', '_cleaned.pdb'))
cmd.quit()