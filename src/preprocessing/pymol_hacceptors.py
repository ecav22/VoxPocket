import sys

input_filename = sys.argv[3]

cmd.load(input_filename, 'molecule_name')


# Define a selection for nitrogen, oxygen, and sulfur atoms (potential hydrogen bond acceptors)
cmd.select('hbd_acceptors', 'resn N+O+S')

# Define a selection for hydrogen bond acceptors (atoms with lone pairs)
#atoms with lone pairs, such as nitrogen, oxygen, and sulfur.
cmd.select('acceptors', 'hbd_acceptors and (elem N+O+S) and not (hydro and neighbor (resn N+O+S))')


acceptor_list = []
cmd.iterate('acceptors', 'acceptor_list.append(name)', space=locals())
print("Hydrogen Bond Acceptor list:", acceptor_list)

# Iterate over the selected atoms and print their indices
acceptor_indices = []
cmd.iterate('acceptors', 'acceptor_indices.append(index)', space=locals())

# Print the indices of hydrogen bond acceptors
print("Hydrogen Bond Acceptor Indices:", acceptor_indices)


with open(input_filename.rstrip(".pdb")+'d_hbacceptors.txt', 'w') as f:
    for hba in acceptor_indices:
        f.write(str(hba) + '\n')

cmd.quit()

