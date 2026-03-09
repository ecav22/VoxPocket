import sys

input_filename = sys.argv[3]

cmd.load(input_filename, 'molecule_name')

# Select hydrogen bond donors
cmd.select('donors', 'hydro and polymer and chain *')


# Iterate over the selected atoms and print their names
donor_list = []
cmd.iterate('donors', 'donor_list.append(name)', space=locals())


#print("Hydrogen Bond Donors:", donor_list)


donor_indices = []
cmd.iterate('donors', 'donor_indices.append(index)', space=locals())

#print("Hydrogen Bond Donor Indices:", donor_indices)

with open(input_filename.rstrip(".pdb")+'d_hbdonors.txt', 'w') as f:
    for hbd in donor_indices:
        f.write(str(hbd) + '\n')

cmd.quit()