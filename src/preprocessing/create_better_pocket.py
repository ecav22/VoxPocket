from __future__ import division
import os, numpy, mdtraj, scipy.stats
import matplotlib.pyplot as plt
from pathlib import Path


from rdkit import Chem
from rdkit.Chem import AllChem



silence = " > /dev/null 2>&1"
CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "config/checkpoints/checkpoint_tensor_better_pocket.txt"

#pqr and pdb files are slightly different because of Amber ff format etc.... and sometimes different number of atoms
#have to re-obtain all the coordinates and atom names to re-map the charges and radiuses
'''
def open_pqr(path_to_file):

	f= open(path_to_file)
	lines=f.readlines()
	f.close()

	x_pqr=[]
	y_pqr=[]
	z_pqr=[]

	charge=[]
	radius = []
	#the last 2 columns of a .pqr file are charge (electron units) and radius (angstrom)
	for i in lines:
		if i.startswith("ATOM") == True:

			#x_pqr.append(float(i.split()[-5]))
			#y_pqr.append(float(i.split()[-4]))
			#z_pqr.append(float(i.split()[-3]))

			x_pqr.append(float(i[30:38]))
			y_pqr.append(float(i[38:46]))
			z_pqr.append(float(i[46:54]))

			#print(x_pqr, y_pqr, z_pqr)
			#print(y_pqr)
			#print(z_pqr)
			

			charge.append(float(i.split()[-2]))
			radius.append(float(i.split()[-1]))

	xyz_pqr=numpy.column_stack((x_pqr, y_pqr, z_pqr))

	return xyz_pqr, charge, radius

'''

def get_new_pocket_indices(protein_coord, ligand_coord, cutoff):

	new_=[]

	for i in protein_coord:

		for j in ligand_coord:

			dist = numpy.linalg.norm(i - j) #euclidean distance

			if dist <= cutoff:
				new_.append(list(i))

	new_pocket = []
	#print(len(new_))
	#remove duplicates
	for x in new_:
		if x not in new_pocket:
			new_pocket.append(x)
	#[new_pocket.append(x) for x in new_ if x not in new_pocket] #remove duplicates

	#print(len(new_pocket))

	return new_pocket



os.system(f"touch {CHECKPOINT_PATH}")
chck=open(CHECKPOINT_PATH, "r")
data_ck=chck.readlines()
chck.close()

check=open(CHECKPOINT_PATH, "a")

n=len(data_ck)
print(n)

#path_folder = "10gs/"
nbins=32 #number of bins of the 3D tensor (go for 50 or 100)

nn=len(data_ck)
folders = os.listdir("refined-set/")

for f in folders:
	
	for p in ["_protein_"]:

		print(f)
		path_folder="refined-set/"+f+"/"
		#os.system("rm -r "+path_folder+"for_labview"+p+"/")
		#continue

		conf_path =path_folder+f+p+"cleaned.pdb"
		ligand_path=path_folder+f+"_ligand.mol2"

		

		exist=False
		for a in data_ck:
			if conf_path in a:
				exist=True

		if exist==True:
			continue

		ligand_pdb_path=path_folder+f+"_ligand.pdb"

		mol = Chem.MolFromMol2File(ligand_path)
		AllChem.MolToPDBFile(mol, ligand_pdb_path)
		#os.system("rm -r "+path_folder+"for_labview"+p+"/"+silence)
		#os.system("mkdir "+path_folder+"for_labview"+p+"/")

		conf = mdtraj.load(conf_path)
		ligand = mdtraj.load(ligand_pdb_path)

		print(conf_path)
		print(conf.n_atoms, " atoms")
		print(conf.n_residues, " residues")
		
		#
		xyz = conf.xyz[0]
		xyz_ligand = ligand.xyz[0]
		#bfactors = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_bfactors.txt")
		#buriedness = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_buriedness.txt")
		#dssp = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_dssp.txt", dtype='str')
		#hbac = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_hbacceptors.txt")
		#hbdon =numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_hbdonors.txt")
		#sasa = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_sasa.txt")
		#xyz_pqr, charge, radius = open_pqr(conf_path.rstrip("pdb")+"pqr")
		#print(xyz_pqr)

		#small = 0.001 #nm

		#min_x = numpy.min(xyz[:,0])-small
		#max_x = numpy.max(xyz[:,0])+small
		#dist_x= max_x - min_x

		#min_y = numpy.min(xyz[:,1])-small
		#max_y = numpy.max(xyz[:,1])+small
		#dist_y= max_y - min_y

		#min_z = numpy.min(xyz[:,2])-small
		#max_z = numpy.max(xyz[:,2])+small
		#dist_z= max_z - min_z

		#max_dist = numpy.max([dist_x, dist_y, dist_z])
		#bin_size = max_dist/nbins

		#x_bins = numpy.linspace(min_x, min_x+max_dist, nbins+1)
		#y_bins = numpy.linspace(min_y, min_y+max_dist, nbins+1)
		#z_bins = numpy.linspace(min_z, min_z+max_dist, nbins+1)
		#print(x_bins, max_x, min_x)
		#print(y_bins, max_y, min_y)
		#print(z_bins, max_z, min_z)
		#print(len(x_bins))


		#NOTE: for the pocket, I need the axis min and max of the whole protein, no rebinning of the pocket
		#size of the box
		#if p=="_protein_":
			#axis_min = numpy.min(xyz)
			#axis_max = numpy.max(xyz)
		#	axis_bins =[x_bins, y_bins, z_bins]
			
		#	numpy.savetxt(path_folder+"for_labview"+p+"/axis_bins.txt", axis_bins, fmt="%1.8f")

		#if p=="_pocket_":
		axis_bins = numpy.loadtxt(path_folder+"for_labview_protein_/axis_bins.txt")

		x_bins = axis_bins[0]
		y_bins = axis_bins[1]
		z_bins = axis_bins[2]

		#x_bins = numpy.linspace(min_x, max_x, nbins)
		#y_bins = numpy.linspace(min_y, max_y, nbins)
		#z_bins = numpy.linspace(min_z, max_z, nbins)

		

		N_ligand_tensor = numpy.zeros((nbins,nbins,nbins))
		#xyz_tensor = numpy.zeros(nbins,nbins,nbins) #this is just the N_tensor
		#bfactors_tensor = numpy.zeros((nbins,nbins,nbins))
		#buriedness_tensor = numpy.zeros((nbins,nbins,nbins))
		#dssp_tensor = numpy.zeros((nbins,nbins,nbins)) #not very useful tbh
		#hbac_tensor = numpy.zeros((nbins,nbins,nbins))
		#hbdon_tensor = numpy.zeros((nbins,nbins,nbins))
		#sasa_tensor = numpy.zeros((nbins,nbins,nbins))
		#charge_tensor = numpy.zeros((nbins,nbins,nbins))
		#radius_tensor = numpy.zeros((nbins,nbins,nbins))

		cutoff=0.45 #6 angstrom
		pocket_coordinates = get_new_pocket_indices(xyz, xyz_ligand, cutoff)

		bin_indices=[]

		for coord in pocket_coordinates:
			#print(coord)
			#print(x_bins)
			c_x = coord[0]
			c_y = coord[1]
			c_z = coord[2]

			done=False

			for x in range(len(x_bins)-1):
				if ((c_x >= x_bins[x]) and (c_x <= x_bins[x+1])):
					a=x
					break


			for y in range(len(y_bins)-1):
				if ((c_y >= y_bins[y]) and (c_y <= y_bins[y+1])):
					b=y
					break


			for z in range(len(z_bins)-1):
				if ((c_z >= z_bins[z]) and (c_z <= z_bins[z+1])):
					c=z
					break

			bin_indices.append([a,b,c])




		#print(bin_indices)

		#bin_indices = numpy.digitize(xyz, axis) - 1 #note the -1 is because the output of digitize is 1-based (starts from 1) and I want it 0-based
		#this is because the value 0 is reserved for the values outside of the bins



		for row in range(len(bin_indices)):

			l=bin_indices[row]

			i=l[0]
			j=l[1]
			k=l[2]

			N_ligand_tensor[i, j, k] += 1
			#bfactors_tensor[i, j, k] += bfactors[row]
			#buriedness_tensor[i, j, k] += buriedness[row]
			#sasa_tensor[i, j, k] += sasa[row]

		
		print("bin size:", x_bins[1]-x_bins[0], "nm")
		#print(nbins**3, "total voxels")
		fil=len(numpy.nonzero(N_ligand_tensor)[0])
		#print(fil, "filled voxels")
		emp=len(numpy.nonzero(N_ligand_tensor==0)[0])
		#print(emp, "empty voxels")
		print( fil/(nbins**3)*100, "percent occupied volume")
			
		#print(sasa_tensor)

		#bfactors_tensor = bfactors_tensor/N_tensor #average bfactor of the bins
		#bfactors_tensor = numpy.nan_to_num(bfactors_tensor) #remove nans (change to 0)
		#buriedness_tensor = buriedness_tensor/N_tensor #average in the bins (aka average number of neighbors)
		#buriedness_tensor = numpy.nan_to_num(buriedness_tensor)
		#sasa_tensor = sasa_tensor/N_tensor #average sasa of the bin
		#sasa_tensor = numpy.nan_to_num(sasa_tensor)


		#for g in hbac: #count the hbdon#

		#	l=bin_indices[int(g)]

		#	i=l[0]
		#	j=l[1]
		#	k=l[2]

		#	hbac_tensor[i,j,k] += 1


		#for g in hbdon: #count the hbac

		#	l=bin_indices[int(g)]

		#	i=l[0]
		#	j=l[1]
		#	k=l[2]

		#	hbdon_tensor[i,j,k]+= 1

		#hbac_tensor=hbac_tensor #no average, just sum
		#hbdon_tensor=hbdon_tensor #no average, just sum




		#### outouts for LabVIEW
		
		
		numpy.savetxt(path_folder+"for_labview_pocket_/xyz_new_pocket.txt", pocket_coordinates, fmt="%1.8f")
		#numpy.savetxt(path_folder+"for_labview"+p+"/bfactors.txt", bfactors, fmt="%1.8f")
		#numpy.savetxt(path_folder+"for_labview"+p+"/buriedness.txt", buriedness, fmt="%i")
		#numpy.savetxt(path_folder+"for_labview"+p+"/sasa.txt", sasa, fmt="%1.8f")


		numpy.save(path_folder+"for_labview_pocket_/N_tensor_new_pocket.npy", N_ligand_tensor)
		#numpy.save(path_folder+"for_labview"+p+"/bfactors_tensor.npy", bfactors_tensor)
		#numpy.save(path_folder+"for_labview"+p+"/buriedness_tensor.npy", buriedness_tensor)
		#numpy.save(path_folder+"for_labview"+p+"/sasa_tensor.npy", sasa_tensor)

		#numpy.save(path_folder+"for_labview"+p+"/hbac_tensor.npy", hbac_tensor)
		#numpy.save(path_folder+"for_labview"+p+"/hbdon_tensor.npy", hbdon_tensor)

		
		#hbac_list=[]
		#for i in range(len(buriedness)):
		#	if i in hbac:
		#		hbac_list.append(1)
		#	else:
		#		hbac_list.append(0)

		#hbdon_list=[]
		#for i in range(len(buriedness)):
		#	if i in hbdon:
		#		hbdon_list.append(1)
		#	else:
		#		hbdon_list.append(0)

		#hbac_list=numpy.array(hbac_list)
		#hbdon_list=numpy.array(hbdon_list)

		#numpy.savetxt(path_folder+"for_labview"+p+"/hbac.txt", hbac_list, fmt="%i")
		#numpy.savetxt(path_folder+"for_labview"+p+"/hbdon.txt", hbdon_list, fmt="%i")


		#xyz_pqr = xyz_pqr/10
		#### have to compute charge and radius differently, using the coordinates from the pqr because pdb2pqr added a few hydrogens
		#bin_indices_pqr = numpy.digitize(xyz_pqr, axis) - 1
		#N_tensor_pqr = numpy.zeros((nbins,nbins,nbins))

		#bin_indices_pqr=[]

		#for coord in xyz_pqr:
		#	c_x = coord[0]
		#	c_y = coord[1]
		#	c_z = coord[2]

		#	done=False

		#	for x in range(len(x_bins)-1):
		#		if ((c_x >= x_bins[x]) and (c_x <= x_bins[x+1])):
		#			a=x
		#			break


		#	for y in range(len(y_bins)-1):
		#		if ((c_y >= y_bins[y]) and (c_y <= y_bins[y+1])):
		#			b=y
		#			break


		#	for z in range(len(z_bins)-1):
		#		if ((c_z >= z_bins[z]) and (c_z <= z_bins[z+1])):
		#			c=z
		#			break

		#	bin_indices_pqr.append([a,b,c])






		#for row in range(len(bin_indices_pqr)):

		#	l=bin_indices_pqr[row]

		#	i=l[0]
		#	j=l[1]
		#	k=l[2]

		#	charge_tensor[i, j, k] += charge[row]
		#	radius_tensor[i, j, k] += radius[row]
		#	N_tensor_pqr[i, j, k] += 1

		#charge_tensor = charge_tensor #no average (charges sum up)
		#radius_tensor = radius_tensor/N_tensor_pqr #average electrostatic radius of the bin
		#radius_tensor = numpy.nan_to_num(radius_tensor)

		#print(xyz_pqr)
		#numpy.savetxt(path_folder+"for_labview"+p+"/xyz_pqr.txt", xyz_pqr, fmt="%1.8f")
		#numpy.savetxt(path_folder+"for_labview"+p+"/radius.txt", radius, fmt="%1.8f")
		#numpy.savetxt(path_folder+"for_labview"+p+"/charge.txt", charge, fmt="%1.8f")

		#numpy.save(path_folder+"for_labview"+p+"/N_tensor_pqr.npy", N_tensor_pqr)
		#numpy.save(path_folder+"for_labview"+p+"/charge_tensor.npy", charge_tensor)
		#numpy.save(path_folder+"for_labview"+p+"/radius_tensor.npy", radius_tensor)
		
		nn+=1
		print(round(nn/len(folders)*100,4), " percent completed")
		print()
		check.write(conf_path+"\n")
		check.flush()


check.close()


#next is only to plot
'''

r=numpy.array(bfactors_tensor)
print(numpy.min(r), numpy.max(r))
r=r-numpy.min(r)
r=r/numpy.max(r)

to_generate = numpy.array(r)

g=(to_generate)
r=1 / (1+ (to_generate/(1-to_generate))**(-2)  )
b=(1-to_generate)



#r=numpy.array(r)
#r=numpy.column_stack((r,r,r))
colors=numpy.stack((r,g,b), axis=-1) #NICE ONE	
print(colors.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#print(numpy.max(H))
ax.voxels(bfactors_tensor,facecolors=colors, alpha=0.8)
ax.axis('off')
plt.show()
plt.close("all")





from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_pqr[:,0], xyz_pqr[:,1], xyz_pqr[:,2], c=charge, marker='o', alpha=0.5, cmap="magma")
ax.axis('off')
plt.show()
plt.close("all")
'''

'''
H, edges = numpy.histogramdd(xyz, bins=(nbins-1), range=[(axis_min, axis_max),(axis_min, axis_max),(axis_min, axis_max)])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#print(numpy.max(H))
ax.voxels(H, edgecolors='black', alpha=0.3)
plt.show()
plt.close("all")
'''






'''
numpy.save("H.npy", H)#.reshape(nbins,nbins), fmt="%s")
#numpy.savetxt("H.npy", H)
H=numpy.load("H.npy")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(numpy.max(H))
ax.voxels(H, edgecolors='black', alpha=0.3)
plt.show()
plt.close("all")


#for labview
#https://forums.ni.com/t5/LabVIEW/3d-surface-graph-Plotting-Multiple-Surfaces-ActiveX-Surface/td-p/1596700
'''



