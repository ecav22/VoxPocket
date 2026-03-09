from __future__ import division
import os, numpy, mdtraj, scipy.stats
import matplotlib.pyplot as plt


#pqr and pdb files are slightly different because of Amber ff format etc.... and sometimes different number of atoms
#have to re-obtain all the coordinates and atom names to re-map the charges and radiuses
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

			x_pqr.append(float(i.split()[-5]))
			y_pqr.append(float(i.split()[-4]))
			z_pqr.append(float(i.split()[-3]))

			charge.append(float(i.split()[-2]))
			radius.append(float(i.split()[-1]))

	xyz_pqr=numpy.column_stack((x_pqr, y_pqr, z_pqr))

	return xyz_pqr, charge, radius


path_folder = "2ovv/"

conf_path =path_folder+"2ovv_protein_cleaned.pdb"
conf = mdtraj.load(conf_path)

#print(path)
print(conf.n_atoms, " atoms")
print(conf.n_residues, " residues")

#
xyz = conf.xyz[0]
bfactors = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_bfactors.txt")
buriedness = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_buriedness.txt")
dssp = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_dssp.txt", dtype='str')
hbac = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_hbacceptors.txt")
hbdon =numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_hbdonors.txt")
sasa = numpy.loadtxt(conf_path.rstrip("d.pdb")+"d_sasa.txt")
xyz_pqr, charge, radius = open_pqr(conf_path.rstrip("pdb")+"pqr")


nbins=20 #number of bins of the 3D tensor (go for 50 or 100)

#size of the box
axis_min = numpy.min(xyz)
axis_max = numpy.max(xyz)

axis=numpy.linspace(axis_min, axis_max, nbins)


N_tensor = numpy.zeros((nbins,nbins,nbins))
#xyz_tensor = numpy.zeros(nbins,nbins,nbins) #this is just the N_tensor
bfactors_tensor = numpy.zeros((nbins,nbins,nbins))
buriedness_tensor = numpy.zeros((nbins,nbins,nbins))
#dssp_tensor = numpy.zeros((nbins,nbins,nbins)) #not very useful tbh
hbac_tensor = numpy.zeros((nbins,nbins,nbins))
hbdon_tensor = numpy.zeros((nbins,nbins,nbins))
sasa_tensor = numpy.zeros((nbins,nbins,nbins))
charge_tensor = numpy.zeros((nbins,nbins,nbins))
radius_tensor = numpy.zeros((nbins,nbins,nbins))

bin_indices = numpy.digitize(xyz, axis) - 1 #note the -1 is because the output of digitize is 1-based (starts from 1) and I want it 0-based
#this is because the value 0 is reserved for the values outside of the bins


for row in range(len(bin_indices)):

	l=bin_indices[row]

	i=l[0]
	j=l[1]
	k=l[2]

	N_tensor[i, j, k] += 1
	bfactors_tensor[i, j, k] += bfactors[row]
	buriedness_tensor[i, j, k] += buriedness[row]
	sasa_tensor[i, j, k] += sasa[row]

print(sasa_tensor)

bfactors_tensor = bfactors_tensor/N_tensor #average bfactor of the bins
bfactors_tensor = numpy.nan_to_num(bfactors_tensor) #remove nans (change to 0)
buriedness_tensor = buriedness_tensor/N_tensor #average in the bins (aka average number of neighbors)
buriedness_tensor = numpy.nan_to_num(buriedness_tensor)
sasa_tensor = sasa_tensor/N_tensor #average sasa of the bin
sasa_tensor = numpy.nan_to_num(sasa_tensor)


for i in hbac: #count the hbdon

	l=bin_indices[int(i)]

	i=l[0]
	j=l[1]
	k=l[2]

	hbac_tensor[i,j,k] += 1


for i in hbdon: #count the hbac

	l=bin_indices[int(i)]

	i=l[0]
	j=l[1]
	k=l[2]

	hbdon_tensor[i,j,k]+= 1

hbac_tensor=hbac_tensor #no average, just sum
hbdon_tensor=hbdon_tensor #no average, just sum




#### outouts for LabVIEW
os.system("rm -r "+path_folder+"for_labview")
os.system("mkdir "+path_folder+"for_labview")
numpy.savetxt(path_folder+"for_labview/xyz.txt", xyz, fmt="%1.8f")
numpy.savetxt(path_folder+"for_labview/bfactors.txt", bfactors, fmt="%1.8f")
numpy.savetxt(path_folder+"for_labview/buriedness.txt", buriedness, fmt="%i")
numpy.savetxt(path_folder+"for_labview/sasa.txt", sasa, fmt="%1.8f")


numpy.save(path_folder+"for_labview/N_tensor.npy", N_tensor)
numpy.save(path_folder+"for_labview/bfactors_tensor.npy", bfactors_tensor)
numpy.save(path_folder+"for_labview/buriedness_tensor.npy", buriedness_tensor)
numpy.save(path_folder+"for_labview/sasa_tensor.npy", sasa_tensor)

numpy.save(path_folder+"for_labview/hbac_tensor.npy", hbac_tensor)
numpy.save(path_folder+"for_labview/hbdon_tensor.npy", hbdon_tensor)


hbac_list=[]
for i in range(len(buriedness)):
	if i in hbac:
		hbac_list.append(1)
	else:
		hbac_list.append(0)

hbdon_list=[]
for i in range(len(buriedness)):
	if i in hbdon:
		hbdon_list.append(1)
	else:
		hbdon_list.append(0)

hbac_list=numpy.array(hbac_list)
hbdon_list=numpy.array(hbdon_list)

numpy.savetxt(path_folder+"for_labview/hbac.txt", hbac_list, fmt="%i")
numpy.savetxt(path_folder+"for_labview/hbdon.txt", hbdon_list, fmt="%i")



xyz_pqr = xyz_pqr/10
#### have to compute charge and radius differently, using the coordinates from the pqr because pdb2pqr added a few hydrogens
bin_indices_pqr = numpy.digitize(xyz_pqr, axis) - 1
N_tensor_pqr = numpy.zeros((nbins,nbins,nbins))

for row in range(len(bin_indices_pqr)):

	l=bin_indices_pqr[row]

	i=l[0]
	j=l[1]
	k=l[2]

	charge_tensor[i, j, k] += charge[row]
	radius_tensor[i, j, k] += radius[row]
	N_tensor_pqr[i, j, k] += 1

charge_tensor = charge_tensor #no average (charges sum up)
radius_tensor = radius_tensor/N_tensor_pqr #average electrostatic radius of the bin
radius_tensor = numpy.nan_to_num(radius_tensor)

numpy.savetxt(path_folder+"for_labview/xyz_pqr.txt", xyz_pqr, fmt="%1.8f")
numpy.savetxt(path_folder+"for_labview/radius.txt", radius, fmt="%1.8f")
numpy.savetxt(path_folder+"for_labview/charge.txt", charge, fmt="%1.8f")

numpy.save(path_folder+"for_labview/N_tensor_pqr.npy", N_tensor_pqr)
numpy.save(path_folder+"for_labview/charge_tensor.npy", charge_tensor)
numpy.save(path_folder+"for_labview/radius_tensor.npy", radius_tensor)






#next is only to plot


r=numpy.array(charge_tensor)
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
ax.voxels(charge_tensor,facecolors=colors, alpha=0.8)
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



