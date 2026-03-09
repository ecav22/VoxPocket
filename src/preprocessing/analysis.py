from __future__ import division
import numpy, os, mdtraj
import matplotlib.pyplot as plt
from pathlib import Path

#KEEP THE COMMENTS


#/usr/local/gromacs/bin/

SCRIPT_DIR = Path(__file__).resolve().parent


path = "10gs_protein.pdb"


#conf.save_pdb("aa.pdb")
#table, bonds = conf.topology.to_dataframe()
#print(table.tail())
silence = " > /dev/null 2>&1"

comando = "pymol -c "+str(SCRIPT_DIR / "pymol_cleaning.py")+" "+path +silence
#print(comando)
os.system(comando)


path=path.rstrip(".pdb")+"_cleaned.pdb"
conf = mdtraj.load(path)



#conf.center_coordinates() #center coordinates at 0,0,0

print(path)
print(conf.n_atoms, " atoms")
print(conf.n_residues, " residues")
print()

#calculate sasa for each atom in the system
sasa = mdtraj.shrake_rupley(conf) #https://mdtraj.org/1.9.4/api/generated/mdtraj.shrake_rupley.html
#print(sasa)


#calculate sec struct for each residue (not each atom)
dssp = mdtraj.compute_dssp(conf, simplified=True) #https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html
#print(dssp)


#hbonds = mdtraj.baker_hubbard(conf) #https://mdtraj.org/1.9.4/api/generated/mdtraj.baker_hubbard.html
#compute hbonds, output is indexes of donor, hydrogen, acceptor
#have to divide between donor and acceptor?
#print(hbonds)

atom_indices = conf.topology.select('all')
buriedness=[]
for i in atom_indices:
	b=mdtraj.compute_neighbors(conf,1, [i])
	buriedness.append(len(b[0])) #cutoff in nanometers; one atom index at a time and returns the number of neighbors of that atom
	

#in the pqr file, the columns are partial charge (units of electron) and atomic radius
os.system("pdb2pqr --ff=amber "+path+" "+path.rstrip("pdb")+"pqr"+silence)


#b factor: measure of thermal motion of an atom in a system
os.system("pymol -c "+str(SCRIPT_DIR / "pymol_bfactors.py")+" "+path+silence)



#H-bonds:
#Donors provide the hydrogen atoms. The donor in a hydrogen bond is usually a strongly electronegative atom such as N, O, or S that is covalently bonded to a hydrogen bond
#acceptors provide the electrons. The hydrogen acceptor is an electronegative atom of a neighboring molecule or ion that contains a lone pair that participates in the hydrogen bond, again N,O,S

#calculate donor indices
os.system("pymol -c "+str(SCRIPT_DIR / "pymol_hdonors.py")+" "+path+silence)


#calculate acceptor indices
os.system("pymol -c "+str(SCRIPT_DIR / "pymol_hdonors.py")+" "+path+silence)











'''
axis_min = numpy.min(conf.xyz[0])
axis_max = numpy.max(conf.xyz[0])

x=conf.xyz[0][:,0]
y=conf.xyz[0][:,1]
z=conf.xyz[0][:,2]

nbins = 20

array = numpy.array(conf.xyz[0])







H, edges = numpy.histogramdd(array, bins=nbins, range=[(axis_min, axis_max),(axis_min, axis_max),(axis_min, axis_max)])








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



