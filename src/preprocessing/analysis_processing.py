from __future__ import division
import numpy, os, mdtraj
import matplotlib.pyplot as plt
from pathlib import Path

#KEEP THE COMMENTS


#/usr/local/gromacs/bin/


#path = "10gs_protein.pdb"


#conf.save_pdb("aa.pdb")
#table, bonds = conf.topology.to_dataframe()
#print(table.tail())
silence = " > /dev/null 2>&1"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "config/checkpoints/checkpoint.txt"



#create a checkpoint file to restart the script from if it crashes
os.system(f"touch {CHECKPOINT_PATH}")

refined_set="refined-set/"
folders_list = os.listdir(refined_set)

chck=open(CHECKPOINT_PATH, "r")
data_ck=chck.readlines()
chck.close()

check=open(CHECKPOINT_PATH, "a")

n=len(data_ck)


#remove everything eventually
delete_all_files = False

if delete_all_files==True:

	for i in folders_list:
		files=os.listdir(refined_set+i)

		input_path = refined_set+i+"/"

		#print("rm "+input_path.rstrip(".pdb")+"_clean*")
		os.system("rm "+input_path.rstrip(".pdb")+"_clean*"+silence) #delete all files containing "clean*"

		n+=1

		if n%100==0:

			print(round(n/len(folders_list)*100, 3), end='\r'  )

print()





n=len(data_ck)

for i in folders_list:
	files=os.listdir(refined_set+i)

	for j in files:

		exist=False
		for l in data_ck:
			if j in l:
				exist=True

		if (("protein" in j) or ("pocket" in j)) and ("clean" not in j) and (exist==False): #apply the analysis on the pocket.pdb and protein.pdb


			input_path = refined_set+i+"/"+j
			path = input_path.rstrip(".pdb")+"_cleaned.pdb"  #new output name

			#print("rm "+input_path.rstrip(".pdb")+"_clean*")
			os.system("rm "+input_path.rstrip(".pdb")+"_clean*"+silence) #remove the old files

			#clean with pymol
			comando = "pymol -c "+str(SCRIPT_DIR / "pymol_cleaning.py")+" "+input_path +silence #remove hetatm, remove H and readd them
			#print(comando)
			os.system(comando)


			#path=path.rstrip(".pdb")+"_cleaned.pdb"
			conf = mdtraj.load(path)

			#print(path)
			print(conf.n_atoms, " atoms")
			print(conf.n_residues, " residues")

			if conf.n_atoms > 50000: #se la proteina troppo grande ci impiega una vita
				print("too big")
				print(path)
				print()
				continue
			

			#calculate sasa for each atom in the system
			sasa = mdtraj.shrake_rupley(conf) #https://mdtraj.org/1.9.4/api/generated/mdtraj.shrake_rupley.html
			#print(sasa)
			numpy.savetxt(path.rstrip("d.pdb")+"d_sasa.txt", sasa[0]) 


			#calculate sec struct for each residue (not each atom)
			dssp = mdtraj.compute_dssp(conf, simplified=True) #https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html
			#print(dssp)
			numpy.savetxt(path.rstrip("d.pdb")+"d_dssp.txt",dssp[0], fmt="%s")


			#hbonds = mdtraj.baker_hubbard(conf) #https://mdtraj.org/1.9.4/api/generated/mdtraj.baker_hubbard.html
			#compute hbonds, output is indexes of donor, hydrogen, acceptor
			#this only calculates the hydrogen bonds already occurring on the protein, not the potential donor/acceptor sites (the latter is more suited for pocket prediction in our case)
			#print(hbonds)


			#calculate the buriedness of each atom (i.e. number of neighbors in a sphere of 10 A)
			atom_indices = conf.topology.select('all')
			buriedness=[]
			for ii in atom_indices:
				b=mdtraj.compute_neighbors(conf,1, [ii])
				buriedness.append(len(b[0])) #cutoff in nanometers; one atom index at a time and returns the number of neighbors of that atom
			numpy.savetxt(path.rstrip("d.pdb")+"d_buriedness.txt", buriedness)


			#in the pqr file, the columns are partial charge (units of electron) and atomic radius
			os.system("pdb2pqr --ff=amber "+path+" "+path.rstrip("pdb")+"pqr"+silence) #saves the output


			#b factor: measure of thermal motion of an atom in a system
			os.system("pymol -c "+str(SCRIPT_DIR / "pymol_bfactors.py")+" "+path+silence) #saves the output



			#H-bonds:
			#Donors provide the hydrogen atoms. The donor in a hydrogen bond is usually a strongly electronegative atom such as N, O, or S that is covalently bonded to a hydrogen bond
			#acceptors provide the electrons. The hydrogen acceptor is an electronegative atom of a neighboring molecule or ion that contains a lone pair that participates in the hydrogen bond, again N,O,S

			#calculate donor indices
			os.system("pymol -c "+str(SCRIPT_DIR / "pymol_hdonors.py")+" "+path+silence) #saves the output


			#calculate acceptor indices
			os.system("pymol -c "+str(SCRIPT_DIR / "pymol_hacceptors.py")+" "+path+silence) #saves the output

			#need to output coordinates and indices (correct indices )


			#CHECK IF FILE ARE COMPLETE?



			n+=1
			print(path, round(n/2/len(folders_list)*100, 3)  )
			print()
			

			check.write(j+"\n")
			check.flush()

check.close()







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



