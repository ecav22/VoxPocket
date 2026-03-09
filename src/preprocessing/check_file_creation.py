import os
from termcolor import colored
from pathlib import Path



def check_file_created(folder, file_name, feature_name):

	if file_name.split("/")[-1] in os.listdir(folder):

		file_stats = os.stat(file_name)
		file_size = file_stats.st_size
		count=0

		#if file_size > 1:
		#	print(feature_name, str(file_size)+" Bytes")

	if file_name.split("/")[-1] not in os.listdir(folder) or file_size<1:
		print(file_name.split("/")[-1])
		print("no "+feature_name+" in "+folder)
		print()
		count=1
		#command="mv "+folder+" removed_pdbs/"
		#os.system(command)
		
	return count


CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "config/checkpoints/checkpoint.txt"

f=open(CHECKPOINT_PATH)
righe=f.readlines()
f.close()

refined_set="refined-set/"

lista=os.listdir(refined_set)

N=0

for i in lista:

	
		path_folder=refined_set+ i#.split("_")[0]+"/"
		#path=path_folder+i.rstrip(".pdb\n")

		#path_bfac = path+"_cleaned_bfactors.txt"
		#path_buried = path+"_cleaned_buriedness.txt"
		#path_dssp = path+"_cleaned_dssp.txt"
		#path_hbac = path+"_cleaned_hbacceptors.txt"
		#path_hbdon = path+"_cleaned_hbdonors.txt"
		#path_sasa = path+"_cleaned_sasa.txt"
		path_pqr = path_folder+"/"+i+"_protein_cleaned.pqr"

		#print(i.rstrip("\n"))
		#count= check_file_created(path_folder, path_bfac, "bfactor")
		#count=check_file_created(path_folder, path_buried, "buriedness")
		#count=check_file_created(path_folder, path_dssp, "dssp")
		#count=check_file_created(path_folder, path_hbac, "hbond acc")
		#count=check_file_created(path_folder, path_hbdon, "hbond don")
		#count=check_file_created(path_folder, path_sasa, "sasa")
		count=check_file_created(path_folder, path_pqr, "pqr")
		N+=count





print()
print(N, "pqr missing")
print(len(righe), "total files")



#eventually fix with modeller and the script to identify the missing residues
