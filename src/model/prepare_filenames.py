from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

f=open(PROJECT_ROOT / "config/checkpoints/checkpoint_tensor_cnn.txt", "r")
files=f.readlines()
f.close()

out=open(PROJECT_ROOT / "config/filepaths.txt", "w")

for i in files:
	if "protein" in i:
		path=i.rstrip(i.split("/")[-1])

		out.write(path+"\n")

out.close()

