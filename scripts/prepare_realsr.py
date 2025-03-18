import os
import shutil

# Specify your path here
base_dir_list = [
"./data/RealSR(V3)/Nikon/Test/4/",
"./data/RealSR(V3)/Canon/Test/4"
]
hq_dir = "./data/RealSR/hq"
lq_dir = "./data/RealSR/lq"

# create
os.makedirs(hq_dir, exist_ok=True)
os.makedirs(lq_dir, exist_ok=True)

# go through each file
for base_dir in base_dir_list:
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)

        # make sure it's a file
        if os.path.isfile(file_path):
            if "HR" in filename:
                new_filename = filename.replace('_HR', '')
                shutil.copy(file_path, os.path.join(hq_dir, new_filename))
                print(f"Copied: {filename} -> {hq_dir}")
            elif "LR" in filename:
                new_filename = filename.replace('_LR4', '')
                shutil.copy(file_path, os.path.join(lq_dir, new_filename))
                print(f"Copied: {filename} -> {lq_dir}")

print("Completed")
