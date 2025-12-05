import os
import shutil

# Source directory containing all the subfolders
source_root = "/mnt/slowdisk/public/HIPPA"

# Target directory
target_root = "dataset"

# Mapping from class number to classes
batch_to_class = {
    0: "Control", 1: "Colletotrichum godetiae", 2: "Fusarium avenaceum", 3: "Botrytis cinera", 4: "Penicillium expansum",
    5: "Phlyctema vagabunda", 6: "Alternaria alternata", 7: "Cadophora luteo-olivacea", 8: "Mucor piriformis"
}

# Make sure the target directory exists
os.makedirs(target_root, exist_ok=True)

# Walk through all subdirectories
for subdir, _, files in os.walk(source_root):
    for filename in files:
        if filename.endswith(".tif"):
            parts = filename.split("_")
            parts = filename.split("_")
            if len(parts) >= 4:
                print(parts)
                print(filename)
                class_number = int(parts[1][2])
                apple_type = batch_to_class.get(class_number, "Unknown")
                
                # Destination folder
                class_folder = os.path.join(target_root, f"class_{class_number}")
                os.makedirs(class_folder, exist_ok=True)
                
                # Source and destination paths
                src_file = os.path.join(subdir, filename)
                dst_file = os.path.join(class_folder, filename)
                
                shutil.copy2(src_file, dst_file)
                print(f"Moved {filename} -> {class_folder}")
