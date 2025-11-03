import os
import shutil

# Path to the folder containing your label .txt files
label_folder = r"C:\Users\jaysu\Downloads\Compressed\Compressed\SpeedBump_Hump.v2i.yolov8\train\labels"

# Optional: create a backup folder before changing anything
backup_folder = os.path.join(label_folder, "backup_labels")
os.makedirs(backup_folder, exist_ok=True)

for label_file in os.listdir(label_folder):
    if label_file.endswith(".txt"):
        label_path = os.path.join(label_folder, label_file)
        backup_path = os.path.join(backup_folder, label_file)

        # Backup the original file
        shutil.copy(label_path, backup_path)

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                parts[0] = "1"  # change class ID to 1 for all lines
                new_lines.append(" ".join(parts) + "\n")

        with open(label_path, "w") as f:
            f.writelines(new_lines)

print("✅ Done! All label IDs changed to 1 (backup saved in 'backup_labels').")
