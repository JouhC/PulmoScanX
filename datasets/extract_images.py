import os
import tarfile

# Ensure output directory exists
output_dir = "cxr"
os.makedirs(output_dir, exist_ok=True)

# Extract all .tar.gz files in current directory
for file in os.listdir():
    if file.endswith(".tar.gz"):
        print(f"Extracting {file}...")
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"Extracted {file} to {output_dir}")
print("Extraction complete.")

