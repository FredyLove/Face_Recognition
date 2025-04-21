import tarfile
import os

# Path to your archive
archive_path = "lfw-funneled.tgz"
extract_path = "dataset"


# Create the dataset directory if it doesn't exist
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

# Extracting the .tgz file
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

print(f"Dataset extracted to {extract_path}")
