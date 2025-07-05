import os
import tarfile
import shutil

# Directory containing the .tar.gz files and extracted_images folder
base_dir = os.path.dirname(os.path.abspath(__file__))
extracted_dir = os.path.join(base_dir, 'extracted_images')

# Ensure the extracted_images directory exists
os.makedirs(extracted_dir, exist_ok=True)

# List all .tar.gz files in the current directory
archives = [f for f in os.listdir(base_dir) if f.endswith('.tar.gz')]

for archive in sorted(archives):
    print(f"Processing {archive}...")
    archive_path = os.path.join(base_dir, archive)
    # Extract to a temporary folder
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Get the top-level folder name (should be 'images' or similar)
        top_folder = tar.getnames()[0].split('/')[0]
        tar.extractall(path=base_dir)
    extracted_folder = os.path.join(base_dir, top_folder)
    # Move all files from extracted_folder to extracted_images
    for fname in os.listdir(extracted_folder):
        src = os.path.join(extracted_folder, fname)
        dst = os.path.join(extracted_dir, fname)
        if os.path.isfile(src):
            shutil.move(src, dst)
    # Delete the now-empty extracted folder
    shutil.rmtree(extracted_folder)
    print(f"Finished {archive}. Images moved to {extracted_dir} and {top_folder} deleted.")

print("All archives processed.") 