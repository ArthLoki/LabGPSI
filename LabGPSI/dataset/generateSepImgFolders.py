import os
import shutil

from ProjetoRedesNeurais.LabGPSI.auxiliary.globalVariables import image_path, current_path

lab_path = os.path.join(os.path.join(current_path.split('/')[0], '/'), '/'.join(current_path.split('/')[1:-1]))


# Define the source and destination directories
source_dir = image_path
og_destination_dir = lab_path + '/og_images'
nog_destination_dir = lab_path + '/nog_images'

# Create destination directories if they don't exist
os.makedirs(og_destination_dir, exist_ok=True)
os.makedirs(nog_destination_dir, exist_ok=True)

# Iterate over files in the source directory
for filename in os.listdir(source_dir):
    # Check if the filename contains 'og_'
    if 'og_' in filename:
        # Move the file to the OG images destination directory
        shutil.move(os.path.join(source_dir, filename), os.path.join(og_destination_dir, filename))
    else:
        # Move the file to the other images destination directory
        shutil.move(os.path.join(source_dir, filename), os.path.join(nog_destination_dir, filename))

print("Images separated successfully.")
